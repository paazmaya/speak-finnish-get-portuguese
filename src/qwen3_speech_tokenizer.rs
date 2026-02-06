use anyhow::Result;
use candle_core::{DType, IndexOp, Module, Tensor, D};
use candle_nn::{conv1d, conv_transpose1d, linear, linear_no_bias, ops::softmax, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, VarBuilder};
use serde::Deserialize;

const DTYPE: DType = DType::F32;

// SnakeBeta activation function: x + (1/beta) * sin^2(alpha * x)
#[allow(dead_code)]
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn new(in_features: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(in_features, "alpha")?;
        let beta = vb.get(in_features, "beta")?;
        Ok(Self { alpha, beta })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Shape: (batch, channels, time)
        // alpha/beta: (channels,) -> (1, channels, 1)
        let alpha = self.alpha.exp()?.unsqueeze(0)?.unsqueeze(2)?;
        let beta = self.beta.exp()?.unsqueeze(0)?.unsqueeze(2)?;
        
        let alpha_x = x.broadcast_mul(&alpha)?;
        let sin_term = alpha_x.sin()?.sqr()?;
        let beta_inv = (beta + 1e-9)?.recip()?;
        let result = (x + sin_term.broadcast_mul(&beta_inv)?)?;
        
        Ok(result)
    }
}

// Causal Conv1d with appropriate padding
#[allow(dead_code)]
struct CausalConv1d {
    conv: Conv1d,
    padding: usize,
    stride: usize,
}

impl CausalConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = Conv1dConfig {
            stride,
            dilation,
            groups,
            padding: 0,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(in_channels, out_channels, kernel_size, config, vb.pp("conv"))?;
        let kernel_size_dilated = (kernel_size - 1) * dilation + 1;
        let padding = kernel_size_dilated - stride;
        
        Ok(Self { conv, padding, stride })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pad on the left for causal convolution
        let (batch, channels, length) = x.dims3()?;
        let kernel_size_dilated = self.padding + self.stride;
        let n_frames = ((length + self.padding - kernel_size_dilated) as f32 / self.stride as f32 + 1.0).ceil() as usize;
        let ideal_length = (n_frames - 1) * self.stride + kernel_size_dilated - self.padding;
        let extra_padding = ideal_length.saturating_sub(length);
        
        let padded = if self.padding > 0 || extra_padding > 0 {
            let mut data = x.to_vec3::<f32>()?;
            for b in 0..batch {
                for c in 0..channels {
                    let mut row = vec![0.0; self.padding];
                    row.extend_from_slice(&data[b][c]);
                    row.resize(row.len() + extra_padding, 0.0);
                    data[b][c] = row;
                }
            }
            let new_len = length + self.padding + extra_padding;
            Tensor::from_vec(
                data.into_iter().flatten().flatten().collect::<Vec<_>>(),
                (batch, channels, new_len),
                x.device()
            )?
        } else {
            x.clone()
        };
        
        Ok(self.conv.forward(&padded)?)
    }
}

// Causal Transpose Conv1d for upsampling
#[allow(dead_code)]
struct CausalTransposeConv1d {
    conv: ConvTranspose1d,
    left_pad: usize,
    right_pad: usize,
}

impl CausalTransposeConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            stride,
            padding: 0,
            ..Default::default()
        };
        let conv = conv_transpose1d(in_channels, out_channels, kernel_size, config, vb.pp("conv"))?;
        let pad = kernel_size - stride;
        let left_pad = (pad as f32 / 2.0).ceil() as usize;
        let right_pad = pad - left_pad;
        
        Ok(Self { conv, left_pad, right_pad })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output = self.conv.forward(x)?;
        let length = output.dim(D::Minus1)?;
        if self.left_pad + self.right_pad >= length {
            return Ok(output);
        }
        Ok(output.narrow(D::Minus1, self.left_pad, length - self.left_pad - self.right_pad)?)
    }
}

// ConvNeXt Block
#[allow(dead_code)]
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: candle_nn::LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = CausalConv1d::new(dim, dim, 7, 1, 1, dim, vb.pp("dwconv"))?;
        let norm = candle_nn::layer_norm(dim, candle_nn::LayerNormConfig::default(), vb.pp("norm"))?;
        let pwconv1 = linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        
        Ok(Self { dwconv, norm, pwconv1, pwconv2, gamma })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input = x.clone();
        let mut hidden = self.dwconv.forward(x)?;
        
        // Permute (B, C, T) -> (B, T, C) for layer norm
        hidden = hidden.transpose(1, 2)?;
        hidden = self.norm.forward(&hidden)?;
        hidden = self.pwconv1.forward(&hidden)?.gelu()?;
        hidden = self.pwconv2.forward(&hidden)?;
        
        // Apply gamma scaling
        hidden = hidden.broadcast_mul(&self.gamma.unsqueeze(0)?.unsqueeze(0)?)?;
        
        // Permute back (B, T, C) -> (B, C, T)
        hidden = hidden.transpose(1, 2)?;
        
        Ok((input + hidden)?)
    }
}

// Residual unit with dilated convolutions
#[allow(dead_code)]
struct DecoderResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl DecoderResidualUnit {
    fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(dim, vb.pp("act1"))?,
            conv1: CausalConv1d::new(dim, dim, 7, 1, dilation, 1, vb.pp("conv1"))?,
            act2: SnakeBeta::new(dim, vb.pp("act2"))?,
            conv2: CausalConv1d::new(dim, dim, 1, 1, 1, 1, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let hidden = self.act1.forward(x)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self.act2.forward(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;
        Ok((hidden + residual)?)
    }
}

// Decoder block with upsampling
#[allow(dead_code)]
struct DecoderBlock {
    initial_act: SnakeBeta,
    upsample: CausalTransposeConv1d,
    residual_units: Vec<DecoderResidualUnit>,
}

impl DecoderBlock {
    fn new(
        in_dim: usize,
        out_dim: usize,
        upsample_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let initial_act = SnakeBeta::new(in_dim, vb.pp("block.0"))?;
        let upsample = CausalTransposeConv1d::new(
            in_dim,
            out_dim,
            2 * upsample_rate,
            upsample_rate,
            vb.pp("block.1"),
        )?;
        
        let mut residual_units = Vec::new();
        for i in 2..5 {  // blocks 2, 3, 4
            residual_units.push(DecoderResidualUnit::new(
                out_dim,
                1,  // dilation = 1 for all
                vb.pp(&format!("block.{}", i)),
            )?);
        }
        
        Ok(Self {
            initial_act,
            upsample,
            residual_units,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = self.initial_act.forward(x)?;
        hidden = self.upsample.forward(&hidden)?;
        
        for unit in &self.residual_units {
            hidden = unit.forward(&hidden)?;
        }
        
        Ok(hidden)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TTSTokenizerConfig {
    pub decoder_config: DecoderConfig,
    pub encoder_config: EncoderConfig,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub codebook_dim: usize,
    pub codebook_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub decoder_dim: usize,
    pub latent_dim: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub num_quantizers: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
    pub upsample_rates: Vec<usize>,
    pub hidden_act: String,
}

// RMSNorm for decoder
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        Ok(x_normed.broadcast_mul(&self.weight)?)
    }
}

// Simplified decoder - loads RVQ codebooks and projects to waveform
pub struct Qwen3SpeechTokenizerDecoder {
    // RVQ codebook embeddings (16 layers)
    codebook_embeddings: Vec<Tensor>,
    // RVQ output projection (256->512)
    rvq_output_proj: Option<Conv1d>,
    // Pre-conv projects from 512 to 1024
    pre_conv: Option<Conv1d>,
    // Pre-transformer processes codebook embeddings
    pre_transformer: Option<PreTransformer>,
    // Initial decoder conv (1024->1536)
    initial_conv: Option<Conv1d>,
    // Decoder blocks for upsampling
    decoder_blocks: Option<Vec<DecoderBlock>>,
    // Final activation and conv
    final_act: Option<SnakeBeta>,
    final_conv: Option<Conv1d>,
    // Upsampling blocks with ConvNeXt
    upsample_blocks: Option<Vec<(Conv1d, ConvNeXtBlock)>>,
    // Final projection to waveform (unused in this architecture)
    output_proj: Option<Linear>,
    config: DecoderConfig,
    output_sample_rate: usize,
    decode_upsample_rate: usize,
}

// Pre-transformer with 8 layers
struct PreTransformer {
    input_proj: Linear,
    layers: Vec<TransformerLayer>,
    norm: RmsNorm,
    output_proj: Linear,
}

struct TransformerLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Attention {
    fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let total_head_dim = config.num_attention_heads * config.head_dim;
        
        Ok(Self {
            q_proj: linear_no_bias(hidden_size, total_head_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(hidden_size, config.num_key_value_heads * config.head_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(hidden_size, config.num_key_value_heads * config.head_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(total_head_dim, hidden_size, vb.pp("o_proj"))?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        
        let q = self.q_proj.forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Attention scores
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        
        Ok(self.o_proj.forward(&attn_output)?)
    }
}

impl Mlp {
    fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

impl TransformerLayer {
    fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?,
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            post_attention_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::new(config, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states)?;
        let hidden_states = (hidden_states + residual)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        Ok((hidden_states + residual)?)
    }
}

impl PreTransformer {
    fn new(config: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let input_proj = linear(config.latent_dim, config.hidden_size, vb.pp("input_proj"))?;
        
        let mut layers = Vec::new();
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(TransformerLayer::new(config, vb_layers.pp(&i.to_string()))?);
        }
        
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let output_proj = linear(config.hidden_size, config.latent_dim, vb.pp("output_proj"))?;
        
        Ok(Self {
            input_proj,
            layers,
            norm,
            output_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch, latent_dim, seq] -> transpose to [batch, seq, latent_dim]
        let x = x.transpose(1, 2)?;
        let mut hidden_states = self.input_proj.forward(&x)?;
        
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        // Apply final norm and output projection
        hidden_states = self.norm.forward(&hidden_states)?;
        hidden_states = self.output_proj.forward(&hidden_states)?;
        
        // Transpose back: [batch, seq, latent_dim] -> [batch, latent_dim, seq]
        Ok(hidden_states.transpose(1, 2)?)
    }
}

impl Qwen3SpeechTokenizerDecoder {
    /// Create a placeholder decoder without loading weights
    pub fn placeholder(config: &Qwen3TTSTokenizerConfig) -> Result<Self> {
        Ok(Self {
            codebook_embeddings: Vec::new(),
            rvq_output_proj: None,
            pre_conv: None,
            pre_transformer: None,
            initial_conv: None,
            decoder_blocks: None,
            final_act: None,
            final_conv: None,
            upsample_blocks: None,
            output_proj: None,
            config: config.decoder_config.clone(),
            output_sample_rate: config.output_sample_rate,
            decode_upsample_rate: config.decode_upsample_rate,
        })
    }

    pub fn load(vb: VarBuilder, config: &Qwen3TTSTokenizerConfig) -> Result<Self> {
        let decoder_config = &config.decoder_config;

        println!("Loading RVQ codebook embeddings...");
        // Load RVQ codebook embeddings (16 layers total)
        let mut codebook_embeddings = Vec::new();
        let codebook_dim = config.encoder_config.codebook_dim;
        let codebook_size = config.encoder_config.codebook_size;
        
        // Load first codebook (semantic)
        let first_codebook = vb.get(
            &[codebook_size, codebook_dim],
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"
        )?;
        codebook_embeddings.push(first_codebook);

        // Load remaining 15 codebooks (acoustic)
        for i in 0..15 {
            let codebook = vb.get(
                &[codebook_size, codebook_dim],
                &format!("decoder.quantizer.rvq_rest.vq.layers.{}._codebook.embedding_sum", i)
            )?;
            codebook_embeddings.push(codebook);
        }
        println!("✓ Loaded {} RVQ codebooks", codebook_embeddings.len());

        // Load RVQ output projection (256->512)
        println!("Loading RVQ output projection...");
        let rvq_output_proj = {
            let proj_vb = vb.pp("decoder.quantizer.rvq_rest.output_proj");
            match proj_vb.get((512, 256, 1), "weight") {
                Ok(weight) => {
                    // Create Conv1d without bias (bias = None)
                    let config = Conv1dConfig::default();
                    let conv = Conv1d::new(weight, None, config);
                    println!("✓ RVQ output projection loaded (256 -> 512, no bias)");
                    Some(conv)
                }
                Err(e) => {
                    println!("⚠ RVQ output projection weight not found: {}, will skip", e);
                    None
                }
            }
        };

        // Load pre-conv (Conv1d 512->1024, kernel=3)
        println!("Attempting to load pre-conv layer...");
        let pre_conv = match conv1d(
            512,
            1024,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("decoder.pre_conv.conv")
        ) {
            Ok(conv) => {
                println!("✓ Pre-conv loaded (512 -> 1024)");
                Some(conv)
            }
            Err(_) => {
                println!("⚠ Pre-conv not found, will skip");
                None
            }
        };

        // Load pre-transformer
        println!("Loading pre-transformer...");
        let pre_transformer = PreTransformer::new(
            decoder_config,
            vb.pp("decoder.pre_transformer")
        )?;
        println!("✓ Pre-transformer loaded");

        // Load decoder blocks
        println!("Loading decoder blocks...");
        let dims = vec![1536, 768, 384, 192, 96]; // From config analysis
        let upsample_rates = vec![8, 5, 4, 3]; // From config
        let mut decoder_blocks = Vec::new();
        
        // Initial conv: decoder.decoder.0.conv (1024->1536, kernel=7)
        let initial_conv = conv1d(
            1024, 1536, 7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("decoder.decoder.0.conv")
        );
        
        if initial_conv.is_ok() {
            println!("  ✓ Initial conv loaded (1024 -> 1536)");
        }
        
        // Blocks 1-4: Upsampling with residual units
        for i in 0..4 {
            match DecoderBlock::new(
                dims[i],
                dims[i + 1],
                upsample_rates[i],
                vb.pp(&format!("decoder.decoder.{}", i + 1))
            ) {
                Ok(block) => {
                    println!("  ✓ Decoder block {} loaded ({} -> {})", i + 1, dims[i], dims[i + 1]);
                    decoder_blocks.push(block);
                }
                Err(e) => {
                    println!("  ⚠ Decoder block {} failed: {}", i + 1, e);
                    return Err(e);
                }
            }
        }
        
        // Final activation: decoder.decoder.5
        let final_act = SnakeBeta::new(96, vb.pp("decoder.decoder.5")).ok();
        
        // Final conv: decoder.decoder.6.conv (96 -> 1, kernel=7)
        let final_conv = conv1d(
            96, 1, 7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("decoder.decoder.6.conv")
        ).ok();
        
        // Load upsample blocks (decoder.upsample.0 and decoder.upsample.1)
        println!("Loading upsample blocks...");
        let mut upsample_blocks = Vec::new();
        for i in 0..2 {
            let upsample_conv = conv1d(
                1024, 1024, 2,
                Conv1dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp(&format!("decoder.upsample.{}.0", i))
            ).ok();
            let convnext = ConvNeXtBlock::new(
                1024,
                vb.pp(&format!("decoder.upsample.{}.1", i))
            ).ok();
            
            if let (Some(conv), Some(block)) = (upsample_conv, convnext) {
                println!("  ✓ Upsample block {} loaded", i);
                upsample_blocks.push((conv, block));
            }
        }

        Ok(Self {
            codebook_embeddings,
            rvq_output_proj,
            pre_conv,
            pre_transformer: Some(pre_transformer),
            initial_conv: initial_conv.ok(),
            decoder_blocks: Some(decoder_blocks),
            final_act,
            final_conv,
            upsample_blocks: if upsample_blocks.is_empty() { None } else { Some(upsample_blocks) },
            output_proj: None,
            config: decoder_config.clone(),
            output_sample_rate: config.output_sample_rate,
            decode_upsample_rate: config.decode_upsample_rate,
        })
    }

    pub fn decode(&self, codec_tokens: &Tensor) -> Result<Vec<f32>> {
        use std::time::Instant;
        let decode_start = Instant::now();
        println!("Decoding with REAL decoder architecture...");
        
        // Get dimensions
        let dims = codec_tokens.dims();
        let (num_quantizers, seq_len) = if dims.len() == 2 {
            (dims[0], dims[1])
        } else if dims.len() == 3 {
            (dims[1], dims[2])
        } else {
            anyhow::bail!("Expected codec_tokens shape (num_quantizers, seq_len) or (batch, num_quantizers, seq_len), got {:?}", dims);
        };

        println!("  Processing {} tokens with {} quantizers", seq_len, num_quantizers);

        // Decode RVQ: Sum embeddings from all quantizers
        let codebook_dim = self.codebook_embeddings[0].dim(1)?;
        let num_codebooks = num_quantizers.min(self.codebook_embeddings.len());
        
        let device = &self.codebook_embeddings[0].device();
        
        // Collect embeddings from each quantizer layer and sum them (residual)
        let mut z: Option<Tensor> = None;
        for q in 0..num_codebooks {
            // Extract codes for this quantizer: shape [seq_len]
            let codes = if dims.len() == 2 {
                codec_tokens.i(q)?
            } else {
                codec_tokens.i((0, q))?
            };
            
            // Look up embeddings: [seq_len, codebook_dim]
            let emb = self.codebook_embeddings[q].embedding(&codes)?;
            
            // Accumulate (residual quantization)
            z = Some(match z {
                None => emb,
                Some(prev) => (prev + emb)?,
            });
        }
        
        let mut z = z.ok_or_else(|| anyhow::anyhow!("No codebooks processed"))?;
        println!("  ✓ RVQ decoded {} frames, shape {:?}", seq_len, z.dims());
        
        // Reshape for Conv1d: [batch=1, channels=codebook_dim, seq_len]
        z = z.t()?.unsqueeze(0)?;
        println!("  After transpose: {:?}", z.dims());
        
        // Apply RVQ output projection: 256 -> 512
        if let Some(ref rvq_proj) = self.rvq_output_proj {
            z = rvq_proj.forward(&z)?;
            println!("  ✓ RVQ output projection applied: {:?}", z.dims());
        }
        
        // Apply pre-conv: 512 -> 1024
        if let Some(ref pre_conv) = self.pre_conv {
            z = pre_conv.forward(&z)?;
            println!("  ✓ Pre-conv applied: {:?}", z.dims());
        }
        
        // Run through pre-transformer (8 layers)
        // Pre-transformer: [1, 1024, seq] -> [1, 1024, seq]
        if let Some(ref transformer) = self.pre_transformer {
            z = transformer.forward(&z)?;
            println!("  ✓ Pre-transformer applied: {:?}", z.dims());
        }
        
        // Apply initial decoder conv (1024->1536)
        if let Some(ref initial_conv) = self.initial_conv {
            z = initial_conv.forward(&z)?;
            println!("  ✓ Initial decoder conv: {:?}", z.dims());
        }
        
        // Apply decoder blocks (8x -> 40x -> 160x -> 480x total upsampling)
        if let Some(ref decoder_blocks) = self.decoder_blocks {
            for (i, block) in decoder_blocks.iter().enumerate() {
                z = block.forward(&z)?;
                println!("  ✓ Decoder block {}: {:?}", i + 1, z.dims());
            }
        }
        
        // Apply final activation
        if let Some(ref final_act) = self.final_act {
            z = final_act.forward(&z)?;
            println!("  ✓ Final activation: {:?}", z.dims());
        }
        
        // Apply final conv (96 -> 1 channel)
        if let Some(ref final_conv) = self.final_conv {
            z = final_conv.forward(&z)?;
            println!("  ✓ Final conv: {:?}", z.dims());
        }
        
        // Extract waveform: [1, 1, samples] -> [samples]
        let audio = z.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        
        // Normalize to [-1, 1]
        let max_val = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let mut audio_normalized = audio.clone();
        if max_val > 1e-6 {
            let scale = 0.95 / max_val;
            for sample in &mut audio_normalized {
                *sample *= scale;
            }
        }

        println!("✓ Decoded to {} samples ({:.2}s at {}Hz) in {:.2}s wallclock", 
                 audio_normalized.len(), 
                 audio_normalized.len() as f32 / self.output_sample_rate as f32,
                 self.output_sample_rate,
                 decode_start.elapsed().as_secs_f32());

        Ok(audio_normalized)
    }

    pub fn sample_rate(&self) -> usize {
        self.output_sample_rate
    }
}

#[allow(dead_code)]

// Simple linear interpolation for audio resampling
fn interpolate_audio(input: &[f32], target_len: usize) -> Vec<f32> {
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    
    let mut output = Vec::with_capacity(target_len);
    let ratio = (input.len() - 1) as f32 / (target_len - 1).max(1) as f32;
    
    for i in 0..target_len {
        let src_idx = i as f32 * ratio;
        let idx0 = src_idx.floor() as usize;
        let idx1 = (idx0 + 1).min(input.len() - 1);
        let frac = src_idx - idx0 as f32;
        
        let sample = input[idx0] * (1.0 - frac) + input[idx1] * frac;
        output.push(sample);
    }
    
    // Normalize to [-1, 1] range
    let max_val = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if max_val > 0.0 {
        for sample in &mut output {
            *sample /= max_val;
        }
    }
    
    output
}
