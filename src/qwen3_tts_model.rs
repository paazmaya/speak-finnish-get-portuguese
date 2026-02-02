use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, ops::softmax, Embedding, Linear, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TTSConfig {
    pub talker_config: TalkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub hidden_size: usize,
    pub text_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_code_groups: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: String,
    pub rope_scaling: Option<RopeScaling>,
    pub code_predictor_config: CodePredictorConfig,
    pub spk_id: HashMap<String, u32>,
    pub codec_language_id: HashMap<String, u32>,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_pad_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub mrope_section: Vec<usize>,
    pub interleaved: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_code_groups: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<RopeScaling>,
}

// RMSNorm implementation
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

// Rotary position embeddings with multi-grid support
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    mrope_section: Option<Vec<usize>>,
}

impl RotaryEmbedding {
    fn new(
        dim: usize,
        max_position_embeddings: usize,
        base: f64,
        device: &Device,
        mrope_section: Option<Vec<usize>>,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        Ok(Self {
            sin: emb.sin()?,
            cos: emb.cos()?,
            mrope_section,
        })
    }

    fn apply_rotary_pos_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = position_ids.dim(D::Minus1)?;

        let cos = self.cos.i(..seq_len)?.unsqueeze(0)?;
        let sin = self.sin.i(..seq_len)?.unsqueeze(0)?;

        let q_embed = Self::rotate_half_and_apply(q, &cos, &sin)?;
        let k_embed = Self::rotate_half_and_apply(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    fn rotate_half_and_apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let last_dim = x.dim(D::Minus1)?;
        let half = last_dim / 2;

        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;

        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        let result = (x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?;
        Ok(result)
    }
}

// Grouped Query Attention
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
}

impl Attention {
    fn new(config: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let mrope_section = config
            .rope_scaling
            .as_ref()
            .map(|s| s.mrope_section.clone());
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
            mrope_section,
        )?;

        // Q/K normalization
        let q_norm = Some(RmsNorm::new(
            head_dim,
            config.rms_norm_eps,
            vb.pp("q_norm"),
        )?);
        let k_norm = Some(RmsNorm::new(
            head_dim,
            config.rms_norm_eps,
            vb.pp("k_norm"),
        )?);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            q_norm,
            k_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply Q/K normalization
        let q = if let Some(ref q_norm) = self.q_norm {
            q_norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref k_norm) = self.k_norm {
            k_norm.forward(&k)?
        } else {
            k
        };

        // Apply rotary embeddings
        let (q, k) = self.rotary_emb.apply_rotary_pos_emb(&q, &k, position_ids)?;

        // Grouped Query Attention: repeat k,v heads if needed
        let k = Self::repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = Self::repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Attention scores
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
            Ok(x.unsqueeze(2)?
                .expand(&[b, n_kv_heads, n_rep, seq_len, head_dim])?
                .reshape((b, n_kv_heads * n_rep, seq_len, head_dim))?)
        }
    }
}

// SwiGLU MLP
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(config: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&gate.mul(&up)?)?)
    }
}

// Decoder Layer
struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

// Code Predictor Attention
struct CodePredictorAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
}

impl CodePredictorAttention {
    fn new(config: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let mrope_section = config
            .rope_scaling
            .as_ref()
            .map(|s| s.mrope_section.clone());
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
            mrope_section,
        )?;

        let q_norm = Some(RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?);
        let k_norm = Some(RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            q_norm,
            k_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = if let Some(ref q_norm) = self.q_norm {
            q_norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref k_norm) = self.k_norm {
            k_norm.forward(&k)?
        } else {
            k
        };

        let (q, k) = self.rotary_emb.apply_rotary_pos_emb(&q, &k, position_ids)?;

        let k = Self::repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = Self::repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
            Ok(x.unsqueeze(2)?
                .expand(&[b, n_kv_heads, n_rep, seq_len, head_dim])?
                .reshape((b, n_kv_heads * n_rep, seq_len, head_dim))?)
        }
    }
}

// Code Predictor MLP
struct CodePredictorMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl CodePredictorMlp {
    fn new(config: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&gate.mul(&up)?)?)
    }
}

// Code Predictor Layer
struct CodePredictorLayer {
    input_layernorm: RmsNorm,
    self_attn: CodePredictorAttention,
    post_attention_layernorm: RmsNorm,
    mlp: CodePredictorMlp,
}

impl CodePredictorLayer {
    fn new(config: &CodePredictorConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: CodePredictorAttention::new(config, vb.pp("self_attn"))?,
            post_attention_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: CodePredictorMlp::new(config, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, position_ids, attention_mask)?;
        let hidden_states = (hidden_states + residual)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        Ok((hidden_states + residual)?)
    }
}

impl DecoderLayer {
    fn new(config: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            post_attention_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::new(config, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, position_ids, attention_mask)?;
        let hidden_states = (hidden_states + residual)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        Ok((hidden_states + residual)?)
    }
}

// Text Projection MLP (ResizeMLP with intermediate layer)
pub struct TextProjection {
    fc1: Linear,
    fc2: Linear,
}

impl TextProjection {
    fn new(text_hidden_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        // Architecture: text_hidden_size -> text_hidden_size -> hidden_size
        // fc1: text_hidden_size -> text_hidden_size (2048 -> 2048)
        // fc2: text_hidden_size -> hidden_size (2048 -> 1024)
        Ok(Self {
            fc1: linear(text_hidden_size, text_hidden_size, vb.pp("linear_fc1"))?,
            fc2: linear(text_hidden_size, hidden_size, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?.silu()?;
        Ok(self.fc2.forward(&h)?)
    }
}

// Main Talker Model
pub struct Qwen3TTSTalkerModel {
    codec_embedding: Embedding,
    text_embedding: Embedding,
    text_projection: TextProjection,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    codec_head: Linear,
    config: TalkerConfig,
}

// Code Predictor Model
pub struct Qwen3TTSCodePredictor {
    codec_embeddings: Vec<Embedding>,
    layers: Vec<CodePredictorLayer>,
    norm: RmsNorm,
    lm_heads: Vec<Linear>,
    config: CodePredictorConfig,
}

impl Qwen3TTSTalkerModel {
    pub fn load(vb: VarBuilder, config: &TalkerConfig) -> Result<Self> {
        let codec_embedding = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.codec_embedding"),
        )?;
        let text_embedding = embedding(
            config.text_vocab_size,
            config.text_hidden_size,
            vb.pp("model.text_embedding"),
        )?;
        let text_projection = TextProjection::new(
            config.text_hidden_size,
            config.hidden_size,
            vb.pp("text_projection"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(config, vb_layers.pp(&i.to_string()))?);
        }

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let codec_head =
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("codec_head"))?;

        Ok(Self {
            codec_embedding,
            text_embedding,
            text_projection,
            layers,
            norm,
            codec_head,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        codec_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        // Embed text and codec separately
        let text_embeds = self.text_embedding.forward(input_ids)?;
        let text_embeds = self.text_projection.forward(&text_embeds)?;

        let codec_embeds = self.codec_embedding.forward(codec_ids)?;

        // Concatenate embeddings
        let hidden_states = Tensor::cat(&[&text_embeds, &codec_embeds], 1)?;

        // Forward through transformer layers
        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, position_ids, None)?;
        }

        // Final norm and codec head
        let hidden_states = self.norm.forward(&hidden_states)?;
        Ok(self.codec_head.forward(&hidden_states)?)
    }

    pub fn create_codec_prefix(
        &self,
        device: &Device,
        language_id: u32,
        speaker_id: u32,
        codec_think_id: u32,
        codec_think_bos_id: u32,
        codec_think_eos_id: u32,
        codec_pad_id: u32,
        codec_bos_id: u32,
    ) -> Result<Tensor> {
        let prefix = vec![
            codec_think_id,
            codec_think_bos_id,
            language_id,
            codec_think_eos_id,
            speaker_id,
            codec_pad_id,
            codec_bos_id,
        ];
        let prefix_len = prefix.len();

        Ok(Tensor::from_vec(prefix, (1, prefix_len), device)?)
    }
}

impl Qwen3TTSCodePredictor {
    pub fn load(vb: VarBuilder, config: &CodePredictorConfig) -> Result<Self> {
        // 12Hz variant uses 15 codebooks (0-14), not 16
        // Try to load actual number of codec embeddings that exist
        let mut codec_embeddings = Vec::new();
        let vb_emb = vb.pp("model.codec_embedding");
        for i in 0..config.num_code_groups {
            match embedding(
                config.vocab_size,
                config.hidden_size,
                vb_emb.pp(&i.to_string()),
            ) {
                Ok(emb) => codec_embeddings.push(emb),
                Err(_) => {
                    println!("⚠ Only {} codec embeddings found (expected {})", i, config.num_code_groups);
                    break;
                }
            }
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..config.num_hidden_layers {
            layers.push(CodePredictorLayer::new(config, vb_layers.pp(&i.to_string()))?);
        }

        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

        // Load same number of lm_heads as codec_embeddings
        let mut lm_heads = Vec::new();
        let vb_lm = vb.pp("lm_head");
        for i in 0..codec_embeddings.len() {
            lm_heads.push(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb_lm.pp(&i.to_string()),
            )?);
        }

        Ok(Self {
            codec_embeddings,
            layers,
            norm,
            lm_heads,
            config: config.clone(),
        })
    }

    pub fn num_quantizers(&self) -> usize {
        self.codec_embeddings.len()
    }

    pub fn forward(&self, codec_ids: &Tensor, position_ids: &Tensor) -> Result<Vec<Tensor>> {
        let dims = codec_ids.dims();
        if dims.len() != 2 {
            anyhow::bail!("codec_ids must be 2D [num_code_groups, seq_len]");
        }
        let seq_len = dims[1];

        let mut hidden_states: Option<Tensor> = None;
        // Use actual number of loaded embeddings, not config value
        let num_embeddings = self.codec_embeddings.len();
        for i in 0..num_embeddings {
            let ids = codec_ids.i(i)?;
            let ids = ids.unsqueeze(0)?;
            let embed = self.codec_embeddings[i].forward(&ids)?;
            hidden_states = Some(match hidden_states {
                Some(h) => (h + embed)?,
                None => embed,
            });
        }

        let mut hidden_states = hidden_states.ok_or_else(|| anyhow::anyhow!("Missing embeddings"))?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, position_ids, None)?;
        }
        let hidden_states = self.norm.forward(&hidden_states)?;

        let mut logits_by_group = Vec::with_capacity(self.lm_heads.len());
        for i in 0..self.lm_heads.len() {
            let logits = self.lm_heads[i].forward(&hidden_states)?;
            let logits = logits.reshape((1, seq_len, self.config.vocab_size))?;
            logits_by_group.push(logits);
        }

        Ok(logits_by_group)
    }
}

// Sampling utilities
pub fn sample_token(logits: &Tensor, temperature: f64, top_k: usize, _top_p: f64) -> Result<u32> {
    let logits = if temperature > 0.0 {
        (logits / temperature)?
    } else {
        logits.clone()
    };

    let probs = softmax(&logits, D::Minus1)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // Simple top-k sampling
    let mut indexed_probs: Vec<(usize, f32)> =
        probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k_probs = &indexed_probs[..top_k.min(indexed_probs.len())];

    // Sample from top-k
    let sum: f32 = top_k_probs.iter().map(|(_, p)| p).sum();
    let mut rng = rand::random::<f32>() * sum;

    for (idx, prob) in top_k_probs {
        rng -= prob;
        if rng <= 0.0 {
            return Ok(*idx as u32);
        }
    }

    Ok(top_k_probs[0].0 as u32)
}
