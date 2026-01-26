#!/usr/bin/env python3
"""
Simple script to convert PyTorch .pt model files to safetensors format.

Usage:
    python convert_pt_to_safetensors.py <input.pt>
    
The output file will be saved in the same directory with .safetensors extension.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    print(f"Error: Required library not installed: {e}", file=sys.stderr)
    print("\nPlease install required dependencies:", file=sys.stderr)
    print("  pip install torch safetensors", file=sys.stderr)
    sys.exit(1)


def convert_pt_to_safetensors(pt_path: str) -> None:
    """
    Convert a .pt file to .safetensors format.
    
    Args:
        pt_path: Path to the input .pt file
    """
    input_path = Path(pt_path)
    
    # Validate input file
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    if input_path.suffix != '.pt':
        print(f"Warning: Input file does not have .pt extension: {input_path}")
    
    # Determine output path
    output_path = input_path.with_suffix('.safetensors')
    
    print(f"Loading PyTorch model from: {input_path}")
    
    try:
        # Load the PyTorch model
        state_dict: Any = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Handle different formats
        if isinstance(state_dict, dict):
            # If it's already a state dict, use it directly
            if 'state_dict' in state_dict:
                # Some checkpoints wrap the actual state dict
                state_dict = state_dict['state_dict']
            
            # Filter out non-tensor values
            tensors: Dict[str, torch.Tensor] = {
                k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)
            }
            
            if not tensors:
                print("Error: No tensors found in the model file", file=sys.stderr)
                sys.exit(1)
            
            print(f"Found {len(tensors)} tensors")
            
        else:
            print(f"Error: Unexpected model format: {type(state_dict)}", file=sys.stderr)
            print("Expected a dictionary of tensors or a state dict", file=sys.stderr)
            sys.exit(1)
        
    except Exception as e:
        print(f"Error loading PyTorch model: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Saving to safetensors format: {output_path}")
    
    try:
        # Save as safetensors
        save_file(tensors, str(output_path))
        print(f"✓ Successfully converted to: {output_path}")
        
        # Print file size comparison
        input_size = input_path.stat().st_size / (1024 * 1024)
        output_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Input size:  {input_size:.2f} MB")
        print(f"  Output size: {output_size:.2f} MB")
        
    except Exception as e:
        print(f"Error saving safetensors file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch .pt model files to safetensors format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.pt
  %(prog)s models/my-model/voice.pt
        """
    )
    parser.add_argument(
        'input',
        metavar='INPUT.pt',
        help='Path to the input .pt file'
    )
    
    args = parser.parse_args()
    
    convert_pt_to_safetensors(args.input)


if __name__ == '__main__':
    main()
