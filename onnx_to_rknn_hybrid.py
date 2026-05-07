#!/usr/bin/env python3

from pathlib import Path
from rknn.api import RKNN  # make sure rknn-toolkit2 is installed
import argparse
import sys

# Optional: implement or import your prepare_dataset function
def prepare_dataset(ds_path):
    """Prepare dataset folder for RKNN quantization."""
    # Example: just return the folder path itself
    # You can modify this to create .txt list of images if needed
    return ds_path

def onnx_to_rknn_hybrid(weights, target, dtype, ds_path, out_path, mean, std, step_num):
    """Convert ONNX model to RKNN format."""
    if ds_path is not None:
        ds_path = Path(ds_path)
        if ds_path.is_dir():
            ds_path = prepare_dataset(ds_path)

    rknn = RKNN(verbose=True)

    try:
        print('--> Configuring model')

        if step_num == 1:
            rknn.config(mean_values=[mean], std_values=[std], target_platform=target)

            # Load
            print('--> Loading ONNX model')
            if rknn.load_onnx(model=str(weights)) != 0:
                raise RuntimeError('Failed to load model')
        
            ret = rknn.hybrid_quantization_step1(dataset=ds_path, proposal=False)
            if ret != 0:
                raise RuntimeError('Failed to hybrid quantization step1')

            print('Step 1 Done!')
            return None
        
        elif step_num == 2:
            ret = rknn.hybrid_quantization_step2(
                model_input=str(weights.with_suffix('.model')),
                data_input=str(weights.with_suffix('.data')),
                model_quantization_cfg=str(weights.with_suffix('.quantization.cfg')))
            if ret != 0:
                raise RuntimeError('Failed to hybrid quantization step2')
            
            # Export
            print(f'--> Exporting to {out_path}')
            if rknn.export_rknn(str(out_path)) != 0:
                raise RuntimeError('Failed to export model')
            
            print('Step 2 Done!')
            return out_path
    finally:
        rknn.release()

    return out_path

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN hybrid quantization")
    parser.add_argument('--weights', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--target', type=str, required=True, help='RKNN target platform, e.g., rk3588')
    parser.add_argument('--dtype', type=str, default='int8', help='Quantization type: int8/uint8/float16')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset folder path for quantization')
    parser.add_argument('--output', type=str, required=True, help='Output RKNN file path')
    parser.add_argument('--mean', type=float, default=[0,0,0], help='Mean value for preprocessing')
    parser.add_argument('--std', type=float, default=[255,255,255], help='Std value for preprocessing')
    parser.add_argument('--step', type=int, default=1, choices=[1,2], help='Hybrid quantization step: 1 or 2')

    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_path = Path(args.output)
    
    try:
        onnx_to_rknn_hybrid(
            weights=weights_path,
            target=args.target,
            dtype=args.dtype,
            ds_path=args.dataset,
            out_path=output_path,
            mean=args.mean,
            std=args.std,
            step_num=args.step
        )
        print("RKNN hybrid conversion finished successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()