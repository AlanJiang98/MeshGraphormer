import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--result_dir", type=str, default="./output/results")
    args = parser.parse_args()
    yaml_path = os.path.join(args.output_dir, "train.yaml")
    for ckpt_folder in os.listdir(args.output_dir):        
        if "checkpoint" in ckpt_folder:

            ckpt_path = os.path.join(args.output_dir, ckpt_folder, "state_dict.bin")
            result_path = os.path.join(args.result_dir, ckpt_folder)
            print(f"Evaluating {ckpt_path}.....")
            cmd = f"CUDA_VISIBLE_DEVICES=3 python ./scripts/train.py --config {yaml_path} --resume_checkpoint {ckpt_path} --config_merge ./src/configs/hfai/eval_evrealhands.yaml --run_eval_only --output_dir {result_path}"
            os.system(cmd)
