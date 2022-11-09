expr_name="final/p-iter-S-resnet50_2"

python -m torch.distributed.launch --nproc_per_node=8 scripts/train_hfai.py \
    --config src/configs/hfai/${expr_name}.yaml \
    --output_dir output/${expr_name}
    --resume_checkpoint output/final/p-iter-S-resnet50/checkpoint-150-27300/state_dict.bin

python scripts/eval_all.py \
    --output_dir output/${expr_name} \
    --result_dir results/${expr_name} 