CUDA_VISIBLE_DEVICES=3,5,6,7 torchrun --nproc_per_node=4 scripts/train.py \
    --config src/configs/final_train_perceiver_2layer_super.yaml \
    --output_dir output/pretrain_last


# CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
#     --config src/configs/final_train_perceiver_2layer_super.yaml \
#     --output_dir output/pretrain_final_debug