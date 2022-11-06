python -m torch.distributed.launch --nproc_per_node=8 scripts/train_hfai.py \
    --config src/configs/hfai/final/p-full-no-bright.yaml \
    --output_dir output/final/p-full-no-bright


# CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
#     --config src/configs/final_train_perceiver_2layer_super.yaml \
#     --output_dir output/pretrain_final_debug