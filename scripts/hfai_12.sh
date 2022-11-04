python -m torch.distributed.launch --nproc_per_node=8 scripts/train_hfai_nodecay.py \
    --config src/configs/hfai/1104/p-full-lr3-evreal.yaml \
    --output_dir output/1104/p-full-lr3-nodecay-evreal


# CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
#     --config src/configs/final_train_perceiver_2layer_super.yaml \
#     --output_dir output/pretrain_final_debug