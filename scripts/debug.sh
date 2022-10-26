CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 scripts/train.py \
    --config src/configs/debug_pretrain.yaml \
    --output_dir /userhome/wangbingxuan/code/MeshGraphormer/output/pretrain_final
    
# CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
#     --config src/configs/debug_pretrain.yaml \
#     --output_dir /userhome/wangbingxuan/code/MeshGraphormer/output/debug_2