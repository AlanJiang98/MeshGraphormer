CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 scripts/train.py \
    --config src/configs/train_perceiver_super.yaml \
    --outut_dir /userhome/wangbingxuan/code/MeshGraphormer/output/test_pretrain