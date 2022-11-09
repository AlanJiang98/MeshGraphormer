expr_name="final/p-iter-S-resnet34-smaller-s"

python -m torch.distributed.launch --nproc_per_node=8 scripts/train_hfai.py \
    --config src/configs/hfai/${expr_name}.yaml \
    --output_dir output/${expr_name}

python scripts/eval_all.py \
    --output_dir output/${expr_name} \
    --result_dir results/${expr_name} 


# CUDA_VISIBLE_DEVICES=7 python scripts/train.py \
#     --config src/configs/final_train_perceiver_2layer_super.yaml \
#     --output_dir output/pretrain_final_debug