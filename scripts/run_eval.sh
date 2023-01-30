root_dir='/userhome/wangbingxuan/data/hfai_output/final/p-full-iter-5-new'
output_dir='./output/final/final/p-full-iter-5-new-fps'
downloadList=("eval_evrealhands_30" "eval_evrealhands_45" "eval_evrealhands_60" "eval_evrealhands_75" "eval_evrealhands_90")
weightList=("checkpoint-300-27300")
for(( i=0;i<${#downloadList[@]};i++));
do
  CUDA_VISIBLE_DEVICES=3 python ./scripts/train.py --config ${root_dir}/train.yaml --resume_checkpoint ${root_dir}/checkpoint-199-48399/state_dict.bin --config_merge ./src/configs/${downloadList[i]}.yaml --run_eval_only --output_dir ${output_dir}${downloadList[i]}
done
#downloadList=("evhands" "f-event" "f-rgb" "p-full-iter" )
#weightList=("checkpoint-150-4200" "checkpoint-300-27300" "checkpoint-300-27300" "checkpoint-270-49140" )
#downloadList=("evhands" "f-event" "f-rgb" "p-full" "p-full-lr3-nodecay-evreal" "g-rgb" "p-full-iter" "p-no-both" "p-no-mask" "p-no-scene")
#weightList=("checkpoint-150-4200" "checkpoint-300-27300" "checkpoint-300-27300" "checkpoint-150-27300" "checkpoint-300-19200" "checkpoint-300-27300" "checkpoint-150-27300" "checkpoint-150-27300" "checkpoint-150-27300" "checkpoint-150-27300")
#for(( i=0;i<${#downloadList[@]};i++));
#do
#  CUDA_VISIBLE_DEVICES=3 python ./scripts/train.py --config ${root_dir}${downloadList[i]}/train.yaml --resume_checkpoint ${root_dir}${downloadList[i]}/${weightList[i]}/state_dict.bin --config_merge ./src/configs/eval_evrealhands.yaml --run_eval_only --output_dir ${output_dir}${downloadList[i]}
#done



