nvidia-smi
export TF_ENABLE_ONEDNN_OPTS=0

# llm="llama-3.2-3b-it"
llm="phi-4-mini-it"
# dataset="hs2"
# dataset_name="HelpSteer2"
dataset="code-uf"
dataset_name="CodeUltraFeedback"
num_samples=1
train_temp="0.0"
base_dir="."
model_path="${base_dir}/models"
output_base_path="./data/inference_intervention"
hf_cache="${base_dir}/cache"
gpus="0,1,2,3,4,5,6"
hidden_dims="[3072, 3072, 3072]"
bsz=24
n_new=512
intervene=True
target="[3,3,3,3,3]"
output_postfix="{tgt=${target}}"
lr=5e-2
n_step=1000 # place holder as our intervention is closed-form
infer_temp=0.0
iteration=0
num_processes=1
# value_model_fnm="vf-{md=llama,dset=code-uf}_{lr=1.0e-04,bsz=32,lambda=0.9}"
# value_model_fnm="vf_{md=phi,dset=hs2,lr=1.0e-04,bsz=64,lambda=0.9}"
# value_model_fnm="vf-{md=phi,dset=code-uf}_{lr=1.0e-04,bsz=32,lambda=0.9}"
value_model_fnm="vf-{md=llama,dset=hs2}_{lr=1.0e-04,bsz=32,lambda=0.9}"
value_model_fnm_path="${value_model_fnm}/best_model.pth"


output_postfix="{tgt=${target}}"
output_dir_nm="Inference-Result_{md=${llm},ds=${dataset}}-ns=${num_samples}-train_temp=${train_temp}"
if [ "$intervene" == "True" ]; then
    if [ "$iteration" -eq 0 ]; then
        pre_inference_path="${output_base_path}/${output_dir_nm}/infer_temp=${infer_temp}/base"
    elif [ "$iteration" -gt 0 ]; then
        pre_inference_path="${output_base_path}/${output_dir_nm}/infer_temp=${infer_temp}/${value_model_fnm}/{lr=${lr_map[$target]:-},#s=${n_step}}/edited_${output_postfix}/iteration=$((iteration-1))"
    fi
    output_dir_nm="${output_dir_nm}/infer_temp=${infer_temp}/${value_model_fnm}/{lr=${lr},#s=${n_step}}"
    PYTHONPATH=$PYTHONPATH:$(pwd) \
    CUDA_VISIBLE_DEVICES=$gpus \
    python src/inference_intervention.py \
        --model_name "${llm}" \
        --dataset_name $dataset_name \
        --use_intervention $intervene \
        --value_model_path "${model_path}/${value_model_fnm_path}" \
        --output_path "${output_base_path}/${output_dir_nm}" \
        --output_postfix "${output_postfix}" \
        --hf_cache_dir "${hf_cache}" \
        --hidden_dims "${hidden_dims}" \
        --max_new_tokens $n_new \
        --batch_size $bsz \
        --intervene_lr $lr \
        --intervene_steps $n_step \
        --target $target \
        --temp $infer_temp \
        --num_processes $num_processes \
        --iteration $iteration \
        --pre_inference_path "${pre_inference_path}"
elif [ "$intervene" == "False" ]; then
    output_dir_nm="${output_dir_nm}/infer_temp=${infer_temp}/base"

    PYTHONPATH=$PYTHONPATH:$(pwd) \
    CUDA_VISIBLE_DEVICES=$gpus \
    python src/inference_intervention.py \
        --model_name "${llm}" \
        --dataset_name $dataset_name \
        --use_intervention $intervene \
        --value_model_path "${model_path}/${value_model_fnm_path}" \
        --output_postfix "${output_postfix}" \
        --output_path "${output_base_path}/${output_dir_nm}" \
        --hf_cache_dir "${hf_cache}" \
        --hidden_dims "${hidden_dims}" \
        --max_new_tokens $n_new \
        --batch_size $bsz \
        --intervene_lr $lr \
        --intervene_steps $n_step \
        --target $target \
        --temp $infer_temp \
        --num_processes $num_processes
fi


