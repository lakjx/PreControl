nvidia-smi

num_samples=1
temp='0.0'
# llm='llama-3.2-3b-it'
llm="phi-4-mini-it"

# dataset_name="HelpSteer2"
dataset_name="CodeUltraFeedback"

base_dir="."
dset_path="${base_dir}/data/value_function_features/${dataset_name}/ns=${num_samples}/temp=${temp}/${llm}"

hf_cache="${base_dir}/cache"

splits=("train" "test")

gpus="0,1,2,3,4,5"
bsz=512

for split in "${splits[@]}"; do
    PYTHONPATH=$PYTHONPATH:$(pwd) \
    CUDA_VISIBLE_DEVICES=$gpus \
    python src/reward_label.py \
        --reward_model_name "ArmoRM-8B" \
        --batch_size $bsz \
        --response_path "${dset_path}/response_${split}.jsonl" \
        --hf_cache_dir $hf_cache \
        --dataset $dataset_name
done