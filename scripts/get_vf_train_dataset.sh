nvidia-smi

# llm="llama-3.2-3b-it"
llm="phi-4-mini-it"

dataset_name="HelpSteer2"
# dataset_name="CodeUltraFeedback"

num_samples=1
temperature=0.0


base_dir="."
hf_cache="${base_dir}/cache"

bsz=48
n_new=1024

gpus="0,1,2,3,4,5,6,7"
num_processes=1


PYTHONPATH=$PYTHONPATH:$(pwd) \
CUDA_VISIBLE_DEVICES=$gpus \
out_dir="${base_dir}/data/value_function_features/${dataset_name}/ns=${num_samples}/temp=${temperature}/${llm}"
accelerate launch --num_processes ${num_processes} --main_process_port 29508 src/get_activations_only.py \
    --model_name "$llm" \
    --dataset_name "$dataset_name" \
    --num_samples $num_samples \
    --output_path $out_dir \
    --hf_cache_dir $hf_cache \
    --max_prompt_tokens 2048 \
    --max_new_tokens $n_new \
    --batch_size $bsz \
    --temperature $temperature

