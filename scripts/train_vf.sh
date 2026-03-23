
num_samples=1
temp="0.0"
# llm="llama-3.2-3b-it"
llm="phi-4-mini-it"

dataset_name="CodeUltraFeedback"
# dataset_name="HelpSteer2"

base_dir="."
dset_path="${base_dir}/data/value_function_features/${dataset_name}/ns=${num_samples}/temp=${temp}/${llm}"
md_path="${base_dir}/models"
gpus="0,1,2,3,4,5,6"

hidden_dims="[3072, 3072, 3072]"



optim="adamw"
weight_decay=0.01
schedule="cosine"

bsz=32
lr=1e-4
lambda=0.9
patience=10
epochs=100

PYTHONPATH=$PYTHONPATH:$(pwd) \
CUDA_VISIBLE_DEVICES=$gpus \
python src/train_value_model.py \
    --model_name ${llm} \
    --dataset_name "${dataset_name}" \
    --feature_path "${dset_path}" \
    --model_output_dir "${md_path}" \
    --hidden_dims "${hidden_dims}" \
    --lr $lr \
    --batch_size $bsz \
    --epochs $epochs \
    --optimizer "${optim}" \
    --scheduler "${schedule}" \
    --weight_decay $weight_decay \
    --num_samples ${num_samples} \
    --temperature ${temp} \
    --patience $patience \
    --lambda_param ${lambda}



