from intervented_model.llama import Intervented_LlamaForCausalLM
from intervented_model.phi import Intervented_PhiForCausalLM
import torch
import argparse
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import datetime
from tqdm import tqdm

from os.path import join as os_join
from typing import Any, Dict, List

import torch.nn.functional as fn
from itertools import chain

from stefutil.prettier import get_logger, style as s, icecream as sic
from src.util import argparse_str2bool, argparse_str2int_list, argparse_str2float_list,\
     model_generation_config2dict, get_last_layer_output_token_hidden_states, set_seed
from src.util.data import helpsteer2_iterative_messages, helpsteer2_prompt2messages, \
                        ultrafeedback_prompt2messages, ultrafeedback_iterative_messages
from src.value_function import ValueFunction
from distutils.util import strtobool
import torch.distributed as dist
import torch.multiprocessing as mp

class DataCollatorReward:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data):
        batch = {}
        data_batch = []
        for sample in data:
            data_batch.append({"input_ids": sample['input_ids'], "attention_mask": sample["attention_mask"]})
        batch_data = self.tokenizer.pad(data_batch, padding=True, return_tensors="pt")
        batch['input_ids'] = batch_data['input_ids']
        batch['attention_mask'] = batch_data['attention_mask']
        return batch

def run_parallel(
    rank: int,
    world_size: int,
    args,
    model_name,
    dataset,
    target_score,
    logger
):
    # Set up multi-process/gpu
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(hours=2))
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')


    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=args.hf_cache_dir)

    if args.model_name in ["llama-3.2-3b-it", "llama-3.2-1b-it"]:
        cls = Intervented_LlamaForCausalLM if args.use_intervention else AutoModelForCausalLM
    elif args.model_name == "phi-4-mini-it":
        cls = Intervented_PhiForCausalLM if args.use_intervention else AutoModelForCausalLM
    model = cls.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map=device, cache_dir=args.hf_cache_dir)

    value_model = ValueFunction(input_dim=args.hidden_dims[0], hidden_dims=args.hidden_dims, num_attributes=5, logger=logger)
    checkpoint = torch.load(args.value_model_path, map_location=device)
    value_model.load_state_dict(checkpoint)
    # Place value model on the same device as the model's last layer for efficient interaction
    value_model = value_model.to(dtype=torch.bfloat16)
    value_model.to(device)

    if args.use_intervention:
        model.logger = logger
        model.value_model = value_model
        model.lr = args.intervene_lr
        model.epochs = args.intervene_steps
        model.patience = args.patience

    generation_config = model.generation_config
    generation_config.top_k = None
    generation_config.do_sample = False
    if generation_config.do_sample:
        set_seed(args.seed)
        generation_config.temperature = args.temp
        generation_config.top_p = 1.0
    else:
        generation_config.temperature = None
        generation_config.top_p = None
    if rank == 0:
        logger.info(f'Setting generation config {s.i(model_generation_config2dict(generation_config), indent=1)} to greedy decoding')
        sic(tokenizer.eos_token, tokenizer.pad_token)
    
    if tokenizer.pad_token is None:
        if rank == 0:
            logger.info(f'Tokenizer pad token is None - Setting it to tokenizer eos token: {s.i(tokenizer.eos_token)}')
        tokenizer.pad_token = tokenizer.eos_token
        generation_config.pad_token_id = tokenizer.eos_token_id

    if args.model_name in ['llama-3.2-1b-it', 'llama-3.2-3b-it']:
        assert generation_config.eos_token_id == [128001, 128008, 128009]  # using this default 3-token eos token will cause processing errors
        assert tokenizer.eos_token_id == 128009  # note this is a discrepancy where the tokenizer has a single-token eos token
        # note without this, processing will not work, cos for some reason,
        #   llama-3.2 finishes the generation w/ eos token (`<|eot_id|>`),
        #   but then pad w/ another pad token (`<|end_of_text|`)
        #   this is cos by default, huggingface::generate will ust the 1st eos token, which is 128001
        if rank == 0:
            logger.warning(
                f'Overriding model eos token ID to match tokenizer for open-ended generation: {s.i(generation_config.eos_token_id)} -> {s.i(tokenizer.eos_token_id)}')
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    elif args.model_name == "phi-4-mini-it":
        print(generation_config.eos_token_id)
        assert generation_config.eos_token_id == [200020, 199999]  # using this default 3-token eos token will cause processing errors
        assert tokenizer.eos_token_id == 199999  # note this is a discrepancy where the tokenizer has a single-token eos token
        # note without this, processing will not work, cos for some reason,
        #   llama-3.2 finishes the generation w/ eos token (`<|eot_id|>`),
        #   but then pad w/ another pad token (`<|end_of_text|`)
        #   this is cos by default, huggingface::generate will ust the 1st eos token, which is 128001
        if rank == 0:
            logger.warning(
                f'Overriding model eos token ID to match tokenizer for open-ended generation: {s.i(generation_config.eos_token_id)} -> {s.i(tokenizer.eos_token_id)}')
        tokenizer.eos_token_id = tokenizer.pad_token_id =  200020
        model.generation_config.eos_token_id = model.generation_config.pad_token_id = 200020
    if rank == 0:
        logger.info(
            f'Generating responses w/ generation config: {s.i(model_generation_config2dict(model.generation_config), indent=1)}')

    # 4) Shard data
    n = len(dataset)
    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end   = min(start + chunk, n)
    local_dataset = dataset.select(range(start, end))

    if args.use_intervention:
        if args.dataset_name in ['HelpSteer2', "CodeUltraFeedback"]:
            def collator(batch: List[Dict[str, Any]]):
                return dict(
                    prompt_messages=[msg['prompt_messages'] for msg in batch],
                    target_score=[msg['target_score'] for msg in batch]
                )
        else:
            raise NotImplementedError
    else:
        if args.dataset_name in ['HelpSteer2', "CodeUltraFeedback"]:
            def collator(batch: List[Dict[str, Any]]):
                return dict(
                    prompt_messages=[msg['prompt_messages'] for msg in batch]
                )
        else:
            raise NotImplementedError

    dataloader = DataLoader(local_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    generated_responses = []

    it = tqdm(dataloader, desc=f'Process {rank} Generating responses', position=rank)

    vf_outputs = []  # get the VF preds to on the default hidden states for analysis
    for batch in it:
        if args.use_intervention:
            lst_msgs, target_score = batch['prompt_messages'], batch['target_score']
            target_score = torch.tensor(target_score, dtype=torch.bfloat16, device=device)
        else:
            lst_msgs = batch['prompt_messages']

        # seems memory sufficient w/o truncation
        inputs = tokenizer.apply_chat_template(lst_msgs, add_generation_prompt=True, return_tensors='pt', padding=True, return_dict=True, return_attention_mask=True)
            
        input_ids = inputs['input_ids']
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        batch_size, n_prompt = input_ids.shape
        if args.use_intervention:
            model.edit_target = args.target
            model.target_score = target_score
            model.rank = rank
            model.sequence_unfinished_flag = torch.ones(batch_size, dtype=torch.bool, device=model.device)  # reset the flag each time for a new batch
            model.original_hidden_states = []  # reset for each batch
            model.hidden_states_delta = []
            model.generated_token_counts = None
        
        # get the VF-edited hidden states for analysis
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_return_sequences=1, return_dict_in_generate=True, output_hidden_states=True)
        
        # get a mask for the padding tokens for filtering
        output_ids = outputs.sequences[:, n_prompt:]
        mask_pad = output_ids == tokenizer.pad_token_id

        hidden_states = get_last_layer_output_token_hidden_states(outputs.hidden_states)  # list by seq len of tensors of shape (batch_size, hidden_dim)
        hidden_states = torch.stack(hidden_states, dim=1)  # (batch_size, seq_len, hidden_dim)
        
        # Make sure hidden states are on the same device as the value model
        hidden_states = hidden_states.to(model.device)
        
        vf_out = value_model(hidden_states)
        # filter by the padding tokens
        vf_out[mask_pad.to(vf_out.device)] = -1
        vf_outputs.append(vf_out.cpu())

        # sanity check seq lengths match
        assert vf_outputs[-1].shape[1] <= args.max_new_tokens

        
        it.set_postfix({
            '|prompt|': s.i(n_prompt),
            '#generated-token': f'{s.i(outputs.sequences.shape[1] - n_prompt)}/{s.i(args.max_new_tokens)}'
        })

        outputs_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for ppt, generated_response, msgs in zip(prompt, outputs_text, lst_msgs):
            res = generated_response.removeprefix(ppt)
            generated_responses.append(dict(prompt=msgs, response=res))
    
    dir_nm = f'edited_{args.output_postfix}/iteration={args.iteration}' if args.use_intervention \
                else ""
    output_path = os_join(args.output_path, dir_nm)
    os.makedirs(output_path, exist_ok=True)
    logger.info(f'Writing outputs to {s.i(output_path)}...')

    # 1) sync all ranks
    dist.barrier()
    world_size = dist.get_world_size()

    # 2) gather Python‐object lists
    gathered_responses = [None] * world_size
    dist.all_gather_object(gathered_responses, generated_responses)

    vf_outputs = [fn.pad(vf_out, (0, 0, 0, args.max_new_tokens - vf_out.shape[1], 0, 0), value=-1) for vf_out in vf_outputs]
    vf_outputs = torch.cat(vf_outputs, dim=0)

    gathered_vf  = [None] * world_size
    dist.all_gather_object(gathered_vf, vf_outputs)

    all_vf  = torch.cat(gathered_vf, dim=0)
    all_responses = list(chain.from_iterable(gathered_responses))

    if rank == 0:
        with open(os_join(output_path, "responses.jsonl"), "w") as f:
            for res in all_responses:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        if args.use_intervention:
            torch.save(all_vf,  os_join(output_path, "vf_outputs_after.pth"))
        else:
            torch.save(all_vf,  os_join(output_path, "vf_outputs.pth"))

def exclude_target_sample(target, pre_inference_path):
    unint_score = torch.load(os_join(pre_inference_path, "responses_scores.pth"))
    unint_score = unint_score.round().clip(0, 4).int().tolist()

    dims_to_check = [i for i,v in enumerate(target) if v != 5]
    int_indices = []
    for idx, score in enumerate(unint_score):
        if not all(score[d] == target[d] for d in dims_to_check):
            int_indices.append(idx)
    
    return int_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-3.2-1b')
    parser.add_argument('--dataset_name', type=str, default='HelpSteer2')
    parser.add_argument('--use_intervention', type=argparse_str2bool, default=True, help='Whether to use intervention')
    parser.add_argument('--hidden_dims', type=argparse_str2int_list, help='JSON list of hidden dimensions', default='[2048]')
    parser.add_argument('--value_model_path', type=str, help='Path to the value model')
    parser.add_argument('--intervene_steps', type=int, default=100)
    parser.add_argument('--intervene_lr', type=float, default=0.001)
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of tokens for the LLM to generate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--output_path', type=str, default=os_join('.', 'data', 'inference_intervention'), help='Path to save the inference responses')
    parser.add_argument('--output_postfix', type=str, help='Postfix for the output files')
    parser.add_argument('--hf_cache_dir', type=str, help='Huggingface cache directory')
    parser.add_argument('--target', type=argparse_str2int_list, default="[3,3,3,2,2]")
    parser.add_argument('--temp', type=float, default=0.6)
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--pre_inference_path', type=str, default="", help="Path for previous generate response")
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    logger = get_logger('Inference-Intervention', kind='both+ansi', file_path=os_join(args.output_path, 'run-inference-intervention.log'))
    logger.info(f'Running inference intervention w/ {s.i(vars(args), indent=1)}')

    model_name_map = {
        'llama-3.2-1b-it': 'meta-llama/Llama-3.2-1B-Instruct',
        'llama-3.2-3b-it': 'meta-llama/Llama-3.2-3B-Instruct',
        'phi-4-mini-it': "microsoft/Phi-4-mini-instruct"
    }
    model_name = model_name_map[args.model_name]
    sic(model_name)
        
    logger.info(f'Using target attribute scores for intervention: {s.i(args.target)}')
    if args.dataset_name == 'HelpSteer2':
        dataset = load_dataset("nvidia/HelpSteer2", cache_dir=args.hf_cache_dir, split='validation')
        # as discussed in `get_activations_only.py`,
        #   take the even rows as prompts since each prompt is duplicated 2X for 2 responses
        prompts = dataset['prompt']
        assert prompts[::2] == prompts[1::2]  # sanity check
        dataset = dataset.select(list(range(0, len(prompts), 2)))
        logger.info(f'Generating responses on {s.i(len(dataset))} prompts...')
    elif args.dataset_name == "CodeUltraFeedback":
        dset: datasets.DatasetDict = load_dataset("coseal/CodeUltraFeedback", cache_dir=args.hf_cache_dir)
        
        splits= dset['train'].train_test_split(test_size=0.1, seed=args.seed)
        dataset = splits["test"]

    if args.use_intervention:
        inter_indices = exclude_target_sample(args.target, args.pre_inference_path)
        output_path = os_join(args.output_path, f"edited_{args.output_postfix}/iteration={args.iteration}")
        os.makedirs(output_path, exist_ok=True)
        torch.save(inter_indices, os_join(output_path, "intervene_indices.pth"))
        
        dataset = dataset.select(inter_indices)

    def preprocessing(example):
        if args.iteration == 0:
            if args.dataset_name == "HelpSteer2":
                example['prompt_messages'] = helpsteer2_prompt2messages(example['prompt'])
            elif args.dataset_name == "CodeUltraFeedback":
                example['prompt_messages'] = ultrafeedback_prompt2messages(example['instruction'])
        else:
            if args.dataset_name == "HelpSteer2":
                example['prompt_messages'] = helpsteer2_iterative_messages(example['prompt'], example['response'])
            elif args.dataset_name == "CodeUltraFeedback":
                example['prompt_messages'] = ultrafeedback_iterative_messages(example['instruction'], example['response'])
        return example

    # Load the dataset first
    dataset = dataset.map(preprocessing)
    

    target_score = torch.tensor(args.target).repeat(len(dataset), 1)

    def add_target_score(examples, indices):
        # Get predictions for the current batch of examples
        batch_targets = [target_score[idx] for idx in indices]
        examples['target_score'] = batch_targets
        return examples
    
    # Apply the function to add predictions to the dataset
    dataset = dataset.map(
        add_target_score, 
        with_indices=True,
        batched=True,
        batch_size=100  # Process in small batches for memory efficiency
    )


    mp.spawn(
        run_parallel,
        args=(
            args.num_processes,
            args,
            model_name,
            dataset,
            target_score,  
            logger
        ),
        nprocs=args.num_processes,
        join=True,
    )
            

if __name__ == '__main__':
    from rich.traceback import install
    install(show_locals=False)

    sic.output_width = 128
    main()
