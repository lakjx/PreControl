import argparse
import random
import time
import datetime
import os
import json
import torch
import torch.nn.functional as F

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from os.path import join as os_join
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed import all_gather_object
from torch.utils.data import Dataset, Sampler
from itertools import chain

import logging
from typing import List, Dict, Any
from typing import Iterator

from stefutil.prettier import get_logger, style as s, icecream as sic
from src.util import set_seed, style_transformers_logging, model_generation_config2dict, get_last_layer_output_token_hidden_states
from src.util.data import helpsteer2_prompt2messages, ultrafeedback_prompt2messages



class DataCollatorReward:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data: List[Dict[str, Any]]):
        batch = {}
        data_batch = []
        for sample in data:
            data_batch.append({"input_ids": sample['input_ids'], "attention_mask": sample["attention_mask"]})
        batch_data = self.tokenizer.pad(data_batch, padding=True, return_tensors="pt")
        batch['input_ids'] = batch_data['input_ids']
        batch['attention_mask'] = batch_data['attention_mask']
        batch['prompt_messages'] = [sample['prompt_messages'] for sample in data]
        return batch

class UnevenDistributedSampler(Sampler[int]):
    """
    A Distributed Sampler that does NOT replicate or drop samples
    allowing for partial batches in some ranks if the dataset size
    is not evenly divisible by the number of processes.
    """
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool=True,
        seed: int = 42
    ):
        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed

        self.epoch = 0  # For reproducible shuffling across epochs

        # Validate inputs
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"Invalid rank {self.rank} for {self.num_replicas} replicas.")
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1.")
    
    def __iter__(self) -> Iterator[int]:
        """Generate the indices for this process."""
        # Create base indices
        indices = list(range(self.num_samples))

        # Shuffle if needed
        if self.shuffle:
            # Ensure a deterministic shuffle for each epoch
            random.seed(self.seed + self.epoch)
            random.shuffle(indices)

        # Compute this process's slice
        # E.g. rank=0 gets [start0 : end0], rank=1 gets [start1 : end1], etc.
        start = (self.num_samples * self.rank) // self.num_replicas
        end   = (self.num_samples * (self.rank + 1)) // self.num_replicas
        print(f"{self.rank}:{start},{end}")
        return iter(indices[start:end])

    def __len__(self) -> int:
        """
        Number of samples on this rank.
        If the dataset is not divisible by num_replicas,
        some ranks will have one extra or one fewer sample.
        """
        start = (self.num_samples * self.rank) // self.num_replicas
        end   = (self.num_samples * (self.rank + 1)) // self.num_replicas
        return end - start

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler, 
        so that the shuffling is reproducible across epochs.
        """
        self.epoch = epoch


def get_llm_activations(
        model_name: str,
        model: AutoModelForCausalLM,
        dataloader: DataLoader,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        accelerator,
        num_samples: int = 1,
        mode: str = 'train',
        max_new_tokens: int = 128,
        logger: logging.Logger = None,
        seed: int = 42,
        base_output_path: str = None,
        temperature: str = None,
):
    
    # log how many responses finished properly
    n_finish = 0

    gen_args = dict(output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=max_new_tokens)

    # sanity check eos token is a single token for future padding processing logic to work, see below
    generation_config, _ = accelerator.unwrap_model(model)._prepare_generation_config(generation_config=None)
    logger.info(f'Default model generation config: {s.i(model_generation_config2dict(generation_config), indent=1)}')
    if model_name in ['llama-3.2-1b-it', 'llama-3.2-3b-it']:
        assert generation_config.eos_token_id == [128001, 128008, 128009]  # using this default 3-token eos token will cause processing errors
        assert tokenizer.eos_token_id == 128009  # note this is a discrepancy where the tokenizer has a single-token eos token
        # note without this, processing will not work, cos for some reason,
        #   llama-3.2 finishes the generation w/ eos token (`<|eot_id|>`),
        #   but then pad w/ another pad token (`<|end_of_text|`)
        #   this is cos by default, huggingface::generate will ust the 1st eos token, which is 128001
        generation_config.eos_token_id = tokenizer.eos_token_id

        # disables warning: `Setting `pad_token_id` to `eos_token_id`:None for open-end generation.`
        assert tokenizer.pad_token is not None and tokenizer.pad_token_id is not None
        assert generation_config.pad_token_id is None
        generation_config.pad_token_id = tokenizer.eos_token_id
    elif model_name == "phi-4-mini-it":
        assert generation_config.eos_token_id == [200020, 199999]
        assert generation_config.pad_token_id == 199999
        generation_config.eos_token_id = tokenizer.eos_token_id = 200020
        generation_config.pad_token_id = tokenizer.pad_token_id = 200020

    generation_config.do_sample = True if num_samples > 1 else False
    logger.info(f"Generating do_sample: {generation_config.do_sample}")
    # if model_name in ["llama-3.2-1b-it", "llama-3.2-3b-it"]:
    if generation_config.do_sample:
        set_seed(seed)
        generation_config.temperature = temperature
        generation_config.top_k = None
        generation_config.top_p = 1.0
        generation_config.num_return_sequences = num_samples
    else:
        generation_config.temperature = None
        generation_config.top_k = None
        generation_config.top_p = None

    gen_args['generation_config'] = generation_config
    logger.info(f'Generating responses w/ generation config: {s.i(model_generation_config2dict(generation_config), indent=1)}')
    
    local_responses = []
    local_hidden_activations = []
    local_mask_list = []

    it = tqdm(dataloader, desc=f'Process {accelerator.process_index} Generating responses', position=accelerator.process_index)
    for i, batch_encoded_input in enumerate(it):
        batch_encoded_input: Dict[str, Any]
        input_ids = batch_encoded_input['input_ids'].to(device)
        attention_mask = batch_encoded_input['attention_mask'].to(device)
        prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        with torch.no_grad():
            gen_args.update(inputs=input_ids, attention_mask=attention_mask)
            outputs = accelerator.unwrap_model(model).generate(**gen_args)

        generated_response = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        if num_samples > 1:
            it_res = iter(generated_response)
            for ppt, msgs in zip(prompt, batch_encoded_input["prompt_messages"]):
                for ns in range(num_samples):
                    res = next(it_res).removeprefix(ppt)
                    local_responses.append(dict(prompt=msgs, response=res))
        else:
            for ppt, generated_response, msgs in zip(prompt, generated_response, batch_encoded_input['prompt_messages']):
                res = generated_response.removeprefix(ppt)  # remove the prompt from the generated response
                local_responses.append(dict(prompt=msgs, response=res)) 

        if num_samples > 1:
            input_ids = input_ids.repeat_interleave(num_samples, dim=0)

        hidden_states = get_last_layer_output_token_hidden_states(outputs.hidden_states)

        if model_name == 'llama-3.2-1b-it' or model_name == "llama-3.2-3b-it":
            assert tokenizer.pad_token_id == tokenizer.eos_token_id == 128009  # sanity check

            # note not all responses finish at the same generated token count
            # for each generated sequence, get the number of padding tokens (i.e. after end of actually generated tokens)
            #   this is used to mask out further padding tokens for value function training
            # we want only the padding tokens after the response, not the left-padded ones before the prompt
            pad_id = tokenizer.pad_token_id
            length_of_prompts_padding = (input_ids == pad_id).sum(dim=1)
            pad_len_all = (outputs.sequences == pad_id).sum(dim=1)
            padding_length = pad_len_all - length_of_prompts_padding

            n_finish += (padding_length.size(0) - (padding_length == 0).sum()).item()
        elif model_name == 'phi-4-mini-it':
            assert tokenizer.pad_token_id == tokenizer.eos_token_id == 200020 # sanity check

            pad_id = tokenizer.pad_token_id
            length_of_prompts_padding = (input_ids == pad_id).sum(dim=1)
            pad_len_all = (outputs.sequences == pad_id).sum(dim=1)
            padding_length = pad_len_all - length_of_prompts_padding
            n_finish += (padding_length.size(0) - (padding_length == 0).sum()).item()
        else:
            raise NotImplementedError

        range_tensor = torch.arange(len(hidden_states)).expand(hidden_states[0].shape[0], -1)

        thresholds = (len(hidden_states) - padding_length).unsqueeze(1)
        thresholds = thresholds.to(device)
        range_tensor = range_tensor.to(device)
        mask = range_tensor < thresholds

        mask = mask.int()

        local_hidden_activations.append([h.cpu() for h in hidden_states])
        local_mask_list.append(mask.cpu())

        it.set_postfix({'|prompt|': s.i(input_ids.shape[1]), '#generated-token': f'{s.i(len(hidden_states))}/{s.i(max_new_tokens)}'})
        
        del outputs, hidden_states, input_ids, attention_mask
        if 'range_tensor' in locals(): del range_tensor
        if 'thresholds' in locals(): del thresholds
        if 'mask' in locals(): del mask
    # print(f"Process {accelerator.process_index} start saving")

    max_length = max(len(hidden) for hidden in local_hidden_activations)

    padded_hiddens = []
    padded_mask = [F.pad(mask, (0, max_length - mask.shape[1])) for mask in local_mask_list]
    mask = torch.cat(padded_mask, dim=0)
    for hidden in local_hidden_activations:
        stacked = torch.stack(hidden, dim=0)
        stacked = F.pad(stacked, (0, 0, 0, 0, 0, max_length - stacked.shape[0])).transpose(0, 1)
        padded_hiddens.append(stacked)
    local_hidden_activations = torch.cat(padded_hiddens, dim=0)
    local_hidden_activations = local_hidden_activations.detach().cpu()

    
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    world_size = accelerator.num_processes

    gathered_responses = [None] * world_size
    all_gather_object(gathered_responses, local_responses)

    gloo_group = torch.distributed.new_group(backend="gloo")
    gathered_hidden = [None] * world_size
    all_gather_object(gathered_hidden, local_hidden_activations, group=gloo_group)

    gathered_masks = [None] * world_size
    all_gather_object(gathered_masks, mask)

    all_responses = list(chain.from_iterable(gathered_responses))
    # all_hidden    = torch.cat(gathered_hidden, dim=0)
    # all_masks     = torch.cat(gathered_masks, dim=0)

    global_max_length = max(t.shape[1] for t in gathered_hidden)

    # 对 hidden_states 进行全局 padding 对齐
    padded_gathered_hidden = []
    for t in gathered_hidden:
        pad_len = global_max_length - t.shape[1]
        # F.pad 从最后往前推: (pad_last_dim_left, pad_last_dim_right, pad_second_last_left, pad_second_last_right)
        # 我们要给 seq_len (倒数第二维) 的右边 padding
        padded_t = F.pad(t, (0, 0, 0, pad_len), value=0.0) 
        padded_gathered_hidden.append(padded_t)
    
    all_hidden = torch.cat(padded_gathered_hidden, dim=0)

    # 对 masks 进行同样的全局 padding 对齐
    # t 的形状是 [batch_size, seq_len]
    padded_gathered_masks = []
    for t in gathered_masks:
        pad_len = global_max_length - t.shape[1]
        # mask 是 2D 张量，只需对倒数第一维 (seq_len) 右边 pad
        padded_t = F.pad(t, (0, pad_len), value=0.0)
        padded_gathered_masks.append(padded_t)
        
    all_masks = torch.cat(padded_gathered_masks, dim=0)

    if accelerator.is_main_process:
        torch.save(all_hidden, os_join(base_output_path, f'token_wise_activations_{mode}.pth'))
        time.sleep(1)
        torch.save(all_masks, os_join(base_output_path, f'mask_{mode}.pth'))
        time.sleep(1)
        with open(os_join(base_output_path, f'response_{mode}.jsonl'), 'w') as f:
            for res in all_responses:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')


def main():
    from rich.traceback import install
    install(show_locals=False)

    sic.output_width = 128

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='phi-4-mini-it')
    parser.add_argument('--dataset_name', type=str, default='HelpSteer2')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_test_split', type=float, default=0.2, help='Fraction of the dataset to use for Value Function testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset splitting')
    parser.add_argument('--max_prompt_tokens', type=int, default=2048, help='Maximum number of tokens in a sample, beyond which it is filtered out')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of tokens for the LLM to generate')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for generation')
    parser.add_argument('--output_path', type=str, default='./data/value_function_features/HelpSteer2/ns=1/temp=0.0/phi-4-mini-it', help='Path to save the features')
    parser.add_argument('--hf_cache_dir', type=str, default='./cache', help='Huggingface cache directory')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature for LLM inference')
    args = parser.parse_args()

    base_output_path = args.output_path
    os.makedirs(base_output_path, exist_ok=True)

    log_path = os_join(base_output_path, 'get-vf-dset.log')
    logger = get_logger('Get-VF-Train-Dataset', kind='both+ansi', file_path=log_path)
    logger.info(f'Getting Value Function train/test dataset w/ {s.i(vars(args), indent=1)}')

    style_transformers_logging(log_level='warning')

    model_name_map = {
        'llama-3.2-1b-it': 'meta-llama/Llama-3.2-1B-Instruct',
        'llama-3.2-3b-it': 'meta-llama/Llama-3.2-3B-Instruct',
        'phi-4-mini-it': "microsoft/Phi-4-mini-instruct"
    }
    model_name = model_name_map[args.model_name]

    cache_dir = args.hf_cache_dir or None

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', cache_dir=cache_dir)
    d_tok = {k: getattr(tokenizer, k) for k in ['model_max_length', 'pad_token', 'pad_token_id', 'eos_token', 'eos_token_id']}
    logger.info(f'Tokenizer loaded w/: {s.i(d_tok, indent=True)}')

    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, \
                                        cache_dir=cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        

    if args.dataset_name == 'HelpSteer2':
        dset: datasets.DatasetDict = load_dataset("nvidia/HelpSteer2", cache_dir=cache_dir)

        for split, dataset in dset.items():
            n = len(dataset)
            # sanity check prompts are paired
            assert dataset.select(range(0, n, 2))["prompt"] == dataset.select(range(1, n, 2))["prompt"]
            # take the even rows as prompts since each prompt is duplicated 2X for 2 responses
            dset[split] = dataset.select(range(0, len(dataset), 2))
        dataset = dset
    elif args.dataset_name == 'CodeUltraFeedback':
        dset: datasets.DatasetDict = load_dataset("coseal/CodeUltraFeedback", cache_dir=cache_dir)
        
        splits= dset['train'].train_test_split(test_size=0.1, seed=args.seed)
        dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})
    else:
        raise NotImplementedError

    def tokenize(example):
        # convert to multi-turn messages & use the chat template
        if args.dataset_name == 'HelpSteer2':
            example['prompt_messages'] = msgs = helpsteer2_prompt2messages(example['prompt'])
            example['multi-turn'] = len(msgs) > 1
            
            example['input_ids'] = tokenized = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
            example['attention_mask'] = [1] * len(tokenized)

            return example
        elif args.dataset_name == 'CodeUltraFeedback':
            example['prompt_messages'] = ultrafeedback_prompt2messages(example["instruction"])

            example['input_ids'] = tokenized = tokenizer.apply_chat_template(example['prompt_messages'], add_generation_prompt=True)
            example['attention_mask'] = [1] * len(tokenized)

            return example
        else:
            raise NotImplementedError

    dataset = dataset.map(tokenize, batched=False)
    if args.dataset_name == 'HelpSteer2':
        n_multi_turn = {split: sum(1 for sample in dset if sample["multi-turn"]) for split, dset in dataset.items()}
        logger.info(f'#multi-turn conversations : {s.i(n_multi_turn, indent=True)}')

    # Filter for tokens within 2048 limit
    # TODO: why this filtering?
    #   => count the number of samples dropped for filtering
    n_tr, n_ts = (len(dset) for dset in dataset.values())

    dataset = dataset.filter(lambda x: len(x["input_ids"]) <= args.max_prompt_tokens)
    n = {split: f'{s.i(n_)} -> {s.i(len(dset))}' for n_, (split, dset) in zip([n_tr, n_ts], dataset.items())}
    logger.info(f'Filtered samples: {s.i(n, indent=True)}')

    data_collator = DataCollatorReward(tokenizer=tokenizer)

    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    train_sampler = UnevenDistributedSampler(dataset["train"], shuffle=False, seed=42, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    test_sampler = UnevenDistributedSampler(dataset["validation"], shuffle=False, seed=42, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    train_dataloader = DataLoader(dataset["train"], batch_size=args.batch_size, collate_fn=data_collator, sampler=train_sampler)
    test_dataloader = DataLoader(dataset["validation"], batch_size=args.batch_size, collate_fn=data_collator, sampler=test_sampler)

    logger.info(f'Generating responses & processing hidden states for the {s.i("train")} split... ')
    args_ = dict(max_new_tokens=args.max_new_tokens, logger=logger, base_output_path=base_output_path, temperature=args.temperature)

    model = accelerator.prepare(model)

    get_llm_activations(
        args.model_name, model, train_dataloader, tokenizer, device, accelerator, args.num_samples, mode="train", **args_
    )

    get_llm_activations(
        args.model_name, model, test_dataloader, tokenizer, device, accelerator, args.num_samples, mode="test", **args_
    )



if __name__ == "__main__":
    main()
