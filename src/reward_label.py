"""
Label the responses in a file with an attributed RM

Used for
1. Training a value function model, and
2. Evaluate the reward of generated responses using VF representation editing
"""

import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from os.path import join as os_join
import json
from tqdm import tqdm
import os
from stefutil.prettier import get_logger, style as s, icecream as sic
from src.util import unwrap_model


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def data_collator(data, iteration):
    ret = []
    for sample in data:
        # append the new generated response to the prompt messasges
        prompt, response = sample['prompt'], sample['response']
        # sanity check prompt is already a list of chat messages
        assert isinstance(prompt, list)
        assert all(isinstance(msg, dict) and set(msg.keys()) == {'role', 'content'} for msg in prompt)
        assert isinstance(response, str)
        if iteration != 0:
            if "re-address" in prompt[-1]["content"]:
                prompt = prompt[:-2]
        ret.append(prompt + [dict(role='assistant', content=response)])
    return ret

def get_rm(data, rm_model, tokenizer, dataset):
    messages = data
    inputs = tokenizer.apply_chat_template(messages, padding=True, return_tensors="pt", return_dict=True).to(unwrap_model(rm_model).device)
    with torch.no_grad():
        rm_out = rm_model(**inputs)

    # total 19 attributes; first 5 are for helpsteer2; map back to 5-point scale
    if dataset == "HelpSteer2":
        rm_vals = rm_out.rewards[:, :5] * 5 - 0.5  # shape: [batch_size, num_attributes]
    elif dataset == "CodeUltraFeedback":
        rm_vals = rm_out.rewards[:, -5:] * 5 - 0.5

    return rm_vals


def main():
    from rich.traceback import install
    install(show_locals=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_model_name', type=str, default='ArmoRM-8B')
    parser.add_argument(
        '--response_path', type=str, default=os_join('localscratch', 'sp', 'features'),
        help='Path to the LLM-generated features, will also be used to store the labels'
    )
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for Reward Model inference')
    parser.add_argument('--hf_cache_dir', type=str, help='Huggingface cache directory')
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default="HelpSteer2")

    args = parser.parse_args()

    base_dir = os.path.dirname(args.response_path)
    log_path = os_join(base_dir, 'get-rm-scores.log')
    logger = get_logger('Get-Attr-RM-Scores', kind='both+ansi', file_path=log_path)
    logger.info(f'Getting attributed response reward scores w/ {s.i(vars(args), indent=1)}')

    model_name_map = {
        'ArmoRM-8B': 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
        # TODO: add reward models trained on HS2
    }
    reward_model_name = model_name_map[args.reward_model_name]

    cache_dir = args.hf_cache_dir or None

    tokenizer = AutoTokenizer.from_pretrained(reward_model_name, use_fast=True, cache_dir=cache_dir)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir
    )

    reward_model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_available_gpus = torch.cuda.device_count()
    logger.info(f'Using {s.i(torch.cuda.device_count())} GPUs...')
    if num_available_gpus > 1:
        reward_model = torch.nn.DataParallel(reward_model, device_ids=list(range(num_available_gpus)))

    reward_model = reward_model.to(device)

    if args.response_path.endswith('.json'):
        response_path = args.response_path.removesuffix('.json')
        with open(f'{response_path}.json', 'r') as f:
            lines = json.load(f)

    elif args.response_path.endswith('.jsonl'):
        response_path = args.response_path.removesuffix('.jsonl')
        with open(f'{response_path}.jsonl', 'r') as f:
            lines = [json.loads(line) for line in f]

    else:
        raise NotImplementedError

    dataset = ListDataset(lines)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: data_collator(batch, iteration=args.iteration))

    rm_scores = []

    for i, data in enumerate(tqdm(data_loader, desc='Getting reward labels')):
        rm_score = get_rm(data, reward_model, tokenizer, args.dataset_name)
        rm_scores.append(rm_score)

    rm_scores = torch.cat(rm_scores, dim=0)

    storage_path = f'{response_path}_scores.pth'
    logger.info(f'Written reward label tensor of shape {s.i(rm_scores.shape)} to {s.i(storage_path)}')

    # get the file path to save the results
    # save the tensor rm_scores to the file
    torch.save(rm_scores, storage_path)


if __name__ == "__main__":
    main()
