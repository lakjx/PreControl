"""
Prepare preference data from HelpSteer2 for DPO training.
Uses Accelerator for multi-GPU RM scoring.

Usage:
    accelerate launch src/prepare_preference_data.py --batch_size 32
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from src.util.data import helpsteer2_prompt2messages

# HelpSteer2 attributes in ArmoRM: first 5 dims
# We pick 2: helpfulness (idx 0) and verbosity (idx 4)
ATTR_INDICES = [0, 4]
ATTR_NAMES = ['helpfulness', 'verbosity']


# ──────────────────────────────────────────────────
# Step 1: Build paired dataset using .map()
# ──────────────────────────────────────────────────
def build_pairs(ds, gen_tokenizer, max_prompt_tokens=1024):
    """
    HelpSteer2 has paired rows: rows 2i and 2i+1 share the same prompt.
    Use .select() + .map() to build pairs efficiently.
    """
    all_pairs = []

    for split in ds:
        dataset = ds[split]
        n = len(dataset)

        # Select even and odd rows
        even_indices = list(range(0, n, 2))
        odd_indices = list(range(1, n, 2))
        min_len = min(len(even_indices), len(odd_indices))
        even_indices = even_indices[:min_len]
        odd_indices = odd_indices[:min_len]

        rows_a = dataset.select(even_indices)
        rows_b = dataset.select(odd_indices)

        # Vectorized prompt matching & filtering via .map()
        def process_pair(example_a, idx):
            example_b = rows_b[idx]
            if example_a['prompt'] != example_b['prompt']:
                return {'valid': False, 'prompt_msgs': '', 'prompt_raw': '',
                        'response_a': '', 'response_b': ''}

            prompt_msgs = helpsteer2_prompt2messages(example_a['prompt'])

            # Length filter
            prompt_tokens = gen_tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)
            if len(prompt_tokens) > max_prompt_tokens:
                return {'valid': False, 'prompt_msgs': '', 'prompt_raw': '',
                        'response_a': '', 'response_b': ''}

            return {
                'valid': True,
                'prompt_msgs': json.dumps(prompt_msgs, ensure_ascii=False),
                'prompt_raw': example_a['prompt'],
                'response_a': example_a['response'],
                'response_b': example_b['response'],
            }

        processed = rows_a.map(process_pair, with_indices=True, num_proc=4,
                               remove_columns=rows_a.column_names)
        processed = processed.filter(lambda x: x['valid'], num_proc=4)

        for row in processed:
            all_pairs.append({
                'prompt_msgs': json.loads(row['prompt_msgs']),
                'prompt_raw': row['prompt_raw'],
                'response_a': row['response_a'],
                'response_b': row['response_b'],
            })

    return all_pairs


# ──────────────────────────────────────────────────
# Step 2: RM scoring dataset & collator
# ──────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch):
    """Build chat messages for ArmoRM. Returns msgs_a, msgs_b, metadata."""
    msgs_a, msgs_b, meta = [], [], []
    for item in batch:
        pm = item['prompt_msgs']
        msgs_a.append(pm + [{"role": "assistant", "content": item['response_a']}])
        msgs_b.append(pm + [{"role": "assistant", "content": item['response_b']}])
        meta.append(item)
    return msgs_a, msgs_b, meta


@torch.no_grad()
def score_batch(messages_list, model, tokenizer):
    """Score with ArmoRM, returns [batch, 19] rewards."""
    device = next(model.parameters()).device
    inputs = tokenizer.apply_chat_template(
        messages_list, padding=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    return out.rewards  # [batch, 19]


def normalize_alpha(alpha_raw):
    norm = torch.norm(alpha_raw, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    return alpha_raw / norm


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='data/helpsteer2_prefs_2attr.jsonl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_prompt_tokens', type=int, default=1024)
    parser.add_argument('--hf_cache_dir', type=str, default='./cache')
    args = parser.parse_args()

    accelerator = Accelerator()

    # ── Load ArmoRM (each process loads on its own GPU) ──
    rm_name = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
    if accelerator.is_main_process:
        print(f"Loading {rm_name}...")

    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name, use_fast=True, cache_dir=args.hf_cache_dir)
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.hf_cache_dir
    )
    rm_model.eval()

    # ── Build pairs on main process, then broadcast ──
    if accelerator.is_main_process:
        print("Loading HelpSteer2 & building pairs...")
        ds = load_dataset("nvidia/HelpSteer2", cache_dir=args.hf_cache_dir)
        gen_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-mini-instruct", cache_dir=args.hf_cache_dir
        )
        if gen_tokenizer.pad_token is None:
            gen_tokenizer.pad_token = gen_tokenizer.eos_token

        pairs = build_pairs(ds, gen_tokenizer, max_prompt_tokens=args.max_prompt_tokens)
        print(f"Total valid pairs: {len(pairs)}")

        # Save pairs to temp file for other processes
        tmp_path = '/tmp/_preference_pairs.json'
        with open(tmp_path, 'w') as f:
            json.dump(pairs, f, ensure_ascii=False)

    accelerator.wait_for_everyone()

    # All processes load pairs
    tmp_path = '/tmp/_preference_pairs.json'
    with open(tmp_path, 'r') as f:
        pairs = json.load(f)

    # ── Create DataLoader and distribute with Accelerator ──
    dataset = PairDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    rm_model, dataloader = accelerator.prepare(rm_model, dataloader)

    # ── Score all pairs ──
    local_records = []

    for msgs_a, msgs_b, meta in tqdm(dataloader, desc=f"GPU{accelerator.process_index}", disable=not accelerator.is_local_main_process):
        # Unwrap model for custom forward
        unwrapped = accelerator.unwrap_model(rm_model)

        scores_a = score_batch(msgs_a, unwrapped, rm_tokenizer)  # [batch, 19]
        scores_b = score_batch(msgs_b, unwrapped, rm_tokenizer)

        scores_a_sel = scores_a[:, ATTR_INDICES].cpu().float()
        scores_b_sel = scores_b[:, ATTR_INDICES].cpu().float()

        total_a = scores_a_sel.sum(dim=1)
        total_b = scores_b_sel.sum(dim=1)

        for j in range(len(meta)):
            if total_a[j] > total_b[j]:
                chosen_msgs, rejected_msgs = msgs_a[j], msgs_b[j]
                s_w, s_l = scores_a_sel[j], scores_b_sel[j]
            elif total_b[j] > total_a[j]:
                chosen_msgs, rejected_msgs = msgs_b[j], msgs_a[j]
                s_w, s_l = scores_b_sel[j], scores_a_sel[j]
            else:
                continue

            alpha_raw = s_w - s_l
            alpha_norm = normalize_alpha(alpha_raw.unsqueeze(0)).squeeze(0)

            local_records.append({
                "prompt": meta[j]['prompt_msgs'],
                "chosen": chosen_msgs,
                "rejected": rejected_msgs,
                "scores_w": s_w.tolist(),
                "scores_l": s_l.tolist(),
                "alpha_raw": alpha_raw.tolist(),
                "alpha": alpha_norm.tolist(),
                "attr_names": ATTR_NAMES,
            })

    # ── Gather all records to main process ──
    from torch.distributed import all_gather_object, is_initialized
    if is_initialized():
        all_records = [None] * accelerator.num_processes
        all_gather_object(all_records, local_records)
    else:
        all_records = [local_records]

    if accelerator.is_main_process:
        merged = []
        for records in all_records:
            merged.extend(records)

        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        print(f"Done! Wrote {len(merged)} preference pairs to {args.output_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
