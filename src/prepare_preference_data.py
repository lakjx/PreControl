"""
Prepare preference data from HelpSteer2 for DPO training.
HelpSteer2 has paired responses per prompt - we use ArmoRM to score them
and compute alpha vectors for the 2 target attributes (helpfulness + verbosity).
"""
import torch
import argparse
import json
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from src.util.data import helpsteer2_prompt2messages

# Which ArmoRM dimensions to use (HelpSteer2 = first 5)
# We pick 2 for now: helpfulness (idx 0) and verbosity (idx 4)
ATTR_INDICES = [0, 4]
ATTR_NAMES = ['helpfulness', 'verbosity']

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_for_rm(batch):
    """Prepare messages for ArmoRM scoring. Returns two lists: resp_a messages, resp_b messages."""
    msgs_a, msgs_b, meta = [], [], []
    for item in batch:
        prompt_msgs = item['prompt_msgs']
        msgs_a.append(prompt_msgs + [{"role": "assistant", "content": item['response_a']}])
        msgs_b.append(prompt_msgs + [{"role": "assistant", "content": item['response_b']}])
        meta.append(item)
    return msgs_a, msgs_b, meta

@torch.no_grad()
def score_batch(messages_list, model, tokenizer):
    """Score with ArmoRM. Returns [batch, 19] rewards."""
    device = next(model.parameters()).device
    inputs = tokenizer.apply_chat_template(messages_list, padding=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    return out.rewards  # [batch, 19]

def normalize_alpha(alpha_raw):
    norm = torch.norm(alpha_raw, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    return alpha_raw / norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='HelpSteer2')
    parser.add_argument('--output_path', type=str, default='data/helpsteer2_prefs_2attr.jsonl')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_prompt_tokens', type=int, default=1024, help='Filter out prompts longer than this')
    parser.add_argument('--hf_cache_dir', type=str, default='./cache')
    args = parser.parse_args()

    # ── Load ArmoRM ──
    rm_name = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
    print(f"Loading {rm_name}...")
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name, use_fast=True, cache_dir=args.hf_cache_dir)
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.hf_cache_dir
    ).cuda().eval()

    # ── Load & process HelpSteer2 ──
    print("Loading HelpSteer2...")
    ds = load_dataset("nvidia/HelpSteer2", cache_dir=args.hf_cache_dir)

    # Also load a tokenizer for length filtering
    gen_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-4-mini-instruct", cache_dir=args.hf_cache_dir
    )
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    pairs = []
    for split in ds:
        dataset = ds[split]
        n = len(dataset)
        # HelpSteer2 has paired responses: even and odd rows share the same prompt
        for i in range(0, n - 1, 2):
            row_a = dataset[i]
            row_b = dataset[i + 1]
            if row_a['prompt'] != row_b['prompt']:
                continue  # skip if not actually paired

            prompt_msgs = helpsteer2_prompt2messages(row_a['prompt'])

            # Length filter
            prompt_tokens = gen_tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)
            if len(prompt_tokens) > args.max_prompt_tokens:
                continue

            pairs.append({
                'prompt_msgs': prompt_msgs,
                'prompt_raw': row_a['prompt'],
                'response_a': row_a['response'],
                'response_b': row_b['response'],
            })

    print(f"Total valid pairs: {len(pairs)}")

    dataset = PairDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_for_rm)

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)

    print(f"Scoring and saving to {args.output_path}...")
    n_written = 0
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for msgs_a, msgs_b, meta in tqdm(dataloader):
            scores_a = score_batch(msgs_a, rm_model, rm_tokenizer)  # [batch, 19]
            scores_b = score_batch(msgs_b, rm_model, rm_tokenizer)

            # Extract only our target attributes
            scores_a_sel = scores_a[:, ATTR_INDICES].cpu().float()  # [batch, 2]
            scores_b_sel = scores_b[:, ATTR_INDICES].cpu().float()

            # Determine chosen/rejected by total RM score (sum of selected attrs)
            total_a = scores_a_sel.sum(dim=1)
            total_b = scores_b_sel.sum(dim=1)

            for j in range(len(meta)):
                if total_a[j] > total_b[j]:
                    chosen_msgs = msgs_a[j]
                    rejected_msgs = msgs_b[j]
                    s_w = scores_a_sel[j]
                    s_l = scores_b_sel[j]
                elif total_b[j] > total_a[j]:
                    chosen_msgs = msgs_b[j]
                    rejected_msgs = msgs_a[j]
                    s_w = scores_b_sel[j]
                    s_l = scores_a_sel[j]
                else:
                    continue  # skip ties

                alpha_raw = s_w - s_l
                alpha_norm = normalize_alpha(alpha_raw.unsqueeze(0)).squeeze(0)

                record = {
                    "prompt": meta[j]['prompt_msgs'],
                    "chosen": chosen_msgs,
                    "rejected": rejected_msgs,
                    "scores_w": s_w.tolist(),
                    "scores_l": s_l.tolist(),
                    "alpha_raw": alpha_raw.tolist(),
                    "alpha": alpha_norm.tolist(),
                    "attr_names": ATTR_NAMES,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                n_written += 1

    print(f"Done! Wrote {n_written} preference pairs.")

if __name__ == "__main__":
    main()
