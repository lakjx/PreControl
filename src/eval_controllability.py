"""
Evaluate controllability: generate responses at varying alpha strengths,
score with ArmoRM, compute correlation with alpha direction.
Includes BASE MODEL (no router) as mandatory baseline.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from src.preference_router import PreferenceRouter
from src.intervented_model.model_wrapper import PreferenceModelWrapper
from tqdm import tqdm

ATTR_NAMES = ['helpfulness', 'verbosity']
ATTR_INDICES = [0, 4]  # ArmoRM dimension indices for HelpSteer2

def load_prompts(data_path, n=50):
    prompts = []
    seen = set()
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # prompt is stored as list of messages
            key = json.dumps(item['prompt'])
            if key not in seen:
                seen.add(key)
                prompts.append(item['prompt'])
                if len(prompts) >= n:
                    break
    return prompts

@torch.no_grad()
def score_with_armo(messages_list, rm_model, rm_tokenizer):
    device = next(rm_model.parameters()).device
    inputs = rm_tokenizer.apply_chat_template(messages_list, padding=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = rm_model(**inputs)
    return out.rewards[:, ATTR_INDICES].cpu().float()  # [batch, 2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-4-mini-instruct')
    parser.add_argument('--cache_dir', type=str, default='/home/zj-xz/data_trx/pre-control/cache')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/preference_router/best_router.pth')
    parser.add_argument('--data_path', type=str, default='data/helpsteer2_prefs_2attr.jsonl')
    parser.add_argument('--output_path', type=str, default='data/controllability_results.json')
    parser.add_argument('--n_prompts', type=int, default=20)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--num_attributes', type=int, default=2)
    parser.add_argument('--layer_idx', type=int, default=-2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load gen model + router ──
    print("Loading generation model + router...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16
    ).to(device).eval()

    router = PreferenceRouter(hidden_dim=base_model.config.hidden_size, rank=args.rank, num_attributes=args.num_attributes)
    router = router.to(device).to(torch.bfloat16)
    ckpt = torch.load(args.checkpoint, map_location=device)
    router.load_state_dict(ckpt['router_state_dict'])
    print(f"Router loaded from epoch {ckpt['epoch']}")

    model = PreferenceModelWrapper(base_model, router, layer_idx=args.layer_idx)

    # ── Load RM ──
    print("Loading ArmoRM...")
    rm_name = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_name, use_fast=True, cache_dir=args.cache_dir)
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id
    rm_model = AutoModelForSequenceClassification.from_pretrained(
        rm_name, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
    ).to(device).eval()

    # ── Load prompts ──
    prompts = load_prompts(args.data_path, n=args.n_prompts)
    print(f"Evaluating {len(prompts)} prompts...")

    # Alpha sweep: for each attribute, vary from -0.5 to 1.0
    # Plus baseline (no router)
    alpha_strengths = [-0.5, 0.0, 0.5, 1.0]

    results = []

    for prompt_msgs in tqdm(prompts, desc="Prompts"):
        input_ids = tokenizer.apply_chat_template(prompt_msgs, return_tensors='pt', add_generation_prompt=True).to(device)

        prompt_result = {"prompt": prompt_msgs, "experiments": []}

        # ── Baseline: no router ──
        output_ids = model.generate(input_ids, alpha=None, max_new_tokens=args.max_new_tokens, do_sample=False)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        conv = prompt_msgs + [{"role": "assistant", "content": response}]
        rm_scores = score_with_armo([conv], rm_model, rm_tokenizer)[0].tolist()
        prompt_result["experiments"].append({
            "attribute": "baseline", "alpha_strength": 0.0,
            "alpha_vec": [0.0, 0.0],
            "response_len": len(new_tokens),
            "rm_scores": {ATTR_NAMES[i]: round(rm_scores[i], 4) for i in range(len(ATTR_NAMES))},
            "response_preview": response[:200],
        })

        # ── Per-attribute alpha sweep ──
        for attr_idx in range(len(ATTR_NAMES)):
            for strength in alpha_strengths:
                alpha_vec = [0.0] * len(ATTR_NAMES)
                alpha_vec[attr_idx] = strength
                alpha = torch.tensor([alpha_vec], dtype=torch.bfloat16, device=device)

                output_ids = model.generate(input_ids, alpha=alpha, max_new_tokens=args.max_new_tokens, do_sample=False)
                new_tokens = output_ids[0, input_ids.shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)

                conv = prompt_msgs + [{"role": "assistant", "content": response}]
                rm_scores = score_with_armo([conv], rm_model, rm_tokenizer)[0].tolist()

                prompt_result["experiments"].append({
                    "attribute": ATTR_NAMES[attr_idx],
                    "alpha_strength": strength,
                    "alpha_vec": alpha_vec,
                    "response_len": len(new_tokens),
                    "rm_scores": {ATTR_NAMES[i]: round(rm_scores[i], 4) for i in range(len(ATTR_NAMES))},
                    "response_preview": response[:200],
                })

        results.append(prompt_result)

    # ── Analysis ──
    print("\n" + "="*60)
    print("CONTROLLABILITY ANALYSIS")
    print("="*60)

    # Collect baseline scores
    baseline_scores = {attr: [] for attr in ATTR_NAMES}
    for pr in results:
        bl = [e for e in pr["experiments"] if e["attribute"] == "baseline"][0]
        for attr in ATTR_NAMES:
            baseline_scores[attr].append(bl["rm_scores"][attr])

    for attr_idx, attr_name in enumerate(ATTR_NAMES):
        alphas_all, scores_all = [], []
        for pr in results:
            for exp in pr["experiments"]:
                if exp["attribute"] == attr_name:
                    alphas_all.append(exp["alpha_strength"])
                    scores_all.append(exp["rm_scores"][attr_name])

        alphas_all = np.array(alphas_all)
        scores_all = np.array(scores_all)

        # Pearson
        if np.std(alphas_all) > 0 and np.std(scores_all) > 0:
            corr = np.corrcoef(alphas_all, scores_all)[0, 1]
        else:
            corr = float('nan')

        # Spearman
        from scipy.stats import spearmanr
        spear_corr, spear_p = spearmanr(alphas_all, scores_all)

        mean_baseline = np.mean(baseline_scores[attr_name])
        mean_by_strength = {}
        for sv in alpha_strengths:
            mask = alphas_all == sv
            mean_by_strength[sv] = scores_all[mask].mean() if mask.any() else float('nan')

        print(f"\n{attr_name.upper()}:")
        print(f"  Baseline (no router): mean={mean_baseline:.4f}")
        print(f"  Pearson r = {corr:.4f}")
        print(f"  Spearman ρ = {spear_corr:.4f} (p={spear_p:.4e})")
        for sv, ms in sorted(mean_by_strength.items()):
            delta = ms - mean_baseline
            print(f"  α={sv:+.1f} → mean={ms:.4f} (Δ={delta:+.4f} vs baseline)")

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output_path}")

if __name__ == '__main__':
    main()
