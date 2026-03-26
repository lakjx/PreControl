"""
Inference demo: Generate responses under different alpha vectors (2 attributes).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.preference_router import PreferenceRouter
from src.intervented_model.model_wrapper import PreferenceModelWrapper

ATTR_NAMES = ['helpfulness', 'verbosity']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-4-mini-instruct')
    parser.add_argument('--cache_dir', type=str, default='/home/zj-xz/data_trx/pre-control/cache')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/preference_router/best_router.pth')
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--num_attributes', type=int, default=2)
    parser.add_argument('--layer_idx', type=int, default=-2)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16
    ).to(device).eval()

    router = PreferenceRouter(
        hidden_dim=base_model.config.hidden_size, rank=args.rank, num_attributes=args.num_attributes
    ).to(device).to(torch.bfloat16)

    ckpt = torch.load(args.checkpoint, map_location=device)
    router.load_state_dict(ckpt['router_state_dict'])
    print(f"Loaded router from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.2%})")

    model = PreferenceModelWrapper(base_model, router, layer_idx=args.layer_idx)

    test_prompts = [
        "Explain the concept of machine learning to a beginner.",
        "Write a short poem about the ocean.",
        "What are the pros and cons of remote work?",
    ]

    # 2-attribute alpha configs: [helpfulness, verbosity]
    alpha_configs = {
        "baseline (no router)":      None,
        "high helpfulness":          [1.0, 0.0],
        "high verbosity":            [0.0, 1.0],
        "helpful + verbose":         [0.7, 0.7],
        "low verbosity (concise)":   [0.0, -1.0],
        "neg helpfulness":           [-1.0, 0.0],
    }

    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}")

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt', add_generation_prompt=True).to(device)

        for config_name, alpha_vals in alpha_configs.items():
            alpha = torch.tensor([alpha_vals], dtype=torch.bfloat16, device=device) if alpha_vals else None

            with torch.no_grad():
                output_ids = model.generate(input_ids, alpha=alpha, max_new_tokens=args.max_new_tokens, do_sample=False)

            new_tokens = output_ids[0, input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)

            print(f"\n── [{config_name}] ──")
            if alpha_vals:
                print(f"   α = [{', '.join(f'{ATTR_NAMES[i]}={alpha_vals[i]:+.1f}' for i in range(len(ATTR_NAMES)))}]")
            else:
                print(f"   α = disabled")
            print(f"   ({len(new_tokens)} tokens): {response[:500]}{'...' if len(response) > 500 else ''}")

if __name__ == '__main__':
    main()
