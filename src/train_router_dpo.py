"""
DPO Training for Preference Router with Accelerator multi-GPU support.
Features: train/val split, early stopping, gradient clipping, L2 + ortho regularization.

Usage:
    # Multi-GPU
    accelerate launch --num_processes 4 src/train_router_dpo.py --max_samples 2000 --epochs 5
    # Single GPU
    python src/train_router_dpo.py --max_samples 2000 --epochs 5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.preference_router import PreferenceRouter
from src.intervented_model.model_wrapper import PreferenceModelWrapper
import json
import argparse
import logging
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────
class PreferencePairDataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        return self.records[idx]

def load_data(path, max_samples=None):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data

def collate_fn(batch):
    """Identity collate: return the list of dicts as-is."""
    return batch

# ──────────────────────────────────────────────────
# Core DPO math
# ──────────────────────────────────────────────────
def get_batch_logprobs(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    vocab_size = shift_logits.size(-1)
    loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    return -loss.view(shift_labels.size())

def compute_dpo_loss(pi_logprobs, ref_logprobs, chosen_mask, rejected_mask, beta=0.1):
    N = pi_logprobs.shape[0] // 2
    pi_chosen  = (pi_logprobs[:N]  * chosen_mask).sum(dim=1)
    pi_rejected = (pi_logprobs[N:] * rejected_mask).sum(dim=1)
    ref_chosen  = (ref_logprobs[:N]  * chosen_mask).sum(dim=1)
    ref_rejected = (ref_logprobs[N:] * rejected_mask).sum(dim=1)

    pi_logratios  = pi_chosen - pi_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    with torch.no_grad():
        acc = (logits > 0).float().mean().item()
        reward_margin = logits.mean().item()

    metrics = dict(
        dpo_loss=loss.item(), dpo_acc=acc, reward_margin=reward_margin,
        pi_chosen=pi_chosen.mean().item(), pi_rejected=pi_rejected.mean().item(),
        ref_chosen=ref_chosen.mean().item(),
    )
    return loss, metrics

# ──────────────────────────────────────────────────
# Process one batch of items (multiple pairs per step)
# ──────────────────────────────────────────────────
def process_batch(items, model, tokenizer, device, beta, lambda_ortho, lambda_l2):
    """Process a batch of preference pairs, accumulate losses."""
    total_loss = torch.tensor(0.0, device=device)
    agg_metrics = {}
    count = 0

    for item in items:
        chosen = item['chosen']
        rejected = item['rejected']
        alpha = torch.tensor(item['alpha'], dtype=torch.bfloat16, device=device).unsqueeze(0)

        chosen_ids  = tokenizer.apply_chat_template(chosen, return_tensors='pt', add_generation_prompt=False).to(device)
        rejected_ids = tokenizer.apply_chat_template(rejected, return_tensors='pt', add_generation_prompt=False).to(device)

        max_len = max(chosen_ids.size(1), rejected_ids.size(1))

        def pad(t, m):
            if t.size(1) == m: return t
            return torch.cat([t, torch.full((1, m - t.size(1)), tokenizer.pad_token_id, device=device)], dim=1)

        input_ids = torch.cat([pad(chosen_ids, max_len), pad(rejected_ids, max_len)], dim=0)
        target_mask = (input_ids != tokenizer.pad_token_id).float()

        prompt_only = tokenizer.apply_chat_template([chosen[0]], return_tensors='pt', add_generation_prompt=True).to(device)
        target_mask[:, :prompt_only.size(1)] = 0.0

        # Reference logprobs (no router)
        with torch.no_grad():
            ref_out = model(input_ids, alpha=None)
            ref_logprobs = get_batch_logprobs(ref_out.logits, input_ids)

        # Policy logprobs (with router)
        pi_out = model(input_ids, alpha=alpha)
        pi_logprobs = get_batch_logprobs(pi_out.logits, input_ids)

        chosen_mask_t  = target_mask[0, 1:]
        rejected_mask_t = target_mask[1, 1:]

        loss_dpo, metrics = compute_dpo_loss(pi_logprobs, ref_logprobs, chosen_mask_t, rejected_mask_t, beta=beta)
        total_loss = total_loss + loss_dpo

        for k, v in metrics.items():
            agg_metrics[k] = agg_metrics.get(k, 0.0) + v
        count += 1

    # Average the loss over the batch
    total_loss = total_loss / count

    # Add ortho + L2 (these are per-router, not per-sample)
    loss_ortho = model.router.get_ortho_loss() * lambda_ortho
    total_loss = total_loss + loss_ortho

    loss_l2 = torch.tensor(0.0, device=device)
    if lambda_l2 > 0:
        loss_l2 = lambda_l2 * sum(p.pow(2).sum() for p in model.router.parameters())
        total_loss = total_loss + loss_l2

    agg_metrics = {k: v / count for k, v in agg_metrics.items()}
    agg_metrics['ortho_loss'] = loss_ortho.item()
    agg_metrics['l2_loss'] = loss_l2.item()
    agg_metrics['total_loss'] = total_loss.item()

    return total_loss, agg_metrics

# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-4-mini-instruct')
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--data_path_train', type=str, default='data/helpsteer2_prefs_2attr_train.jsonl')
    parser.add_argument('--data_path_val', type=str, default='data/helpsteer2_prefs_2attr_val.jsonl')
    parser.add_argument('--output_dir', type=str, default='checkpoints/preference_router')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.1,  help='DPO temperature')
    parser.add_argument('--lambda_ortho', type=float, default=0.01)
    parser.add_argument('--lambda_l2', type=float, default=0.01)
    parser.add_argument('--num_attributes', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--layer_idx', type=int, default=-2)
    parser.add_argument('--max_samples', type=int, default=0, help='0 = use all')
    parser.add_argument('--batch_size', type=int, default=4, help='Pairs per gradient step per GPU')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ── Accelerator setup ──
    accelerator = Accelerator()
    set_seed(args.seed)
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Load tokenizer & base model ──
    # NOTE: base model is FROZEN and NOT wrapped by DDP.
    # Only the router is trainable and synced across GPUs.
    if accelerator.is_main_process:
        logger.info(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16
    ).to(device)
    base_model.eval()

    router = PreferenceRouter(
        hidden_dim=base_model.config.hidden_size, rank=args.rank, num_attributes=args.num_attributes
    ).to(device).to(torch.bfloat16)

    model = PreferenceModelWrapper(base_model, router, layer_idx=args.layer_idx)
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=args.lr, weight_decay=0.01)

    # ── Load data ──
    train_data = load_data(args.data_path_train, max_samples=args.max_samples if args.max_samples > 0 else None)
    val_data = load_data(args.data_path_val, max_samples=args.max_samples if args.max_samples > 0 else None)
    
    if len(train_data) == 0:
        raise RuntimeError(f"No train data at {args.data_path_train}")
    if len(val_data) == 0:
        raise RuntimeError(f"No val data at {args.data_path_val}")

    train_dataset = PreferencePairDataset(train_data)
    val_dataset   = PreferencePairDataset(val_data)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Accelerator prepares: distributes data, wraps optimizer
    # We only prepare the router for gradient sync, not the frozen base model
    router, optimizer, train_loader, val_loader = accelerator.prepare(
        router, optimizer, train_loader, val_loader
    )
    # Re-attach the prepared (DDP-wrapped) router to the model wrapper
    model.router = accelerator.unwrap_model(router)

    if accelerator.is_main_process:
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, "
                     f"GPUs: {accelerator.num_processes}, Batch/GPU: {args.batch_size}")

    # ── Training loop ──
    best_val_loss = float('inf')
    n_no_improve = 0

    for epoch in range(args.epochs):
        # ── Train ──
        model.base_model.eval()
        train_agg, train_count = {}, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} train",
                    disable=not accelerator.is_main_process, leave=False)
        for batch_items in pbar:
            loss, metrics = process_batch(
                batch_items, model, tokenizer, device,
                args.beta, args.lambda_ortho, args.lambda_l2
            )
            optimizer.zero_grad()
            accelerator.backward(loss)
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.router.parameters(), max_norm=args.grad_clip)
            
            # Print grad norm for the very first step to ensure gradients are flowing
            if train_count == 0 and accelerator.is_main_process:
                total_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.router.parameters() if p.grad is not None) ** 0.5
                logger.info(f"First step grad norm: {total_grad_norm:.6f}")
                
            optimizer.step()

            for k, v in metrics.items():
                train_agg[k] = train_agg.get(k, 0.0) + v
            train_count += 1

            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{metrics['total_loss']:.4f}", acc=f"{metrics['dpo_acc']:.0%}")

        train_metrics = {k: v / train_count for k, v in train_agg.items()}

        # ── Validate ──
        val_agg, val_count = {}, 0
        with torch.no_grad():
            for batch_items in tqdm(val_loader, desc="val", disable=not accelerator.is_main_process, leave=False):
                _, metrics = process_batch(
                    batch_items, model, tokenizer, device,
                    args.beta, args.lambda_ortho, args.lambda_l2
                )
                for k, v in metrics.items():
                    val_agg[k] = val_agg.get(k, 0.0) + v
                val_count += 1

        val_metrics = {k: v / val_count for k, v in val_agg.items()} if val_count > 0 else {}

        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train: Loss={train_metrics['total_loss']:.4f} "
                f"(DPO:{train_metrics['dpo_loss']:.4f} Ortho:{train_metrics['ortho_loss']:.6f} L2:{train_metrics['l2_loss']:.6f}) "
                f"Acc={train_metrics['dpo_acc']:.2%} Margin={train_metrics['reward_margin']:.2f} | "
                f"Val: Loss={val_metrics.get('total_loss',0):.4f} Acc={val_metrics.get('dpo_acc',0):.2%}"
            )

            val_loss = val_metrics.get('total_loss', float('inf'))
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                n_no_improve = 0
                save_path = os.path.join(args.output_dir, 'best_router.pth')
                torch.save({
                    'router_state_dict': accelerator.unwrap_model(router).state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'val_acc': val_metrics.get('dpo_acc', 0),
                    'args': vars(args),
                }, save_path)
                logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                n_no_improve += 1
                logger.info(f"  No improvement ({n_no_improve}/{args.patience})")
                if n_no_improve >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

    if accelerator.is_main_process:
        logger.info("Training complete.")

if __name__ == '__main__':
    main()
