"""
DPO Training for Preference Router with:
- Train/Val split
- Early stopping with patience (best model saving)
- Gradient clipping
- L2 regularization on hidden state edits
- Orthogonal loss for attribute decoupling
- DPO accuracy & reward margin logging
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.preference_router import PreferenceRouter
from src.intervented_model.model_wrapper import PreferenceModelWrapper
import json
import argparse
import os
import logging
from tqdm import tqdm

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

# ──────────────────────────────────────────────────
# Core DPO math
# ──────────────────────────────────────────────────
def get_batch_logprobs(logits, labels):
    """Return token-level log-probs (negative cross-entropy) for labels."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    vocab_size = shift_logits.size(-1)
    loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    return -loss.view(shift_labels.size())

def compute_dpo_loss(pi_logprobs, ref_logprobs, chosen_mask, rejected_mask, beta=0.1):
    """
    Standard DPO loss.
    Returns: loss, dict of metrics
    """
    N = pi_logprobs.shape[0] // 2

    pi_chosen  = (pi_logprobs[:N]  * chosen_mask).sum(dim=1)
    pi_rejected = (pi_logprobs[N:] * rejected_mask).sum(dim=1)
    ref_chosen  = (ref_logprobs[:N]  * chosen_mask).sum(dim=1)
    ref_rejected = (ref_logprobs[N:] * rejected_mask).sum(dim=1)

    # Log-ratio margins
    pi_logratios  = pi_chosen - pi_rejected
    ref_logratios = ref_chosen - ref_rejected
    logits = pi_logratios - ref_logratios      # the "reward margin"

    loss = -F.logsigmoid(beta * logits).mean()

    # DPO accuracy: how often does the policy prefer chosen?
    with torch.no_grad():
        acc = (logits > 0).float().mean().item()
        reward_margin = logits.mean().item()

    metrics = dict(
        dpo_loss=loss.item(),
        dpo_acc=acc,
        reward_margin=reward_margin,
        pi_chosen=pi_chosen.mean().item(),
        pi_rejected=pi_rejected.mean().item(),
        ref_chosen=ref_chosen.mean().item(),
    )
    return loss, metrics

# ──────────────────────────────────────────────────
# Single step (shared by train & eval)
# ──────────────────────────────────────────────────
def run_step(item, model, tokenizer, device, beta, lambda_ortho, lambda_l2):
    """Run forward pass for one preference pair, return total loss & metrics."""
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

    # Mask out the prompt portion
    prompt_only = tokenizer.apply_chat_template([chosen[0]], return_tensors='pt', add_generation_prompt=True).to(device)
    target_mask[:, :prompt_only.size(1)] = 0.0

    # ── Reference logprobs (no router) ──
    with torch.no_grad():
        ref_out = model(input_ids, alpha=None)
        ref_logprobs = get_batch_logprobs(ref_out.logits, input_ids)

    # ── Policy logprobs (with router) ──
    pi_out = model(input_ids, alpha=alpha)
    pi_logprobs = get_batch_logprobs(pi_out.logits, input_ids)

    chosen_mask_t  = target_mask[0, 1:]
    rejected_mask_t = target_mask[1, 1:]

    # DPO loss
    loss_dpo, metrics = compute_dpo_loss(pi_logprobs, ref_logprobs, chosen_mask_t, rejected_mask_t, beta=beta)

    # Orthogonal loss
    loss_ortho = model.router.get_ortho_loss() * lambda_ortho

    # L2 regularization on the router edit magnitude
    # We measure the edit by doing a forward pass through just the router
    # and computing the norm of the delta
    loss_l2 = torch.tensor(0.0, device=device)
    if lambda_l2 > 0:
        # Get hidden states at the hook layer (sample a few tokens)
        with torch.no_grad():
            # Use the first input only to get representative hidden states
            sample_ids = input_ids[:1]
            # We temporarily disable the router to get clean hidden states
            old_alpha = model.current_alpha
            model.set_alpha(None)
            # We need to get the hidden states at the target layer
            # Use output_hidden_states=True
            clean_out = model.base_model(sample_ids, output_hidden_states=True)
            model.set_alpha(old_alpha)
            h_clean = clean_out.hidden_states[model.layer_idx]  # [1, seq, hidden]

        # Compute the router edit
        delta = model.router(h_clean.to(model.router.W.dtype), alpha) - h_clean.to(model.router.W.dtype)
        loss_l2 = lambda_l2 * (delta ** 2).mean()

    total_loss = loss_dpo + loss_ortho + loss_l2

    metrics['ortho_loss'] = loss_ortho.item()
    metrics['l2_loss'] = loss_l2.item()
    metrics['total_loss'] = total_loss.item()

    return total_loss, metrics

# ──────────────────────────────────────────────────
# Epoch runners
# ──────────────────────────────────────────────────
def run_epoch(data, model, tokenizer, device, optimizer, beta, lambda_ortho, lambda_l2,
              grad_clip, mode='train'):
    """Run one full epoch. Returns averaged metrics dict."""
    agg = {}
    count = 0

    for item in tqdm(data, desc=mode, leave=False):
        loss, metrics = run_step(item, model, tokenizer, device, beta, lambda_ortho, lambda_l2)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.router.parameters(), max_norm=grad_clip)
            optimizer.step()

        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1

    return {k: v / count for k, v in agg.items()}

# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-4-mini-instruct')
    parser.add_argument('--cache_dir', type=str, default='/home/zj-xz/data_trx/pre-control/cache')
    parser.add_argument('--data_path', type=str, default='data/helpsteer2_prefs_2attr.jsonl')
    parser.add_argument('--output_dir', type=str, default='checkpoints/preference_router')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta', type=float, default=0.1, help='DPO temperature')
    parser.add_argument('--lambda_ortho', type=float, default=0.01, help='Orthogonal loss weight')
    parser.add_argument('--lambda_l2', type=float, default=0.01, help='L2 edit magnitude penalty')
    parser.add_argument('--num_attributes', type=int, default=2, help='Number of preference attributes')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--rank', type=int, default=32, help='Router rank')
    parser.add_argument('--layer_idx', type=int, default=-2, help='Which layer to hook')
    parser.add_argument('--max_samples', type=int, default=0, help='0 = use all')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load tokenizer & model ──
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

    # ── Load & split data ──
    all_data = load_data(args.data_path, max_samples=args.max_samples if args.max_samples > 0 else None)
    logger.info(f"Loaded {len(all_data)} preference pairs total.")
    if len(all_data) == 0:
        raise RuntimeError(f"No data found at {args.data_path}. Run prepare_preference_data.py first.")

    n_val = max(1, int(len(all_data) * args.val_ratio))
    n_train = len(all_data) - n_val
    # Deterministic split
    indices = list(range(len(all_data)))
    torch.manual_seed(args.seed)
    perm = torch.randperm(len(all_data)).tolist()
    train_data = [all_data[i] for i in perm[:n_train]]
    val_data   = [all_data[i] for i in perm[n_train:]]
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # ── Training loop with early stopping ──
    best_val_loss = float('inf')
    n_no_improve = 0

    for epoch in range(args.epochs):
        model.base_model.eval()  # Always keep base frozen

        # Train
        train_metrics = run_epoch(
            train_data, model, tokenizer, device, optimizer,
            args.beta, args.lambda_ortho, args.lambda_l2, args.grad_clip, mode='train'
        )

        # Validate (no grad)
        with torch.no_grad():
            val_metrics = run_epoch(
                val_data, model, tokenizer, device, optimizer,
                args.beta, args.lambda_ortho, args.lambda_l2, args.grad_clip, mode='val'
            )

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_metrics['total_loss']:.4f} (DPO: {train_metrics['dpo_loss']:.4f}, "
            f"Ortho: {train_metrics['ortho_loss']:.4f}, L2: {train_metrics['l2_loss']:.4f}) "
            f"Acc: {train_metrics['dpo_acc']:.2%} Margin: {train_metrics['reward_margin']:.2f} | "
            f"Val Loss: {val_metrics['total_loss']:.4f} Acc: {val_metrics['dpo_acc']:.2%}"
        )

        # Early stopping check
        val_loss = val_metrics['total_loss']
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            n_no_improve = 0
            save_path = os.path.join(args.output_dir, 'best_router.pth')
            torch.save({
                'router_state_dict': model.router.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_acc': val_metrics['dpo_acc'],
                'args': vars(args),
            }, save_path)
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            n_no_improve += 1
            logger.info(f"  No improvement ({n_no_improve}/{args.patience})")
            if n_no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("Training complete.")

if __name__ == '__main__':
    main()
