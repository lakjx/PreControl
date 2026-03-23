# We first need to train a value function 
# The value function is a MLP on top of the llama features
# We need to first load the HH-RLHF dataset from huggingface and initialize the model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import wandb
import logging
from os.path import join as os_join
from tqdm.auto import trange
from tqdm import tqdm
import pickle

from stefutil.prettier import icecream as sic, style as s, get_logger, now
from src.util import set_seed, argparse_str2int_list
from src.value_function import ValueFunction


def default_list():
    return [0, 0, 0, 0, 0]

class Trainer:
    def __init__(
            self,
            model: nn.Module = None,
            train_dataloader: DataLoader = None,
            test_dataloader: DataLoader = None,
            optimizer: str = 'adam',
            scheduler: str = None,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.0,
            num_train_epochs: int = 100,
            grad_norm_clipping: float = 1.0,
            patience: int = 10,
            lambda_param: float = 0.99,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            logger: logging.Logger = None,
            model_output_dir: str = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.patience = patience
        self.lambda_param = lambda_param

        optim_cls = dict(adam=optim.Adam, adamw=optim.AdamW)[optimizer]
        self.optimizer = optim_cls(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.grad_norm_clipping = grad_norm_clipping

        if scheduler == 'cosine':
            n_step = len(train_dataloader) * num_train_epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=n_step, eta_min=0.05 * learning_rate)
        else:
            self.scheduler = None

        self.epoch_it = None
        self.global_step = 0

        self.logger = logger
        self.model_output_dir = model_output_dir


    def run_epoch_tdlambda(self, mode: str = 'train'):
        if mode == 'train':
            self.model.train()
            dataloader = self.train_dataloader
        else:
            self.model.eval()
            dataloader = self.test_dataloader
        total_loss = 0
        count_batches = 0
        
        for batch_input in dataloader:
            if mode == 'train':
                self.optimizer.zero_grad()
            
            hidden_states, labels, mask = batch_input
            hidden_states = hidden_states.to(self.device)
            assert isinstance(labels, torch.Tensor)
            labels = labels.to(self.device)
            mask = mask.to(self.device)
            batch_size, length, hidden_dim = hidden_states.shape

            all_predictions = self.model(hidden_states.view(-1, hidden_dim)).view(batch_size, length, -1)
            
            G_lambda = self.backward_td_lambda(all_predictions, labels, mask)

            last_indices = mask.sum(dim=1, keepdim=True) - 1 # shape (batch_size, 1)
            valid_mask = mask.sum(dim=1) != 0
            batch_indices = torch.arange(batch_size).to(last_indices.device)
            last_valid_preds = all_predictions[batch_indices, last_indices.squeeze()]
            final_loss = F.mse_loss(last_valid_preds[valid_mask], labels[valid_mask], reduction='none')

            G_lambda = G_lambda[valid_mask]
            all_predictions = all_predictions[valid_mask]
            mask = mask[valid_mask]

            G_lambda = G_lambda[mask.bool(), :]
            valid_preds = all_predictions[mask.bool(), :]
            
            loss = F.mse_loss(valid_preds, G_lambda, reduction='none')
            loss = torch.sum(loss)

            loss = loss / batch_size

            if mode == "train":
                loss.backward()
                if self.grad_norm_clipping:
                    norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.grad_norm_clipping)

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            
            final_loss = torch.sum(final_loss) / batch_size

            total_loss += loss.item()
            count_batches += 1

        average_loss = total_loss / count_batches
        return average_loss


    def backward_td_lambda(self, all_predictions, rewards, mask):
        batch_size, seq_len, reward_dim = all_predictions.size()

        G_lambda = torch.zeros_like(all_predictions)

        immediate_reward = torch.where(
            (mask[:, seq_len-1].view(batch_size, 1) == 1.0),    # current token is valid
            rewards,            # final reward if next is invalid (since t+1 is out of range)
            torch.zeros((batch_size, reward_dim), device=rewards.device, dtype=rewards.dtype)
        )
        G_lambda[:, seq_len-1, :] = immediate_reward

        for t in reversed(range(seq_len-1)):
            valid_next = mask[:, t+1].view(batch_size, 1)
            immediate_reward = torch.where(
                (mask[:, t].view(batch_size, 1) == 1.0) & (valid_next == 0.0),
                rewards,  # final reward if next is invalid (last token)
                torch.zeros((batch_size, reward_dim), dtype=rewards.dtype, device=rewards.device)  # zero reward if next is valid
            )

            bracket = (1 - self.lambda_param) * all_predictions[:, t+1, :] + self.lambda_param * G_lambda[:, t+1, :]

            continuation = valid_next * bracket

            this_return = immediate_reward + continuation

            this_return = this_return * mask[:, t].view(batch_size, 1)

            this_return = this_return  #.round().clip(0, 4).int()

            G_lambda[:, t, :] = this_return

        return G_lambda


    def train(self):
        self.epoch_it = trange(self.num_train_epochs, desc='Epoch', leave=True)

        best_test_loss = float('inf')
        n_no_improve = 0
        for epoch in self.epoch_it:
            train_epoch_loss = self.run_epoch_tdlambda(mode='train')
            test_loss = self.run_epoch_tdlambda(mode='test')

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                d = dict(epoch=epoch+1, loss=f'{test_loss:.4f}')
                self.logger.info(f'Saving model with best test loss: {s.i(d)}')
                if isinstance(self.model, torch.nn.DataParallel):
                    torch.save(self.model.module.state_dict(), os_join(self.model_output_dir, 'best_model.pth'))
                else:
                    torch.save(self.model.state_dict(), os_join(self.model_output_dir, 'best_model.pth'))

                n_no_improve = 0
            else:
                n_no_improve += 1
                if n_no_improve >= self.patience:
                    self.logger.info(f'Early stopping at epoch {s.i(epoch)}')
                    break

class TensorDataset(Dataset):
    def __init__(self, data, labels, mask):
        """
        Args:
            data (Tensor): A tensor containing the data with shape (num_data, hidden_dim).
        """
        self.data = data
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.mask[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Llama-3.2-1B')
    parser.add_argument('--dataset_name', type=str, default='HelpSteer2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dims', type=argparse_str2int_list, help='JSON list of hidden dimensions', default='[2048]')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine'])
    parser.add_argument('--grad_norm_clipping', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--temperature', type=str, default="1.2")
    parser.add_argument('--lambda_param', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument(
        '--feature_path', type=str, default=os_join('localscratch', 'sp', 'features'),
        help='Path to the LLM-generated features, will also be used to store the labels'
    )
    parser.add_argument(
        '--model_output_dir', type=str, default=os_join('localscratch', 'sp', 'trained_model'),
        help='Directory to save the trained model'
    )
    args = parser.parse_args()

    dset_map = {'HelpSteer2': 'hs2', 'CodeUltraFeedback': 'code-uf'}
    d_md = dict(md=args.model_name, dset=dset_map[args.dataset_name])
    d_tr = {'lr':f'{args.lr:.1e}', 'bsz':args.batch_size, "lambda":args.lambda_param}
    model_dir_nm = f'vf-{s.pa(d_md)}_{s.pa(d_tr)}'
    model_output_path = os_join(args.model_output_dir, model_dir_nm)

    os.makedirs(model_output_path, exist_ok=True)

    log_path = os_join(model_output_path, 'train-vf.log')
    logger = get_logger('Train-VF', kind='both+ansi', file_path=log_path)
    logger.info(f'Training Value Function w/ {s.i(vars(args), indent=1)}')

    set_seed(args.seed)

    if args.model_name in ["llama-3.2-3b-it", "phi-4-mini-it"]:
        value_model = ValueFunction(input_dim=3072, hidden_dims=args.hidden_dims, num_attributes=5, logger=logger)
    elif args.model_name == "llama-3.2-1b-it":
        value_model = ValueFunction(input_dim=2048, hidden_dims=args.hidden_dims, num_attributes=5, logger=logger)

    num_available_gpus = torch.cuda.device_count()
    if num_available_gpus > 1:
        value_model = torch.nn.DataParallel(value_model, device_ids=list(range(num_available_gpus)))

    hidden_states_train, mask_train, hidden_states_test, mask_test = (
        torch.load(os_join(args.feature_path, f'{fnm}_{split}.pth'))
        for split in ('train', 'test') for fnm in ('token_wise_activations', 'mask')
    )
    print(hidden_states_train.size())

    labels_train, labels_test = (
        torch.load(os_join(args.feature_path, f'response_{split}_scores.pth')) 
        for split in ('train', 'test')
    )
    
    train_dataset = TensorDataset(hidden_states_train, labels_train, mask_train)
    test_dataset = TensorDataset(hidden_states_test, labels_test, mask_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    value_model = value_model.to(dtype=torch.bfloat16)

    # save the model w/ the lowest test loss
    trainer = Trainer(
        model=value_model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=args.optimizer,
        learning_rate=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler, num_train_epochs=args.epochs,
        grad_norm_clipping=args.grad_norm_clipping, patience=args.patience, lambda_param=args.lambda_param,
        logger=logger, model_output_dir=model_output_path
    )
    trainer.train()


if __name__ == '__main__':
    from rich.traceback import install
    install(show_locals=False)

    main()
