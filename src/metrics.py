import torch
import json
import nltk
from nltk.tokenize import word_tokenize
from typing import List
from os.path import join as os_join
from collections import defaultdict
from src.util import argparse_str2int_list
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def self_bleu(responses, weights=(0.25,0.25,0.25,0.25)):
    """
    Given a list of 3 response-strings, return each response's Self-BLEU
    (i.e., BLEU of that response vs. the other two as references).
    """
    # tokenize and lowercase
    tokenized = [word_tokenize(r.lower()) for r in responses]
    smoothie = SmoothingFunction().method1

    scores = []
    for i in range(len(tokenized)):
        hyp = tokenized[i]
        # the other two are the references
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        score = sentence_bleu(
            refs,
            hyp,
            weights=weights,
            smoothing_function=smoothie
        )
        scores.append(score)
    return scores

def l1_distance(score, target):
    target = torch.tensor(target).repeat(score.size(0), 1)
    return torch.norm(score-target.float(), p=1, dim=1).mean().item()

def success_rate(score, target):
    score_int = score.round().clip(0,4).int()
    success_count = (score_int==torch.tensor(target).repeat(score.size(0), 1)).all(dim=1).sum().item()
    sr = success_count*100 / score.size(0)
    
    return sr

def get_metric(
        llm,
        ds,
        lr,
        ns,
        target,
        vf,
        iteration,
):
    intervene_path = f"./data/inference_intervention/Inference-Result_{{md={llm},ds={ds}}}-ns=1-train_temp=0.0/infer_temp=0.0/{vf}/{{lr={lr},#s={ns}}}/edited_{{tgt={target}}}/iteration={iteration}"

    score = torch.load(os_join(intervene_path, "responses_scores.pth"), map_location="cpu")

    with open(os_join(intervene_path, "responses.jsonl"), "r") as f:
        responses = [json.loads(line) for line in f]
        responses = [res['response'] for res in responses]

    target = argparse_str2int_list(target)
    diversity = self_bleu(responses)
    distance = l1_distance(score, target)
    succ_rate = success_rate(score, target)

    return diversity, distance, succ_rate

if __name__ == "__main__":
    args = {
        "llm": "phi-4-mini-it",
        "ds": "code-uf",
        "lr": "5e-2",
        "ns": 1000,
        "target": "[3,3,3,3,3]",
        "vf": "vf-{md=phi-4-mini-it,dset=code-uf}_{lr=1.0e-04,bsz=32,lambda=0.9}",
        "iteration": 0
    }

    diversity, distance, succ_rate = get_metric(**args)