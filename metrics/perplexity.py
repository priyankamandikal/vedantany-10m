import os
import os.path as osp
import argparse
import json
import torch
import evaluate

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="prompt type. choose from [rag-kw, rag]")
    return parser.parse_args()

def get_answer(file):
    with open(file, "r") as f:
        d = json.load(f)
        a = d['a']
        a = a.replace('\n\n', ' ')
    return a

if __name__ == "__main__":

    args = parse_args()
    perplexity = evaluate.load("perplexity", module_type="metric")

    evaldir = osp.join(project_dir, "eval/2-rag-vs-kwrag")
    datadir = osp.join(evaldir, "answers/mixtral-nomic/json")
    prompt_type = args.model

    savedir = osp.join(evaldir, "metrics", "perplexity")
    os.makedirs(savedir, exist_ok=True)
    savepath = osp.join(savedir, f"{prompt_type}.txt")
    if osp.exists(savepath):
        os.remove(savepath)

    all_scores = []
    category_scores = {}
    for fname in sorted(os.listdir(osp.join(datadir, prompt_type))):
        cat = fname.split('_')[0]
        a = get_answer(osp.join(datadir, prompt_type, fname))
        results = perplexity.compute(model_id='gpt2',
                                    add_start_token=False,
                                    predictions=[a])
        score = results["mean_perplexity"]
        print(f"{fname}: {score:.4f}")
    
        with open(savepath, 'a') as fout:
            print(f"{fname}", end='\t', file=fout)
            print(f"{score:.4f}", end='\t', file=fout)
            print(file=fout)

        all_scores.append(score)
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(score)

    avg_category_scores = {cat: sum(scores) / len(scores) for cat, scores in category_scores.items()}
    avg_overall_score = sum(all_scores) / len(all_scores)
    print("score per category:")
    for cat, score in avg_category_scores.items():
        print(f"{cat}: {score}")
    print("\nAverage:", avg_overall_score)

    with open(savepath, 'a') as fout:
        # Category
        for cat in avg_category_scores:
            print(cat, end='\t', file=fout)
            print(f"{avg_category_scores[cat]:.4f}", end='\t', file=fout)
            print(file=fout)
        # Avg
        print("Average", end='\t', file=fout)
        print(f"{avg_overall_score:.4f}", end='\t', file=fout)
        print(file=fout)

    print(f"Saved to {savepath}")

