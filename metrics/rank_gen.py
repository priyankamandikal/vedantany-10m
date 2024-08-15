import os
import os.path as osp
import argparse
import json
import torch

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *

repo_dir = 'path/to/rankgen/repo'  # Change to path of the RankGen repo
sys.path.append(repo_dir)
from rankgen import RankGenEncoder, RankGenGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="prompt type. choose from [rag-kw, rag]")
    return parser.parse_args()

def get_qa_pair(file):
    with open(file, "r") as f:
        d = json.load(f)
        q = d['q']
        a = d['a']
        a = a.replace('\n\n', ' ')
    return q, a

if __name__ == "__main__":

    args = parse_args()
    rankgen_encoder = RankGenEncoder("kalpeshk2011/rankgen-t5-xl-all")

    evaldir = osp.join(project_dir, "eval/2-rag-vs-kwrag")
    datadir = osp.join(evaldir, "answers/mixtral-nomic/json")
    prompt_type = args.model

    savedir = osp.join(evaldir, "metrics", "rankgen")
    os.makedirs(savedir, exist_ok=True)
    savepath = osp.join(savedir, f"{prompt_type}.txt")
    if osp.exists(savepath):
        os.remove(savepath)

    all_scores = []
    category_scores = {}
    for fname in sorted(os.listdir(osp.join(datadir, prompt_type))):
        cat = fname.split('_')[0]
        q, a = get_qa_pair(osp.join(datadir, prompt_type, fname))

        q_emb = rankgen_encoder.encode(q, vectors_type="prefix")["embeddings"][0] # 2048
        a_emb = rankgen_encoder.encode(a, vectors_type="suffix")["embeddings"][0] # 2048
        # normalize
        q_emb = q_emb / torch.norm(q_emb)
        a_emb = a_emb / torch.norm(a_emb)
        score = torch.dot(q_emb, a_emb).item()
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
