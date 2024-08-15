"""
To be run in qafacteval conda environment
"""

import os
import os.path as osp
import argparse
import json

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *

import sys
repo_dir = '<path-to-qafacteval-repo?'  # Change path
sys.path.append(repo_dir)
from qafacteval import QAFactEval

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="prompt type. choose from [rag-kw, rag]")
    return parser.parse_args()

def get_ar_pair(file):
    with open(file, "r") as f:
        d = json.load(f)
        a = d['a']
        a = a.replace('\n\n', ' ')
        r = d['r'][0]["content"]
        r = r.replace('<br><br>', ' ')
    return a, r

if __name__ == "__main__":
    
    args = parse_args()

    evaldir = osp.join(project_dir, "eval/2-rag-vs-kwrag")
    datadir = osp.join(evaldir, "answers/mixtral-nomic/json")
    prompt_type = args.model

    savedir = osp.join(evaldir, "metrics", "qafacteval")
    os.makedirs(savedir, exist_ok=True)
    savepath = osp.join(savedir, f"{prompt_type}.txt")
    if osp.exists(savepath):
        os.remove(savepath)

    # QAFactEval
    kwargs = {"cuda_device": 0, 
              "use_lerc_quip": True,
              "verbose": True, 
              "generation_batch_size": 32,
              "answering_batch_size": 32, 
              "lerc_batch_size": 8
              }
    model_folder = osp.join(repo_dir, "models")
    metric = QAFactEval(
        lerc_quip_path=f"{model_folder}/quip-512-mocha",
        generation_model_path=f"{model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{model_folder}/answering",
        lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
        **kwargs
    )

    all_scores = []
    category_scores = {}
    for fname in sorted(os.listdir(osp.join(datadir, prompt_type))):
        cat = fname.split('_')[0]
        a, r = get_ar_pair(osp.join(datadir, prompt_type, fname))
        results = metric.score_batch_qafacteval([r], [[a]], return_qa_pairs=True)
        score = results[0][0]['qa-eval']['lerc_quip']
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

