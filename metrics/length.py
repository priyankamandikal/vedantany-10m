import os
import os.path as osp
import argparse
import json
import spacy

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="prompt type. choose from [rag-kw, rag]")
    parser.add_argument("--metric", type=str, default='word', help="Choose from [word, sentence]")
    return parser.parse_args()

def get_qar(file, model):
    with open(file, "r") as f:
        d = json.load(f)
        q = d['q']
        a = d['a']
        a = a.replace('\n\n', ' ')
        if model != 'standard':
            r = d['r'][0]["content"]
            r = r.replace('<br><br>', ' ')
        else:
            r = None
    return q, a, r

if __name__ == "__main__":

    args = parse_args()
    nlp = spacy.load("en_core_web_sm")
    if args.metric == 'sentence':
        nlp.add_pipe('sentencizer')

    evaldir = osp.join(project_dir, "eval/2-rag-vs-kwrag")
    datadir = osp.join(evaldir, "answers/mixtral-nomic/json")
    prompt_type = args.model

    savedir = osp.join(evaldir, "metrics", f"length-{args.metric}")
    os.makedirs(savedir, exist_ok=True)
    savepath = osp.join(savedir, f"{prompt_type}.txt")
    if osp.exists(savepath):
        os.remove(savepath)

    all_scores = []
    category_scores = {}
    for fname in sorted(os.listdir(osp.join(datadir, prompt_type))):
        cat = fname.split('_')[0]
        q, a, r = get_qar(osp.join(datadir, prompt_type, fname), prompt_type)

        scores = []
        for text in [q, a, r]:
            if text is None:
                scores.append(0)
                continue
            doc = nlp(text)
            if args.metric == 'word':
                items = [token.text for token in doc if not token.is_punct]
            elif args.metric == 'sentence':
                items = [sent.text for sent in doc.sents]
            scores.append(len(items))
    
        with open(savepath, 'a') as fout:
            print(f"{fname}", end='\t', file=fout)
            for n_gram in range(3):
                print(f"{scores[n_gram]}", end='\t', file=fout)
            print(file=fout)

        all_scores.append(scores)
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(scores)

    avg_all_scores = [sum([scores[i] for scores in all_scores]) / len(all_scores) for i in range(3)]
    avg_category_scores = {}
    for cat, cat_scores in category_scores.items():
        avg_category_scores[cat] = [sum([scores[i] for scores in cat_scores]) / len(cat_scores) for i in range(3)]

    with open(savepath, 'a') as fout:
        # Category
        for cat in avg_category_scores:
            print(cat, end='\t', file=fout)
            for n_gram in range(3):
                print(f"{avg_category_scores[cat][n_gram]:.2f}", end='\t', file=fout)
            print(file=fout)
        # Avg
        print("Average", end='\t', file=fout)
        for n_gram in range(3):
            print(f"{avg_all_scores[n_gram]:.2f}", end='\t', file=fout)
        print(file=fout)

    print(f"Saved to {savepath}")

