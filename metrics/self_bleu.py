"""
Usage:
$ python metrics/self_bleu.py --model rag-kw
"""
import argparse
import json
import os
import os.path as osp
import random
from functools import partial
from multiprocessing.pool import Pool

import spacy
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from utils.setup import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="prompt type. choose from [rag-kw, rag]")
    return parser.parse_args()

def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)

def get_answer(file):
    with open(file, "r") as f:
        d = json.load(f)
        a = d['a']
        a = a.replace('\n\n', ' ')
    return a


def main():
    args = parse_args()
    random.seed(0)

    evaldir = osp.join(project_dir, "eval/2-rag-vs-kwrag")
    datadir = osp.join(evaldir, "answers/mixtral-nomic/json")
    prompt_type = args.model

    savedir = osp.join(evaldir, "metrics", "self-bleu")
    os.makedirs(savedir, exist_ok=True)
    savepath = osp.join(savedir, f"{prompt_type}.txt")

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
    nlp.add_pipe('sentencizer')

    all_bleu_scores = []
    category_bleu_scores = {}
    for fname in sorted(os.listdir(osp.join(datadir, prompt_type))):
        cat = fname.split('_')[0]
        all_sentences = []
        answer = get_answer(osp.join(datadir, prompt_type, fname))
        doc = nlp(answer)
        for sent in doc.sents:  # Iterate over sentences
            tokens = [token.text for token in sent]  # Tokenize each sentence
            all_sentences.append(tokens)
        n_sample = len(all_sentences)
        smoothing_function = SmoothingFunction().method1

        pool = Pool(processes=os.cpu_count())
        sent_bleu_scores = []
        for n_gram in range(1, 6):

            if n_gram == 1:
                weights = (1.0, 0, 0, 0)
            elif n_gram == 2:
                weights = (0.5, 0.5, 0, 0)
            elif n_gram == 3:
                weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
            elif n_gram == 4:
                weights = (0.25, 0.25, 0.25, 0.25)
            elif n_gram == 5:
                weights = (0.2, 0.2, 0.2, 0.2, 0.2)
            else:
                raise ValueError
            sent_bleu_scores.append(
                list(tqdm(
                    pool.imap_unordered(
                        partial(bleu_i, weights, all_sentences, smoothing_function),
                        random.sample(range(len(all_sentences)), n_sample)),
                    total=n_sample,
                    smoothing=0.0,
                    desc=f"bleu-{n_gram}")))

        sample_bleu_scores = [sum(sent_bleu_scores[n_gram]) / n_sample for n_gram in range(5)]
        for n_gram in range(5):
            print(f"bleu-{n_gram + 1} = {sample_bleu_scores[n_gram]:.4f}")

        with open(savepath, 'a') as fout:
            print(f"{fname}", end='\t', file=fout)
            for n_gram in range(5):
                print(f"{sample_bleu_scores[n_gram]:.4f}", end='\t', file=fout)
            print(file=fout)

        all_bleu_scores.append(sample_bleu_scores)
        if cat not in category_bleu_scores:
            category_bleu_scores[cat] = []
        category_bleu_scores[cat].append(sample_bleu_scores)

    avg_bleu_scores = [sum([bleu_scores[i] for bleu_scores in all_bleu_scores]) / len(all_bleu_scores) for i in range(5)]
    print("Average BLEU scores:")
    for n_gram in range(5):
        print(f"bleu-{n_gram + 1} = {avg_bleu_scores[n_gram]:.4f}")

    avg_category_bleu_scores = {}
    for cat, cat_bleu_scores in category_bleu_scores.items():
        avg_category_bleu_scores[cat] = [sum([bleu_scores[i] for bleu_scores in cat_bleu_scores]) / len(cat_bleu_scores) for i in range(5)]

    with open(savepath, 'a') as fout:
        # Category
        for cat in avg_category_bleu_scores:
            print(cat, end='\t', file=fout)
            for n_gram in range(5):
                print(f"{avg_category_bleu_scores[cat][n_gram]:.4f}", end='\t', file=fout)
            print(file=fout)
        # Avg
        print("Average", end='\t', file=fout)
        for n_gram in range(5):
            print(f"{avg_bleu_scores[n_gram]:.4f}", end='\t', file=fout)
        print(file=fout)


if __name__ == '__main__':
    main()
