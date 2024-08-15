"""
Script to extract keywords from queries using different keyword extraction models.
Run as:
    Extract keywords:
        python keyword_extraction.py --model keybert --thr 0.3
        python keyword_extraction.py --model openkp
        python keyword_extraction.py --model wikineural
        python keyword_extraction.py --model spanmarker
        python keyword_extraction.py --model spanmarker --uncased
    Aggregate keywords:
        python keyword_extraction.py --mode aggr --models keybert-0.3 openkp wikineural spanmarker-cased spanmarker-uncased
"""

import os
import os.path as osp
import pandas as pd
from openai import OpenAI
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)
from transformers.pipelines import AggregationStrategy
import numpy as np


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])


class KeywordExtractor:

    def __init__(self, model, outfile):
        self.queries_file = "eval/2-rag-vs-kwrag/queries.xlsx"
        outdir = "eval/2-rag-vs-kwrag/keywords"
        os.makedirs(outdir, exist_ok=True)
        self.outfile = osp.join(outdir, f"{outfile}.xlsx")
        if model == "openkp":
            self.extractor = self.openkp_extractor
        elif model == "wikineural":
            self.extractor = self.wikineural_extractor
        elif model == "spanmarker":
            self.extractor = self.spanmarker_extractor
        elif model == "keybert":
            self.extractor = self.keybert_extractor
        else:
            raise ValueError("Invalid model for keyword extractor")
    
    def extract_keywords(self, **kwargs):
        self.load_queries()
        self.extractor(**kwargs)
        self.save_keywords()

    def load_queries(self):
        df = pd.read_excel(self.queries_file)
        self.queries = df['Query'].tolist()
        print(f"Number of queries: {len(self.queries)}")

    def keybert_extractor(self, model="all-mpnet-base-v2", threshold=0.35):
        from keybert import KeyBERT
        from nltk.corpus import stopwords as stopwords_nltk
        from spacy.lang.en.stop_words import STOP_WORDS as stopwords_spacy
        # write stop words to file
        with open("eval/stopwords-nltk.txt", "w") as f:
            f.write("\n".join(stopwords_nltk.words("english")))
        with open("eval/stopwords-spacy.txt", "w") as f:
            f.write("\n".join(stopwords_spacy))
        # union of NLTK and spaCy stop words
        stop_words_nltk = stopwords_nltk.words("english")
        stop_words = list(set(stopwords_spacy).union(set(stop_words_nltk)))
        stop_words.extend(["swami", "swamiji", "swamis", "swamijis"])
        kw_model = KeyBERT(model)
        self.keywords = []
        for query in self.queries:
            kw_list = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 1), stop_words=stop_words, use_mmr=True, diversity=0.5, top_n=3)
            print("Query:", query)
            print("Keywords:", kw_list)
            print("\n")
            self.keywords.append([kw[0] for kw in kw_list if kw[1] >= threshold])

    def openkp_extractor(self, model="ml6team/keyphrase-extraction-distilbert-openkp"):
        extractor = KeyphraseExtractionPipeline(model=model)
        self.keywords = []
        for q in self.queries:
            self.keywords.append(extractor(q))

    def wikineural_extractor(self, model="Babelscape/wikineural-multilingual-ner"):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForTokenClassification.from_pretrained(model)
        extractor = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
        self.keywords = []
        for q in self.queries:
            kw_dict = extractor(q)
            if type(kw_dict) == dict:
                kw_dict = [kw_dict]
            kw_list = [ele['word'] for ele in kw_dict]
            self.keywords.append(kw_list)

    def spanmarker_extractor(self, model="tomaarsen/span-marker-mbert-base-multinerd"):
        from span_marker import SpanMarkerModel
        model = SpanMarkerModel.from_pretrained(model).cuda()
        self.keywords = []
        for q in self.queries:
            kw_dict = model.predict(q)
            if type(kw_dict) == dict:
                kw_dict = [kw_dict]
            kw_list = [ele['span'] for ele in kw_dict]
            self.keywords.append(kw_list)

    def save_keywords(self):
        df_out = pd.DataFrame(columns=['Query', 'Keywords'])
        for q, k in zip(self.queries, self.keywords):
            k = [x for x in k if 'swami' not in x.lower()] # remove swami from keywords
            k = [x for x in k if x.lower() in q.lower()] # retain keyword if in query string
            k = list(dict.fromkeys(k)) # remove duplicate keywords
            w = q.split(" ") # split query into words
            w = [x.strip("'\",.!?").lower() for x in w] # remove punctuation
            w = [x.split("'")[0] for x in w] # remove apostrophe
            for x in k.copy(): # remove single keyword if not in the query
                if len(x.split(" ")) == 1:
                    if x.lower() not in w:
                        k.remove(x)
            k = ', '.join(k)
            df_out = pd.concat([df_out, pd.DataFrame([[q, k]], columns=['Query', 'Keywords'])])
            print(f"Query: {q}")
            print(f"Keywords: {k}")
            print("\n")
        df_out.to_excel(self.outfile, index=False)


class KeywordAggregator:

    def __init__(self, models):
        self.models = models
        self.queries_file = "eval/2-rag-vs-kwrag/queries.xlsx"
        self.outdir = "eval/2-rag-vs-kwrag/keywords"
        self.outfile = osp.join(self.outdir, "aggregate.xlsx")
        self.load_keywords()

    def load_keywords(self):
        df = pd.read_excel(self.queries_file)
        self.queries = df['Query'].tolist()
        self.categories = df['Category'].tolist()
        print(f"Number of queries: {len(self.queries)}")

        self.keywords = {}
        for model in self.models:
            df = pd.read_excel(osp.join(self.outdir, f"{model}.xlsx"))
            k = df['Keywords'].tolist()
            k = ["" if type(x) == float else x for x in k] # replace nan with empty string
            self.keywords[model] = [x.split(", ") for x in k]
            print("Model:", model)
            print(self.keywords[model])
            print("\n")
        print(f"Number of models: {len(self.models)}")

    def aggregate_keywords(self):
        df_out = pd.DataFrame(columns=['Category', 'Query', 'Keywords'])
        for idx, q in enumerate(self.queries):
            c = self.categories[idx]
            k = []
            for model in self.models:
                k.extend([x.lower() for x in self.keywords[model][idx]])
            k = list(dict.fromkeys(k)) # remove duplicate keywords
            k = [x for x in k if x != ""] # remove  empty strings
            # remove keywords that are substrings of other keywords
            for x in k.copy():
                for y in k.copy():
                    if x != y and x in y:
                        k.remove(x)
                        break
            # check if multiple keywords when combined are a substring of the query
            for x in k.copy():
                for y in k.copy():
                    if x != y:
                        if x + " " + y in q.lower():
                            k.remove(x)
                            k.remove(y)
                            k.append(x + " " + y)
            k = ', '.join(k)
            df_out = pd.concat([df_out, pd.DataFrame([[c, q, k]], columns=['Category', 'Query', 'Keywords'])])
            print(f"Query: {q}")
            print(f"Keywords: {k}")
            print("\n")
        df_out.to_excel(self.outfile, index=False)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="extr", help="Mode of operation: extr, aggr")
    # Extract params
    parser.add_argument("--model", type=str, default="spanmarker", help="Model used for keyword extractor. Options: keybert, openkp, wikineural, spanmarker")
    parser.add_argument("--uncased", action="store_true", help="Use uncased model for spanmarker")
    parser.add_argument("--thr", type=float, default=0.3, help="Threshold for keybert extractor")
    # Aggregate params
    parser.add_argument("--models", nargs="+", default=["keybert-0.3", "openkp", "wikineural", "spanmarker-cased", "spanmarker-uncased"], help="Models to aggregate")
    args = parser.parse_args()

    if args.mode == "extr":
        if args.model == "keybert":
            ke = KeywordExtractor(model=args.model, outfile=f"{args.model}-{args.thr}")
            ke.extract_keywords(threshold=args.thr)
        elif args.model == "spanmarker":
            if not args.uncased:
                ke = KeywordExtractor(model=args.model, outfile=f"{args.model}-cased")
                ke.extract_keywords(model="tomaarsen/span-marker-mbert-base-multinerd")
            else:
                ke = KeywordExtractor(model=args.model, outfile=f"{args.model}-uncased")
                ke.extract_keywords(model="lxyuan/span-marker-bert-base-multilingual-uncased-multinerd")
        else:
            ke = KeywordExtractor(model=args.model, outfile=args.model)
            ke.extract_keywords()
        print("Keywords saved in", ke.outfile)

    elif args.mode == "aggr":
        ka = KeywordAggregator(models=args.models)
        ka.aggregate_keywords()
        print("Keywords saved in", ka.outfile)

    else:
        raise ValueError("Invalid mode of operation")
