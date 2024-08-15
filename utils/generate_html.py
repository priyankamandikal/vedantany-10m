import os
import pandas as pd

from utils.bot_utils import save_html


K = 5
KEYWORD_FILE = "eval/2-rag-vs-kwrag/keywords/aggregate.xlsx"
OUTDIR = f"eval/2-rag-vs-kwrag/answers/mixtral-nomic-large-v2"
JSON_DIR = f"{OUTDIR}/json"
HTML_DIR = f"{OUTDIR}/html"
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# read queries
df = pd.read_excel(KEYWORD_FILE)
categories = df['Category'].tolist()
queries = df['Query'].tolist()
keywords = df['Keywords'].tolist()
print(f"Number of queries: {len(queries)}")

# generate html
prompt_types = ["rag-kw", "rag"]
for idx, (c, q, k) in enumerate(zip(categories, queries, keywords)):
    if idx != 4:
        continue
    print("="*80)
    print(f"Processing {idx}/{len(queries)}...")
    print(f"Category: {c}")
    print(f"Query: {q}")
    print(f"Keywords: {k}")
    try:
        fname = f"{c}_{q.replace(' ', '_')}"[:47]
        save_html(q, c, None, prompt_types, fname, JSON_DIR, HTML_DIR)
    except Exception as e:
        fname = f"{c}_{q.replace(' ', '_')}"[:48]
        save_html(q, c, None, prompt_types, fname, JSON_DIR, HTML_DIR)