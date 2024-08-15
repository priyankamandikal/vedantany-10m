# Description: Run keyword extraction on the queries using different models.
# Usage: bash scripts/keyword_extraction.sh
# Results: Extracted keywords are saved in the eval/2-rag-vs-kwrag/keywords

python keyword_extraction.py --model keybert --thr 0.3
python keyword_extraction.py --model openkp
python keyword_extraction.py --model wikineural
python keyword_extraction.py --model spanmarker
python keyword_extraction.py --model spanmarker --uncased
python keyword_extraction.py --mode aggr --models keybert-0.3 openkp wikineural spanmarker-cased spanmarker-uncased