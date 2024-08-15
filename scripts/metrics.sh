# Description: Run all automatic metrics for RAG and RAG-KW models
# Usage: bash scripts/metrics.sh
# Results: Metrics are saved in the eval/2-rag-vs-kwrag/metrics

# Perplexity
screen -dmS ppl_rag bash -c "python metrics/perplexity.py --model rag"
screen -dmS ppl_ragkw bash -c "python metrics/perplexity.py --model rag-kw"

# Self-BLEU
screen -dmS selfbleu_rag bash -c "python metrics/self_bleu.py --model rag"
screen -dmS selfbleu_ragkw bash -c "python metrics/self_bleu.py --model rag-kw"

# Length
## Word
screen -dmS len_word_rag bash -c "python metrics/length.py --metric word --model rag"
screen -dmS len_word_ragkw bash -c "python metrics/length.py --metric word --model rag-kw"
## Sentence
screen -dmS len_sent_rag bash -c "python metrics/length.py --metric sentence --model rag"
screen -dmS len_sent_ragkw bash -c "python metrics/length.py --metric sentence --model rag-kw"

# RankGen
screen -dmS rankgen_rag bash -c "python metrics/rank_gen.py --model rag"
screen -dmS rankgen_ragkw bash -c "python metrics/rank_gen.py --model rag-kw"

# # QAFactEval - doesn't run in screen for whatever reason - just run it in the terminal
# # To be run in qafacteval conda environment
# python metrics/qafact_eval.py --model rag
# python metrics/qafact_eval.py --model rag-kw
