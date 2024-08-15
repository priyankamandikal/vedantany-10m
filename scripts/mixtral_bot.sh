# Description: Run Mixtral 8x7B Instruct on a set of queries in parallel
# Usage: bash scripts/mixtral_bot.sh
# Results: Generations are saved in the eval/2-rag-vs-kwrag/answers

n_queries_total=25 # Total number of queries to run
n_queries_batch=5  # Number of queries to run in parallel. To be set as per available resources

n_batches=$((n_queries_total / n_queries_batch  + n_queries_total % n_queries_batch))

for i in $(seq 0 $((n_batches - 1))); do
    start=$((i * n_queries_batch))
    if [ $((i + 1)) -eq $n_batches ]; then
        n_queries_batch=$((n_queries_total - start))
    fi
    end=$((start + n_queries_batch - 1))
    query_indices=( $(seq $start $end) )
    echo "Running queries $start to $end"
    python mixtral_bot.py --llm mixtral --embedding_model nomic --vectorstore chroma --k 5 --ensemble_k 100 --fusion_type similarity_fusion --query_indices "${query_indices[@]}"
done
