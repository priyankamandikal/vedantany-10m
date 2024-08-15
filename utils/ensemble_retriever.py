"""
Modified langchain's ensemble retriever to handle nomic prefix
"""
from typing import Any, Dict, List, Optional, cast

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config

from langchain.retrievers import EnsembleRetriever

class CustomEnsembleRetriever(EnsembleRetriever):
    retrievers: List[RetrieverLike]
    weights: List[float]
    c: int = 60
    _new_arg_supported = True
    _expects_other_args = True
    includes_nomic = False
    use_keywords  =False

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        ensemble_args: Dict[str, Any] = {"k": 100, "fusion_type": "rank_fusion", "ensemble_k": 100},
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """
        # print("Getting relevant documents")
        # print(query)
        # print(keywords)

        # Get fused result of the retrievers.
        fused_documents = self.apply_fusion(query, run_manager, ensemble_args=ensemble_args)

        return fused_documents

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, keywords: str = None, **kwargs: Any
    ) -> List[Document]:
        from langchain_core.callbacks import CallbackManager

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            keywords=keywords,
            **kwargs,
        )
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config, keywords=keywords)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    def apply_fusion(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
        ensemble_args: Dict[str, Any] = {"k": None, "fusion_type": "rank_fusion", "ensemble_k": 100},
    ) -> List[Document]:
        """
        Retrieve the results of the retrievers and use rank_fusion_func to get
        the final result.

        Args:
            query: The query to search for.
            keywords: The keywords to search for. Used in TF-IDF retriever if self.use_keywords is True.

        Returns:
            A list of reranked documents.
        """
        # initialize ensemble_args
        keywords = ensemble_args.get("k", None)
        fusion_type = ensemble_args.get("fusion_type", "rank_fusion")
        ensemble_k = ensemble_args.get("ensemble_k", 100)

        # Set the standard query for TF-IDF retriever
        if self.use_keywords:
            tfidf_query = keywords
        else:
            if self.includes_nomic and query.startswith("search_query: "): # remove nomic prefix
                tfidf_query = query[len("search_query: "):]
            else:
                tfidf_query = query
    
        # Get the results of all retrievers.
        retriever_docs = []
        for i, retriever in enumerate(self.retrievers):
            if type(retriever) == VectorStoreRetriever:
                # print("Using vectorstore retriever")
                search_query = query
                docs_and_similarities = (
                    retriever.vectorstore.similarity_search_with_relevance_scores(search_query, k=ensemble_k)
                )
                # add similarity scores to the metadata of the documents
                for doc, similarity in docs_and_similarities:
                    doc.metadata["similarity_score"] = similarity
                doc_list = [doc for doc, _ in docs_and_similarities]
            else:
                # print("Using tfidf retriever")
                search_query = tfidf_query
                doc_list = retriever.invoke(
                    search_query,
                    patch_config(
                        config, callbacks=run_manager.get_child(tag=f"retriever_{i+1}")
                    ),
                )
            retriever_docs.append(doc_list)

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc
                for doc in retriever_docs[i]
            ]

        # Handle nomic prefix
        for i in range(len(retriever_docs)):
            for doc in retriever_docs[i]:
                if self.includes_nomic:
                    if doc.page_content.startswith("search_document: "):
                        doc.page_content = doc.page_content.replace("search_document: ", "")

        # apply fusion
        if fusion_type == "rank_fusion":
            # print("Using rank fusion")
            fused_documents = self.weighted_reciprocal_rank(retriever_docs)
        elif fusion_type == "similarity_fusion":
            # print("Using similarity fusion")
            fused_documents = self.weighted_similarity(retriever_docs)

        return fused_documents
    
    
    def weighted_reciprocal_rank(
        self, doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Create a union of all unique documents in the input doc_lists
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc.page_content)

        # Initialize the RRF score dictionary for each document
        rrf_score_dic = {doc: 0.0 for doc in all_documents}

        # Calculate RRF scores for each document
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + self.c))
                rrf_score_dic[doc.page_content] += rrf_score

        # Sort documents by their RRF scores in descending order
        sorted_documents = sorted(
            rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
        )

        # Map the sorted page_content back to the original document objects
        page_content_to_doc_map = {
            doc.page_content: doc for doc_list in doc_lists for doc in doc_list
        }
        sorted_docs = [
            page_content_to_doc_map[page_content] for page_content in sorted_documents
        ]

        return sorted_docs
    
    def weighted_similarity(self, retriever_docs: List[List[Document]]) -> List[Document]:
        """
        Weighted similarity fusion of documents.

        Args:
            retriever_docs: A list of lists of documents from the retrievers.

        Returns:
            A list of reranked documents.
        """

        # Get a combined list of unique documents by metadata['link']. Similarity score will be a dictionary with key as retriever index.
        fused_documents = []
        for i in range(len(retriever_docs)):
            for doc in retriever_docs[i]:
                if doc.metadata.get("link") not in [doc.metadata.get("link") for doc in fused_documents]:
                    doc.metadata["sim_score_dict"] = {}
                    doc.metadata["sim_score_dict"][i] = doc.metadata.get("similarity_score", 0)
                    doc.metadata.pop("similarity_score", None)
                    fused_documents.append(doc)
                else:
                    for combined_doc in fused_documents:
                        if combined_doc.metadata.get("link") == doc.metadata.get("link"):
                            combined_doc.metadata["sim_score_dict"][i] = doc.metadata.get("similarity_score", 0)
        
        # Assign 0 similarity score to missing retriever index.
        for doc in fused_documents:
            for i in range(len(retriever_docs)):
                doc.metadata["sim_score_dict"][i] = doc.metadata["sim_score_dict"].get(i, 0)

        # Calculate the weighted similarity score for each document.
        for doc in fused_documents:
            doc.metadata["sim_score_dict"]['fused'] = 0
            for i in range(len(retriever_docs)):
                doc.metadata["sim_score_dict"]['fused'] += self.weights[i] * doc.metadata["sim_score_dict"].get(i, 0)

        # Rank the documents by the combined similarity scores.
        fused_documents.sort(key=lambda doc: doc.metadata["sim_score_dict"]['fused'], reverse=True)
        return fused_documents
