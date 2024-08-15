from __future__ import annotations

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever


class CustomTFIDFRetriever(TFIDFRetriever):
    """Adding a custom method to the TFIDFRetriever to retrieve similarity scores
    """


    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc
        return_docs = [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
        # add the similarity score to the metadata of the documents
        return_scores = results[results.argsort()[-self.k :][::-1]]
        for doc, score in zip(return_docs, return_scores):
            doc.metadata["similarity_score"] = score
        # print(f"Similarity scores: {return_scores}")
        return return_docs
