# src/hybrid_retrieval.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetrieval:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z-]{2,}\b'
        )
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectors = None
        self.semantic_embeddings = None
        self.corpus = []

    def index_documents(self, documents):
        """Index documents using both methods"""
        self.corpus = documents
        
        # TF-IDF indexing
        self.tfidf_vectors = self.tfidf_vectorizer.fit_transform(documents)
        
        # Semantic indexing
        self.semantic_embeddings = self.semantic_model.encode(documents)

    def retrieve(self, query, top_k=5):
        """Hybrid retrieval combining both methods"""
        # TF-IDF retrieval
        tfidf_results = self._tfidf_retrieval(query, top_k)
        
        # Semantic retrieval
        semantic_results = self._semantic_retrieval(query, top_k)
        
        # Combine and deduplicate
        combined = list(dict.fromkeys(tfidf_results + semantic_results))
        return combined[:top_k]

    def get_relevant_documents(self, query, k=5):
        """LangChain compatible interface"""
        return self.retrieve(query, top_k=k)

    def _tfidf_retrieval(self, query, top_k):
        if not self.corpus:
            return []
            
        query_vec = self.tfidf_vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_vectors).flatten()
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        return [self.corpus[i] for i in top_indices]
        
    def _semantic_retrieval(self, query, top_k):
        if not self.corpus:
            return []
            
        query_embed = self.semantic_model.encode([query])
        sim_scores = cosine_similarity(query_embed, self.semantic_embeddings).flatten()
        top_indices = np.argsort(sim_scores)[-top_k:][::-1]
        return [self.corpus[i] for i in top_indices]

    def extract_keywords(self, text, top_n=10):
        """TF-IDF based keyword extraction"""
        try:
            if not text or len(text) < 50:
                return []
                
            tfidf_scores = self.tfidf_vectorizer.transform([text])
            features = self.tfidf_vectorizer.get_feature_names_out()
            sorted_indices = np.argsort(tfidf_scores.toarray()[0])[::-1]
            return [str(features[i]) for i in sorted_indices[:top_n]]  # Ensure string conversion
        except Exception as e:
            return []