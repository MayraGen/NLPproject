from src.rag_pipeline import RAGPipeline
from src.document_processing import extract_text, clean_text, chunk_text, detect_language, generate_chunk_summaries
from src.hybrid_retrieval import HybridRetrieval

class DocumentAnalysisService:
    def __init__(self, config):
        self.processor = document_processing(
            lang_detect_config=config.LANG_DETECT,
            quantized_models=config.MODEL_PATHS
        )
        self.rag = RAGPipeline(
            retrieval_mode="hybrid",
            safety_filters=config.SAFETY_FILTERS
        )

    def analyze_document(self, file_path):
        processed = self.processor.pipeline(file_path)
        analysis = self.rag.generate_analysis(processed)
        return self._format_output(analysis)