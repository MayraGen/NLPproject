# src/generation.py
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import cohere
import os


STRUCTURED_PROMPT_TEMPLATE = """Generate a comprehensive summary with this exact structure:

### Key Themes
{bullet_points} 
- **Impact**: [1-2 sentences]

### Detailed Analysis
{examples}
- **Technical Challenges**: [3-5 challenges]
- **Solution Approaches**: [2-4 methods]

Use professional academic language. Focus on conceptual relationships and technical implementations.
"""

class SummaryGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "google/flan-t5-small"  # Better model for summarization google/flan-t5-base
        # self.tokenizer = None
        # self.model = None
        self.initialize_model()
        self.co = cohere.Client(os.getenv("DocuRAG-ChatBot-main/config/COHERE_API_KEY"))

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    def initialize_model(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            logging.info(f"Loaded model {self.model_name} on {self.device}")
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            raise

    def generate(self, text, max_input_length=1024, max_new_tokens=200):  # Increased from 1024
        try:
            if not text or len(text.strip()) < 50:
                return "Insufficient text content for meaningful summary"

    # Better prompt engineering
            inputs = self.tokenizer(
                f"Generate a comprehensive, detailed summary of the following document: {text}",
                max_length=max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            # Enhanced generation parameters
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,     # Increased from 150
                min_length=50,          # Increased from 50
                # length_penalty=2,      # Adjusted for longer summaries
                # no_repeat_ngram_size=4,
                early_stopping=True,
                num_beams=4,
                # temperature=0.8,         # Slightly higher for diversity
                do_sample=False,         # Better coherence with beam search
                repetition_penalty=2.0
            )

            return self.postprocess_summary(
                self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
    
        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            return "Summary generation error"

    def postprocess_summary(self, summary):
        """Clean up generated summary"""
        # Remove any bullet points or markdown artifacts
        summary = summary.replace("•", "").replace("##", "").strip()
        
        # Capitalize first letter if needed
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
            
        # Ensure proper sentence endings
        if summary and summary[-1] not in {'.', '!', '?'}:
            summary += '.'
            
        return summary
    
    def generate_structured_summary(self, text):
        """Generate structured summary using Cohere"""
        try:
            response = self.co.chat(
                message=self._build_cohere_prompt(text),
                model="command-r-plus",
                temperature=0.3,
                preamble="You are a technical documentation analyst",
                connectors=[{"id": "web-search"}]
            )
            return self._format_cohere_response(response.text)
            
        except Exception as e:
            return f"Cohere Error: {str(e)}"

    def _build_cohere_prompt(self, text):
        return f"""Analyze this document and structure your response with:
        
        1. Key Themes (3-5 bullet points)
        2. Detailed Analysis with:
           - Technical Challenges
           - Solution Approaches
        3. Impact Assessment

        Include quantitative estimates where possible. Focus on technical implementations.

        Document excerpt: "{text[:3000]}"
        """

    def _format_cohere_response(self, response):
        """Convert Cohere response to markdown format"""
        sections = {
            "1. Key Themes": "### Key Themes",
            "2. Detailed Analysis": "### Detailed Analysis",
            "3. Impact Assessment": "### Impact"
        }
        
        for k, v in sections.items():
            response = response.replace(k, v)
            
        return response.replace("- ", "• ")

    