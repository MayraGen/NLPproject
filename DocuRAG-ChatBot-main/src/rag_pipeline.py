# src/rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("config/api_keys.env")

class RAGPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="deepseek/deepseek-r1:free", #gpt-3.5-turbo #deepseek/deepseek-r1:free
            temperature=0.7,
            max_tokens=1000,
            # model_kwargs={
            #     "headers": {    
            #         "HTTP-Referer": os.getenv("HTTP_REFERER", "http://localhost:7861"),
            #         "X-Title": "DocuRAG AI"
            #     }
            # }
        )
        
        self.prompt = ChatPromptTemplate.from_template(
            """Answer using this context:
            {context}
            
            Question: {question}
            """
        )

    def generate_response(self, question):
        context = self.retriever.get_relevant_documents(question)
        chain = self.prompt | self.llm
        return chain.invoke({
            "context": context,
            "question": question
        }).content
        
