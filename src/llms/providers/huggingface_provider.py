from ..llm_interface import LLMInterface
from ..llms_enums import HuggingFaceEnums
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import List, Dict
from logging import getLogger



class HuggingFaceProvider(LLMInterface):
    """Implementation of LLMInterface using Hugging Face models."""
    
    def __init__(self, vector_store=None,
                 input_max_characters: int=1000,
                 max_output_tokens: int=1000,
                 temperature: float=0.1,
                 top_k: int=10,
                 top_p: float=0.95):

        self.vector_store = vector_store

        self.default_input_max_characters = input_max_characters
        self.default_generation_max_output_tokens = max_output_tokens
        self.default_generation_temperature = temperature
        self.default_generation_top_k = top_k
        self.default_generation_top_p = top_p

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.tokenizer = None
        self.model = None
        self.embedding_tokenizer = None
        self.embedding_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_generation_model(self, model_id: str) -> None:
        """Set the model for text generation."""
        self.generation_model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        """Set the model for text embedding."""
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.embedding_model = AutoModel.from_pretrained(model_id).to(self.device)

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()
    
    def generate_text(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the Hugging Face model based on messages."""
        if not self.model:
            raise ValueError("Generation model not set. Call set_generation_model first.")

        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(inputs, return_tensors='pt', return_token_type_ids=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.default_generation_max_output_tokens,
                do_sample=True,
                top_k=self.default_generation_top_k,
                top_p=self.default_generation_top_p,
                temperature=self.default_generation_temperature,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def embed_text(self, text: str, document_type: str = None):
        """Generate embeddings for the given text."""
        if not self.embedding_model:
            raise ValueError("Embedding model not set. Call set_embedding_model first.")

        inputs = self.embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embeddings = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy()
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from the Hugging Face model based on messages."""
        if not self.model:
            raise ValueError("Generation model not set. Call set_generation_model first.")

        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(inputs, return_tensors='pt', return_token_type_ids=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                do_sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def retrieve_and_generate(self, query: str) -> str:
        """Retrieve relevant chunks and generate a response."""
        docs = self.vector_store.query(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.apply_prompt_template(context, query)
        messages = [
            self.construct_prompt(HuggingFaceEnums.SYSTEM.value, "You are a helpful AI assistant."),
            self.construct_prompt(HuggingFaceEnums.USER.value, prompt),
        ]
        return self.generate_response(messages)

    def retrieve_and_generate_history(self, query: str, chat_history: str) -> str:
        """Retrieve relevant chunks and generate a response while considering chat history."""
        docs = self.vector_store.query(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.apply_prompt_template(context, query, chat_history)
        messages = [
            self.construct_prompt(HuggingFaceEnums.SYSTEM.value, "You are a helpful AI assistant."),
            self.construct_prompt(HuggingFaceEnums.USER.value, prompt),
        ]
        return self.generate_response(messages)

    def construct_prompt(self, role: str, prompt: str) -> dict:
        return {
            "role": role,
            "content": self.process_text(prompt)
        }