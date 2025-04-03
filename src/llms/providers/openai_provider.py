from ..llm_interface import LLMInterface
from ..llms_enums import OpenAIEnums
from openai import OpenAI
from typing import List, Dict
from logging import getLogger



class OpenAIProvider(LLMInterface):
    """Implementation of LLMInterface using OpenAI API."""
    
    def __init__(self, api_key: str, vector_store=None,
                 input_max_characters: int=1000,
                 max_output_tokens: int=1000,
                 temperature: float=0.1,
                 top_p: float=0.95):

        self.api_key = api_key
        self.vector_store = vector_store

        self.default_input_max_characters = input_max_characters
        self.default_generation_max_output_tokens = max_output_tokens
        self.default_generation_temperature = temperature
        self.default_generation_top_p = top_p

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = OpenAI(
            api_key = self.api_key
        )

        self.logger = getLogger(__name__)

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[],
                      max_output_tokens: int=None, temperature: float = None,
                      top_p: float = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        top_p = top_p if top_p else self.default_generation_top_p

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        response = self.client.chat.completions.create(
            model = self.generation_model_id,
            messages = chat_history,
            max_tokens = max_output_tokens,
            temperature = temperature,
            top_p=top_p
        )

        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error("Error while generating text with OpenAI")
            return None

        return response.choices[0].message.content
    
    def embed_text(self, text: str, document_type: str = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        response = self.client.embeddings.create(
            model = self.embedding_model_id,
            input = text,
            dimensions=self.embedding_size
        )

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None

        return response.data[0].embedding
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from OpenAI's API based on messages."""
        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            max_tokens=self.default_generation_max_output_tokens,
            temperature=self.default_generation_temperature,
            top_p=self.default_generation_top_p
        )

        return response.choices[0].message.content
    
    def retrieve_and_generate(self, query: str) -> str:
        """Retrieve relevant chunks and generate a response."""
        ## NOTE: double check
        docs = self.vector_store.query(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.apply_prompt_template(context, query)
        messages = [
            self.construct_prompt(OpenAIEnums.SYSTEM.value, "You are a helpful AI assistant."),
            self.construct_prompt(OpenAIEnums.USER.value, prompt),
        ]
        return self.generate_response(messages)
    
    def retrieve_and_generate_history(self, query: str, chat_history: str) -> str:
        """Retrieve relevant chunks and generate a response while considering chat history."""
        docs = self.vector_store.query(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        ## TODO NOTE: RECHECH THIS QREA ABOUT INPUT CHARACTERS AND PROMPT
        prompt = self.apply_prompt_template(context, query, chat_history)
        messages = [
            self.construct_prompt(OpenAIEnums.SYSTEM.value, "You are a helpful AI assistant."),
            self.construct_prompt(OpenAIEnums.USER.value, prompt),
        ]
        return self.generate_response(messages)


    def construct_prompt(self, role: str, prompt: str) -> dict:
        return {
            "role": role,
            "content": self.process_text(prompt)
        }