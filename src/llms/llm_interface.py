from abc import ABC, abstractmethod
from typing import List, Dict


class LLMInterface(ABC):
    """Abstract class defining the interface for different LLM implementations."""
    
    @abstractmethod
    def set_generation_model(self, model_id: str):
        """Set the language generation model to be used.

        Args:
            model_id (str): The identifier of the generation model (e.g., Hugging Face model name or OpenAI model).
        """
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """Set the embedding model for vector representation of text.

        Args:
            model_id (str): The identifier of the embedding model.
            embedding_size (int): The dimensionality of the embedding vectors.
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                    temperature: float = None, top_k: float = None, top_p: float = None):
        """Generate text based on the given prompt and optional chat history.

        Args:
            prompt (str): The input prompt for text generation.
            chat_history (list, optional): A list of previous messages to maintain conversation context. Defaults to an empty list.
            max_output_tokens (int, optional): The maximum number of tokens to generate. Defaults to None, using the class default.
            temperature (float, optional): Controls randomness in generation. Higher values make output more diverse. Defaults to None, using the class default.
            top_k (float, optional): Limits sampling to the top-k most probable tokens. Lower values make output more deterministic. Defaults to None.
            top_p (float, optional): Nucleus sampling, limiting choices to tokens whose probabilities sum up to top_p. Defaults to None.

        Returns:
            str: The generated text.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str, document_type: str = None):
        """Convert input text into an embedding vector representation.

        Args:
            text (str): The text to be embedded.
            document_type (str, optional): Specifies the type of document (e.g., 'query' or 'document') for different embedding strategies. Defaults to None.

        Returns:
            List[float]: The embedding vector representation of the input text.
        """
        pass

    def apply_prompt_template(self, context: str, query: str, chat_history: str = "") -> str:
        """
        Apply the prompt template based on the presence of chat history.
        
        Custom prompt is applied for handling chat history, context, and detailed steps.

        Args:
            context (str): The retrieved context for the query.
            query (str): The user's query.
            chat_history (str, optional): The chat history, if available. Default is an empty string.

        Returns:
            str: The final prompt string with custom instructions.
        """

        if chat_history:
            prompt = f"""Use the chat history to maintain context:
            
            Chat History:
            {chat_history}

            Analyze the question and context through these steps:
            1. Identify key entities and relationships.
            2. Resolve contradictions between sources.
            3. Integrate information from multiple contexts.
            4. Answer in English if the question is in English, and in Arabic Saudi Dialect if the question is in Arabic.

            Context:
            {context}

            Question: {query}
            Answer:
            """
        else:
            prompt = f"""
            Context:
            {context}

            Question: {query}
            Answer:
            """
            
        return prompt