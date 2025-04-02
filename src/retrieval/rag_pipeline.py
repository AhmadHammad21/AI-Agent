from utils.cleaning import remove_think_tags
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class RAGPipeline:
    """Retrieves relevant chunks and generates answers using an LLM."""

    def __init__(self, vector_store, model_name):
        self.vector_store = vector_store

        # Load the tokenizer and model for CausalLM (e.g., ALLaM-7B-Instruct-preview)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Move the model to GPU if available
        self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_response(self, messages):
        """Generate response from the model based on the given chat history."""

        # Apply the chat template and tokenize the inputs
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(inputs, return_tensors='pt', return_token_type_ids=False)
        
        # Move inputs to GPU
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        # Generate output from the model using the specified parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,  # Limit the length of the generated response
                do_sample=True,       # Enable sampling (for more diverse outputs)
                top_k=50,             # Top-k sampling (controls diversity)
                top_p=0.95,           # Nucleus sampling (controls diversity)
                temperature=0.6,      # Controls randomness
            )

        # Decode the generated tokens to text
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def retrieve_and_generate(self, query):
        """Retrieves relevant chunks and generates a response."""

        # Retrieve relevant documents from the vector store
        docs = self.vector_store.query(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Structure the conversation as messages
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Use the given context to answer the question."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ]

        # Generate the response using the model
        response = self.generate_response(messages)
        return response
        # return remove_think_tags(response)

    def retrieve_and_generate_history(self, query, chat_history):
        """Retrieves relevant chunks and generates a response with chat history."""

        # Retrieve relevant documents from the vector store
        docs = self.vector_store.query(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Full system prompt with additional instructions
        system_prompt = f"""Use the chat history to maintain context:
        
        Chat History:
        {chat_history}

        Analyze the question and context through these steps:
        1. Identify key entities and relationships.
        2. Resolve contradictions between sources.
        3. Integrate information from multiple contexts.
        4. Answer in English if the question is in English, and in Arabic if the question is in Arabic.

        Context:
        {context}

        Question: {query}
        Answer:
        """

        # Structure the conversation as messages
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Generate the response using the model
        response = self.generate_response(messages)

        return remove_think_tags(response)


# class RAGPipeline:
#     """Retrieves relevant chunks and generates answers using an LLM."""

#     def __init__(self, vector_store, model_name):
#         self.vector_store = vector_store
#         self.llm = OllamaLLM(model=model_name)

#     def retrieve_and_generate(self, query):
#         """Retrieves relevant chunks and generates a response."""
#         docs = self.vector_store.query(query)
#         context = "\n\n".join([doc.page_content for doc in docs])

#         prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"
        
#         response = self.llm.invoke(prompt)
        
#         return remove_think_tags(response)

#     def retrieve_and_generate_history(self, query, prompt_chain):
#         """Retrieves relevant chunks and generates a response."""
#         docs = self.vector_store.query(query)
#         context = "\n\n".join([doc.page_content for doc in docs])

#         # prompt = f"Context:\n{context}\n\nUser Query: {prompt_chain}\n\nAnswer:"
        
#         # response = self.llm.invoke(prompt)
        
#         # Combine the prompt chain with the retrieved context

#         system_prompt = f"""Use the chat history to maintain context:
#             Chat History:
#             {prompt_chain}

#             Analyze the question and context through these steps:
#             1. Identify key entities and relationships
#             2. Resolve Contradictions between sources
#             3. Integrate information from multiple contexts
#             4. Answer in English if the question is is english, and in Arabic if it's Arabic

#             Context:
#             {context}

#             Question: {query}
#             Answer:"""
        
#         # full_prompt = f"{prompt_chain}\n\nContext:\n{context}"
#         processing_pipeline = self.llm | StrOutputParser()
#         response = processing_pipeline.invoke(system_prompt)
#         return remove_think_tags(response)
