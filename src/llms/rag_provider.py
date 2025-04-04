class RAGProvider:
    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def answer_rag_question(self, query: str, limit: int = 5):
        
        answer, full_prompt, chat_history = None, None, None
        
        retrieved_documents = self.vectordb_client.query(
            query_text=query,
            k=limit
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc[0].page_content,#self.generation_client.process_text() if you want to cut texts

            })
            for idx, doc in enumerate(retrieved_documents)
        ])
        footer_prompt = self.template_parser.get("rag", "footer_prompt",
                                                 {"query": query})

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history