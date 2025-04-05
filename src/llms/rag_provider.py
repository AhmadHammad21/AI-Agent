class RAGProvider:
    def __init__(self, vectordb_client, chat_log_manager, generation_client, 
                 embedding_client, template_parser):
        self.vectordb_client = vectordb_client
        self.chat_log_manager = chat_log_manager
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    async def answer_rag_question(self, user_id: str, session_id: str,
                                  query: str, limit: int = 5):
        
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

        mongo_chat_history = await self.chat_log_manager.get_chat_history(
            user_id=user_id, session_id=session_id
        )

        new_messages = []

        # Add the system prompt if it's a first-time user
        if mongo_chat_history is None:
            new_messages.append(
                self.generation_client.construct_prompt(
                    prompt=system_prompt,
                    role=self.generation_client.enums.SYSTEM.value,
                )
            )

        # Add the user query
        new_messages.append(
            self.generation_client.construct_prompt(
                prompt=query,
                role=self.generation_client.enums.USER.value,
            )
        )

        full_prompt = "\n\n".join([documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=mongo_chat_history if mongo_chat_history else new_messages
        )

        # Append the assistant's response
        new_messages.append(
            self.generation_client.construct_prompt(
                prompt=answer,
                role=self.generation_client.enums.ASSISTANT.value,
                full_prompt=full_prompt
            )
        )

        # Just append new messages to DB
        await self.chat_log_manager.insert_or_update_chat_log(
            user_id=user_id,
            session_id=session_id,
            messages=new_messages  # only the delta, not the full history
        )

        return answer, full_prompt, chat_history