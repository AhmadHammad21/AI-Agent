import streamlit as st
from vector_dbs.providers.vector_store import VectorStore
from llms.rag_provider import RAGProvider
from llms.llm_provider_factory import LLMProviderFactory
from llms.templates.template_parser import TemplateParser
from utils.logger import get_logger
from config.settings import settings
from config.config import config
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from dotenv import load_dotenv
import time
load_dotenv()

# Set up logger
logger = get_logger(__name__)

# Custom CSS styling
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar configuration
with st.sidebar:
    st.markdown("## Desaisiv AI Agent")
    st.markdown("### Capabilities")
    st.markdown("""
    - 🤖 AI Assistant
    - ❓ Question Answering
    """)
    
@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():

    template_parser = TemplateParser(language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG)
    llm_provider_factory = LLMProviderFactory(config, settings)

    generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                                embedding_size=settings.EMBEDDING_MODEL_SIZE)

    vector_store = VectorStore()
    vector_store.load_vector_store(settings.VECTOR_STORE_PATH, embedding_client.embed_text)

    rag_object = RAGProvider(vectordb_client=vector_store, generation_client=generation_client,
                             embedding_client=embedding_client, template_parser=template_parser)
    
    return rag_object


# Load the pipeline
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = initialize_rag_pipeline()


# # System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI customer support assistant. Provide concise, correct answers"
    "Always respond in English or in Arabic Saudi Dialect."
)

# Session state management
if "messages" not in st.session_state:
    # st.session_state.messages = [{"role": "system", "content": "Hi! I'm Salem, you're AI assistant. How can I help you today?"}]
    st.session_state.messages = [{"role": "assistant", "content": "أنا سالم من ديسايسيف، تفضل كيف أقدر أخدمك اليوم؟"}]

    # st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def response_generator(response: str):

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# User input
query = st.chat_input("Enter your message...")

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    logger.info(f"Processing query: {query}")
    with st.spinner("🧠 Processing..."):
        # prompt_chain = build_prompt_chain()
        
        response, full_prompt, chat_history = st.session_state.rag_pipeline.answer_rag_question(query)
        ### TODO: CHANGE THIS TO USE THE API DIRECTLY INSTEAD OF THE MANUAL INVOCATION

        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(response=response))
            # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    print("History")
    print(st.session_state.messages)
    print(f"len of stored chat history: {len(st.session_state.messages)}")
    # Rerun to update chat display
    st.rerun()
