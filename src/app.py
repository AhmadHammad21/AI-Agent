import streamlit as st
from utils.logger import get_logger
import requests
import random
import time

# Set up logger
logger = get_logger(__name__)

API_URL = "http://localhost:5000/api/v1/chatbot/answer"  # Replace with your actual URL

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
    

# Session state management
if "messages" not in st.session_state:
    # st.session_state.messages = [{"role": "system", "content": "Hi! I'm Salem, you're AI assistant. How can I help you today?"}]
    st.session_state.messages = [{"role": "assistant", "content": "أنا سالم من ديسايسيف، تفضل كيف أقدر أخدمك اليوم؟"}]

    # st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def response_generator(response: str):

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def generate_user_id(length=6):
    return str(random.randint(10**(length-1), 10**length - 1))

# # User input
query = st.chat_input("Enter your message...")


user_id = generate_user_id()  # You can dynamically assign this too
session_id = user_id  # Ideally, generate or persist one per session

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Send query to your FastAPI endpoint
    try:
        with st.spinner("🧠 Processing..."):
            response = requests.post(API_URL, json={
                "user_id": user_id,
                "session_id": session_id,
                "query": query
            })

            if response.status_code == 200:
                data = response.json()
                assistant_reply = data.get("answer", "❌ لم يتم العثور على رد.")
            else:
                assistant_reply = "⚠️ حدث خطأ في الخادم."
    except Exception as e:
        assistant_reply = f"❌ خطأ في الاتصال: {e}"

    # Show assistant response
    with st.chat_message("assistant"):
        st.write_stream(response_generator(response=assistant_reply))

    # Save display-only chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Rerun to refresh UI
    st.rerun()