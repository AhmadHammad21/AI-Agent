import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000/query/"

def get_rag_response(query):
    response = requests.post(FASTAPI_URL, json={"query": query})
    return response.json().get("response", "Error: No response")



query = "Hello, My name is Ahmad Hammad"
response = get_rag_response(query)
print(response)
