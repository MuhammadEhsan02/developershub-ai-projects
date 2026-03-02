# app.py
import streamlit as st
from agents import Runner
from hybrid_chatbot import mental_health_agent  # Import agent from hybrid_chatbot.py

# Page configuration
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="centered")
st.title("🧠 Hybrid Mental Health Support Chatbot")
st.markdown(
    "This chatbot combines **RAG**, **Fine-Tuned DistilGPT2**, and **Gemini Agent** "
    "to provide empathetic and safe responses."
)

# Sidebar instructions
st.sidebar.header("Instructions")
st.sidebar.write(
    """
    - Type your message below.
    - Press Enter or click 'Send'.
    - Type 'exit' to end the conversation.
    """
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    if user_input.lower() == "exit":
        st.session_state.messages.append({"user": user_input, "bot": "Chat ended."})
    else:
        # Run the agent
        result = Runner.run_sync(mental_health_agent, user_input)
        # Store in session
        st.session_state.messages.append({"user": user_input, "bot": result.final_output})

# Display chat history
for chat in st.session_state.messages:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")