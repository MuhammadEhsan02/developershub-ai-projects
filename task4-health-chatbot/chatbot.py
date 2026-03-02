# -----------------------------------
# Task 4: General Health Query Chatbot
# Using OpenAI Agents SDK + Gemini + Streamlit
#
# Objective:
# Build a safe and interactive chatbot that answers
# general health-related questions using an LLM.
#
# Features:
# - Uses Gemini LLM via OpenAI Agents SDK
# - Streamlit web interface
# - Safety filter for dangerous queries
# - Chat history memory
# - Friendly and safe responses
# -----------------------------------

# -----------------------------------
# Import Streamlit for Web UI
# -----------------------------------
import streamlit as st

# -----------------------------------
# Import Agent SDK components
# -----------------------------------
# Agent → defines chatbot behavior
# Runner → executes the agent
# OpenAIChatCompletionsModel → connects LLM
# AsyncOpenAI → handles Gemini API connection
# set_tracing_disabled → disables internal logs
# -----------------------------------

# Agent SDK imports
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled

# -----------------------------------
# Import dotenv and os
# dotenv → loads environment variables
# os → access API key securely
# -----------------------------------

from dotenv import load_dotenv
import os

# -----------------------------------
# Load environment variables from .env file
# This allows secure API key usage
# -----------------------------------

load_dotenv()


# -----------------------------------
# Disable internal tracing/logging
# This keeps output clean
# -----------------------------------

set_tracing_disabled(True)

# -----------------------------------
# Configure Gemini LLM Model
# -----------------------------------
# OpenAIChatCompletionsModel acts as wrapper
# AsyncOpenAI connects to Gemini API endpoint
# -----------------------------------

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", # Gemini model name
    openai_client=AsyncOpenAI(   # API configuration
        api_key=os.getenv("GEMINI_API_KEY"),  # Load API key securely
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini OpenAI-compatible endpoint
    )
)

# -----------------------------------
# Safety Filter Function
# -----------------------------------
# Purpose:
# Prevent chatbot from answering dangerous
# or emergency-related questions.
#
# Returns:
# True  → Safe question
# False → Dangerous question
# -----------------------------------

def safety_filter(user_input):
      # List of dangerous keywords
    dangerous_keywords = [
        "suicide",
        "kill myself",
        "overdose",
        "self harm",
        "die",
        "emergency"
    ]
      # Check if any dangerous word exists in user input
    for word in dangerous_keywords:

        if word in user_input.lower():

            return False
     # Safe question
    return True


# -----------------------------------
# Create Medical Assistant Agent
# -----------------------------------
# This Agent defines chatbot personality,
# behavior, and safety instructions.
# -----------------------------------

medical_assistant_chatbot = Agent(

    name="Medical_Assistant", # Agent name
      # Prompt Engineering Instructions
    instructions="""
You are a helpful medical assistant chatbot.

Responsibilities:
- Provide general health information
- Explain symptoms simply
- Suggest healthy lifestyle tips

Safety Rules:
- Do NOT diagnose
- Do NOT prescribe medicine
- Always recommend doctor consultation when needed

Keep answers short and friendly.
""",
       # Connect LLM model
    model=model
)

# -----------------------------------
# Chat Function
# -----------------------------------
# This function:
# 1. Checks safety
# 2. Runs the Agent
# 3. Returns response
# -----------------------------------

def ask_health_question(user_question):
     # Step 1: Safety check
    if not safety_filter(user_question):

        return "⚠️ This may require immediate medical attention. Please contact a doctor."

    try:
         # Step 2: Run Agent with user question as input
        response = Runner.run_sync(
            starting_agent=medical_assistant_chatbot, # Starting agent
            input=user_question # User input
        )

        return response.final_output  # Step 3: Return final response

    except Exception as e:
        # Handle errors safely
        return f"Error: {e}"


# -----------------------------------
# Streamlit UI Configuration
# -----------------------------------
# Sets page title, icon, and layout
# -----------------------------------

st.set_page_config(
    page_title="Health Assistant",
    page_icon="🩺",
    layout="centered"
)
# -----------------------------------
# Display title and description
# -----------------------------------
st.title("🩺 General Health Assistant")
st.write("Ask general health-related questions safely.")

# -----------------------------------
# Initialize Chat History
# -----------------------------------
# session_state stores conversation memory
# -----------------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []


# -----------------------------------
# Display Previous Chat Messages
# -----------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------------
# Take User Input from Chat Box
# -----------------------------------
user_input = st.chat_input("Ask your health question...")
# -----------------------------------
# Process User Input
# -----------------------------------
if user_input:

     # Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
     # Display user message in chat interface
    with st.chat_message("user"):

        st.markdown(user_input)

    # Generate chatbot response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):
                # Call chatbot function to get response
            reply = ask_health_question(user_input)
            # Display response in chat interface
            st.markdown(reply)

    # Save bot response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })


# -----------------------------------
# Task 4 Complete
#
# Features Implemented:
# ✔ Prompt Engineering
# ✔ Gemini LLM Integration
# ✔ Agent SDK Architecture
# ✔ Safety Filter
# ✔ Streamlit Web UI
# ✔ Chat History Memory
# ✔ Error Handling
#
# This chatbot safely answers general health queries.
# -----------------------------------












# # -----------------------------------
# # Task 4: General Health Query Chatbot
# # Using OpenAI Agents SDK + Gemini
# # Objective:
# # Build a chatbot that answers general health-related questions safely.
# # -----------------------------------

# # Import Agent SDK components
# from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled

# # Import dotenv to load environment variables
# from dotenv import load_dotenv

# # Import os to access API key securely
# import os

# # -----------------------------------
# # Load Environment Variables
# # -----------------------------------

# load_dotenv() # Load .env file

# # Disable internal agent tracing/logging for cleaner output
# set_tracing_disabled(True)


# # -----------------------------------
# # Configure Gemini LLM Model
# # -----------------------------------

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash",  # Confirmed working name from list_models()
#     openai_client=AsyncOpenAI(
#         api_key=os.getenv("GEMINI_API_KEY"),# API key from .env
#         base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     )
# )


# # -----------------------------------
# # Safety Filter Function
# # -----------------------------------

# def safety_filter(user_input):
#     """
#     Prevents chatbot from answering dangerous questions
#     that may require immediate medical attention.
#     Returns False if question is risky.
#     """

#     dangerous_keywords = [
#         "suicide",
#         "kill myself",
#         "overdose",
#         "self harm",
#         "die",
#         "emergency"
#     ]

#     for word in dangerous_keywords:

#         if word in user_input.lower():

#             return False

#     return True


# # -----------------------------------
# # Define Medical Assistant Agent
# # -----------------------------------

# medical_assistant_chatbot = Agent(

#     name="Medical_Assistant",

#     instructions="""
# You are a helpful medical assistant chatbot.

# Your responsibilities:

# - Provide general health information.
# - Explain common symptoms and causes.
# - Suggest preventive care and healthy lifestyle tips.
# - Keep responses friendly, simple, and safe.

# Safety Rules:

# - Do NOT give diagnosis.
# - Do NOT prescribe medicine doses.
# - Do NOT replace professional medical advice.
# - Always recommend consulting a doctor when necessary.
# - If emergency-related question appears, advise contacting doctor immediately.

# Keep answers concise and easy to understand.
# """,

#     model=model
# )


# # -----------------------------------
# # Chat Function using Agent Runner
# # -----------------------------------

# def ask_health_question(user_question):
#     """
#     Function to handle user input and get response from the Agent.
#     Checks safety first, then runs the Agent.
#     """

#     # Safety check
#     if not safety_filter(user_question):

#         return "⚠️ This question may require immediate medical attention. Please contact a doctor immediately."

#     try:

#         # Run the Agent synchronously with the user's question as input
#         response = Runner.run_sync(

#             starting_agent=medical_assistant_chatbot,

#             input=user_question

#         )

#         return response.final_output

#     except Exception as e:

#         return f"Error: {e}"


# # -----------------------------------
# # Interactive Chat Loop
# # -----------------------------------

# print("🌸 General Health Assistant Chatbot (Gemini Powered)")
# print("Type 'exit' to quit.\n")

# while True:
#      # Take user input
#     user_input = input("You: ")
#      # Exit condition
#     if user_input.lower() == "exit":

#         print("Chatbot: Stay healthy! Goodbye.")
#         break
#      # Get chatbot response
#     reply = ask_health_question(user_input)
#      # Print response
#     print("Chatbot:", reply)
    
# # -----------------------------
# # Task Complete
# # - User can ask general health questions safely
# # - Safety filter prevents dangerous advice
# # - Gemini LLM provides friendly, informative responses
# # -----------------------------



