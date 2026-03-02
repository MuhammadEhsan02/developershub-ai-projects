# ============================================
# Hybrid Mental Health Chatbot
# Gemini Agent + Fine-tuned DistilGPT2 + RAG
# ============================================


# --------------------------------------------
# Import required libraries
# --------------------------------------------

# Agent SDK imports

# Agent → AI agent create karne ke liye
# Runner → agent ko run karne ke liye
# OpenAIChatCompletionsModel → Gemini ko OpenAI-compatible format me use karne ke liye
# AsyncOpenAI → async client jo Gemini API call karega, Gemini API client jo asynchronous API calls handle karega
# set_tracing_disabled → debugging tracing off karne ke liye, tracing logs disable karne ke liye
# function_tool → function ko agent tool banane ke liye , Agents SDK me tools ko tool object banana hota hai using: function_tool decorator

# Agents SDK imports
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    set_tracing_disabled,
    function_tool
)

# dotenv → .env file se API key load karne ke liye
# os → environment variables access karne ke liye
from dotenv import load_dotenv
import os

# transformers → HuggingFace library jo fine-tuned model load karegi
# AutoTokenizer → text ko tokens me convert karega
# AutoModelForCausalLM → causal language/fine-tuned language model load karega
from transformers import AutoTokenizer, AutoModelForCausalLM

# torch → PyTorch backend, model execution ke liye required deep learning library
import torch

# RAG system import
# retrieve_context → vector DB se relevant knowledge retrieve karega
from rag_system import retrieve_context

# -----------------------------------
# Setup environment
# -----------------------------------

# .env file load karega (GEMINI_API_KEY), # Is file me GEMINI_API_KEY stored hoti hai
load_dotenv()

# tracing disable karega (clean output ke liye)
set_tracing_disabled(True)

# -----------------------------------
# Load Fine-Tuned Mental Health Model
# -----------------------------------

print("Loading fine-tuned mental health model...")

# tokenizer load karega (text → tokens conversion)
tokenizer = AutoTokenizer.from_pretrained("./mental_health_model")

# fine-tuned DistilGPT2 model load karega
model_local = AutoModelForCausalLM.from_pretrained(
    "./mental_health_model"
)

print("Model loaded successfully ✅")

# -----------------------------------
# Create function (Tool)
# -----------------------------------

# --------------------------------------------
# Tool 1: RAG Tool
# --------------------------------------------
# function_tool decorator is function ko Tool object bana deta hai

@function_tool
def rag_tool(user_input: str) -> str:
    """
    Yeh function user input se relevant mental health
    knowledge retrieve karega using RAG system
    """

    # retrieve_context vector DB se relevant info fetch karega
    context = retrieve_context(user_input)

    return context


# --------------------------------------------
# Tool 2: Fine-Tuned Model Tool
# --------------------------------------------

# function_tool decorator is function ko agent tool bana deta hai
# Agents SDK expects tool object like:

# Tool(
#    name="empathetic_response",
#    description="...",
#    function=...
# )

# @function_tool automatically converts function → Tool object.
@function_tool
def empathetic_response(user_input: str) -> str:
    """
    Yeh function fine-tuned DistilGPT2 model use karke
    empathetic response generate karega.
    
    Yeh RAG context + user input combine karega.
    """
    # Step 1: RAG context retrieve karega
    context = retrieve_context(user_input)

    # Step 2: prompt create karega (context + user input)
    prompt = f"""
Context:
{context}

User: {user_input}
Bot:
"""

    # Step 3: prompt ko tokens me convert karega
    inputs = tokenizer(
        prompt,
        return_tensors="pt" # pytorch tensor format
    )

    # Step 4: model response generate karega
    outputs = model_local.generate(

        **inputs, # tokenized input

        max_length=200,  # max response length

        pad_token_id=tokenizer.eos_token_id, # padding token

        do_sample=True,  # randomness enable karega (sampling)

        temperature=0.7,  # creativity level (0.7 = balanced)

        top_p=0.9  # best tokens choose karega
    )

    # Step 5: tokens → text convert karega
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Step 6: sirf Bot response return karega
    final_response = response.split("Bot:")[-1].strip()

    return final_response

# -----------------------------------
# Configure Gemini Model
# -----------------------------------
 # Gemini model ko OpenAI-compatible format me configure karte hain
gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", # Gemini model name
    openai_client=AsyncOpenAI( # async Gemini client create karte hain
        api_key=os.getenv("GEMINI_API_KEY"), # API key environment variable se  API key load from .env
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # # Gemini base URL (important) endpoint for API calls , hum Gemini API ko OpenAI-compatible format me use karne ke liye base URL set karna zaroori hai, hit karne ke liye https://generativelanguage.googleapis.com/v1beta/openai/ endpoint use hota hai, is URL ke through hum Gemini API ko OpenAI API ki tarah call kar sakte hain
    )
)

# --------------------------------------------
# Create Mental Health Agent
# --------------------------------------------

mental_health_agent = Agent( # # Agent create karte hain jo Gemini use karega 
    name="Mental Health Support Agent", # agent name
     # agent instructions (VERY IMPORTANT)
    instructions="""
You are a professional mental health support assistant.

Your tasks:

1. Understand user's emotions carefully
2. Use rag_tool to retrieve relevant mental health knowledge
3. Use empathetic_response tool to generate empathetic response
4. Provide caring and supportive responses
5. Never provide medical diagnosis
6. Always be kind and supportive
""",
    # agent ko Gemini model use karne ke liye configure karte hain  Gemini model use hoga reasoning ke liye
    model=gemini_model,
        # tools list (RAG tool + fine-tuned model tool)
    tools=[

        rag_tool,              # RAG knowledge tool

        empathetic_response    # fine-tuned response tool
    ]
)

# -----------------------------------
# Chat loop
# -----------------------------------

print("\nMental Health Chatbot Ready ✅")
print("Type 'exit' to quit\n")
# # infinite loop for chatting
# while True:

#     user_input = input("You: ") # user input lega

#     if user_input.lower() == "exit": # exit condition 
#         break
#      # agent run hoga user input ke saath aur result return karega
#     result = Runner.run_sync(
#         mental_health_agent, # agent run karte hain user input ke saath  konsa agent run karna hai
#         user_input # user input
#     )
#     # final output print karega (agent ka response)
#     print("\nBot:", result.final_output)
#     print()












# # ============================================
# # Hybrid Mental Health Chatbot
# # Gemini Agent + Fine-tuned DistilGPT2
# # ============================================

# # -----------------------------------
# # Import required libraries
# # -----------------------------------

# # Agent SDK imports
# # Agent → AI agent create karne ke liye
# # Runner → agent ko run karne ke liye
# # OpenAIChatCompletionsModel → Gemini ko OpenAI-compatible format me use karne ke liye
# # AsyncOpenAI → async client jo Gemini API call karega
# # set_tracing_disabled → debugging tracing off karne ke liye
# # function_tool → function ko agent tool banane ke liye , Agents SDK me tools ko tool object banana hota hai using: function_tool decorator

# # Agents SDK imports
# from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool

# # dotenv → .env file se API key load karne ke liye
# # os → environment variables access karne ke liye
# # dotenv + os
# from dotenv import load_dotenv
# import os

# # transformers (your fine-tuned model)
# # transformers → HuggingFace library jo fine-tuned model load karegi
# # AutoTokenizer → text ko tokens me convert karega
# # AutoModelForCausalLM → causal language model load karega
# from transformers import AutoTokenizer, AutoModelForCausalLM
# # torch → model execution ke liye required deep learning library
# import torch

# # -----------------------------------
# # Setup environment
# # -----------------------------------

# # .env file load karega (GEMINI_API_KEY)

# load_dotenv()
# # tracing disable karega (clean output ke liye)
# set_tracing_disabled(True)

# # -----------------------------------
# # Load Fine-Tuned Model
# # -----------------------------------

# print("Loading fine-tuned mental health model...")
# # tokenizer load karega (text → tokens conversion)
# tokenizer = AutoTokenizer.from_pretrained("./mental_health_model")
# # fine-tuned DistilGPT2 model load karega
# model_local = AutoModelForCausalLM.from_pretrained("./mental_health_model")

# print("Model loaded successfully ✅")

# # -----------------------------------
# # Create function (Tool)
# # -----------------------------------
# # function_tool decorator is function ko agent tool bana deta hai
# # Agents SDK expects tool object like:

# # Tool(
# #    name="empathetic_response",
# #    description="...",
# #    function=...
# # )

# # @function_tool automatically converts function → Tool object.
# @function_tool
# def empathetic_response(user_input: str) -> str:
#     """
#     Yeh function user input lega aur fine-tuned model se
#     empathetic response generate karega
#     """

#     # Input ko proper format me convert karte hain
#     # Example:
#     # User: I feel sad
#     # Bot:
#     inputs = tokenizer(
#         f"User: {user_input}\nBot:",
#         return_tensors="pt" # pytorch tensor format
#     )
    
#      # Model response generate karega
#     outputs = model_local.generate(
#         **inputs, # tokenized input
#         max_length=150,  # maximum response length
#         pad_token_id=tokenizer.eos_token_id,  # padding token
#         do_sample=True, # sampling enable karega (randomness)
#         temperature=0.7, # creativity level (0.7 = balanced)
#         top_p=0.9 # best tokens choose karega
#     )
#      # tokens → text conversion
#     response = tokenizer.decode(
#         outputs[0],
#         skip_special_tokens=True
#     )
#     # sirf bot ka answer return karega (User: ke baad ka part)
#     return response.split("Bot:")[-1].strip()


# # -----------------------------------
# # Configure Gemini Model
# # -----------------------------------
#  # Gemini model ko OpenAI-compatible format me configure karte hain
# gemini_model = OpenAIChatCompletionsModel(
#     model="gemini-2.5-flash", # Gemini model name
#     openai_client=AsyncOpenAI( # async Gemini client create karte hain
#         api_key=os.getenv("GEMINI_API_KEY"), # API key environment variable se  API key load from .env
#         base_url="https://generativelanguage.googleapis.com/v1beta/openai/", # # Gemini base URL (important) endpoint for API calls , hum Gemini API ko OpenAI-compatible format me use karne ke liye base URL set karna zaroori hai, hit karne ke liye https://generativelanguage.googleapis.com/v1beta/openai/ endpoint use hota hai, is URL ke through hum Gemini API ko OpenAI API ki tarah call kar sakte hain
#     )
# )

# # -----------------------------------
# # Create Agent
# # -----------------------------------

# mental_health_agent = Agent( # # Agent create karte hain jo Gemini use karega 
#     name="Mental Health Support Agent", # agent ka naam
#      # agent instructions (VERY IMPORTANT)
#     instructions="""
# You are a professional mental health support assistant.

# Your job:

# 1. Understand user's emotions
# 2. Use empathetic_response tool
# 3. Provide caring and supportive answers
# 4. Never give medical diagnosis
# 5. Always be empathetic and kind
# """,
#     # agent ko Gemini model use karne ke liye configure karte hain  Gemini model use hoga reasoning ke liye
#     model=gemini_model,
#      # tool list (fine-tuned model as tool)
#     tools=[empathetic_response] # agent ke paas empathetic_response tool available hoga
# )

# # -----------------------------------
# # Chat loop
# # -----------------------------------

# print("\nMental Health Chatbot Ready ✅")
# print("Type 'exit' to quit\n")
# # infinite loop for chatting
# while True:

#     user_input = input("You: ") # user input lega

#     if user_input.lower() == "exit": # exit condition 
#         break
#      # agent run hoga user input ke saath aur result return karega
#     result = Runner.run_sync(
#         mental_health_agent, # agent run karte hain user input ke saath  konsa agent run karna hai
#         user_input # user input
#     )
#     # final output print karega (agent ka response)
#     print("\nBot:", result.final_output)
#     print()