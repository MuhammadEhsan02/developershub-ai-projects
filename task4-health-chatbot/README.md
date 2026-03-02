# Task 4: General Health Query Chatbot

## Task Objective
The objective is to develop a conversational AI agent capable of answering general health-related questions using a Large Language Model (LLM) while utilizing strict prompt engineering to ensure safety and clarity.

## Dataset Used
* **Source:** No static dataset; utilizes pre-trained LLM knowledge accessed via API.

## Models and Techniques Applied
* **Models:** OpenAI GPT / Mistral-7B via API.
* **Techniques:** Advanced prompt engineering (e.g., persona adoption as a "helpful medical assistant"), API integration, and implementation of safety filters to prevent the dissemination of harmful medical advice.

## Key Results and Findings
The resulting chatbot effectively answers general health queries (e.g., causes of a sore throat) in a clear, empathetic tone. Safety guardrails successfully trigger disclaimers, ensuring the bot does not attempt to replace professional medical diagnosis.
