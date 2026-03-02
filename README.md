# DevelopersHub AI/ML Engineering Projects

**Author:** Muhammad Ehsan  
**Program:** DevelopersHub Corporation AI/ML Engineering Internship  

Welcome to my repository for the AI/ML Engineering Internship at DevelopersHub Corporation. This repository contains my implementations of various artificial intelligence and machine learning tasks, showcasing a progression of skills from fundamental data exploration to advanced predictive modeling and natural language processing. 

These projects highlight my proficiency in exploratory data analysis (EDA), regression and classification algorithms, and the deployment of Large Language Models (LLMs) using modern prompt engineering and fine-tuning techniques.

---

## Repository Directory Structure

```text
developershub-ai-projects/
│
├── task1-iris-eda/                 # Task 1: Exploratory Data Analysis
│   ├── explore_iris.py             # Main EDA script
│   └── README.md
│
├── task2-stock-prediction/         # Task 2: Time-Series Forecasting
│   ├── predict_stocks.py           # Stock prediction script
│   └── README.md
│
├── task3-heart-disease/            # Task 3: Binary Classification
│   ├── train_heart_model.py        # Heart disease model training
│   ├── heart.csv                   # Dataset
│   └── README.md
│
├── task4-health-chatbot/           # Task 4: LLM-based API Chatbot
│   ├── chatbot.py                  # Agent LLM script
│   ├── Dockerfile                  # Containerization
│   └── README.md
│
├── task5-empathetic-chatbot/       # Task 5: LLM Fine-Tuning & RAG
│   ├── finetune_model.py           # Model tuning script
│   ├── hybrid_chatbot.py           # RAG and inference script
│   ├── app.py                      # Streamlit UI
│   └── README.md
│
├── task6-house-pricing/            # Task 6: Multivariable Regression
│   ├── main.py                     # House pricing model training
│   ├── test_model.py               # Model evaluation
│   ├── house_prices.csv           # Dataset
│   └── README.md
│
└── README.md                       # Root documentation
```

---

## Projects Overview

Here is a summary of the internship tasks completed in this repository:

1. **[Task 1: Exploring and Visualizing a Simple Dataset](./task1-iris-eda)**  
   * **Domain:** Data Analytics & Visualization  
   * **Focus:** Performing EDA on the standard Iris dataset to uncover data trends, distributions, and feature correlations using `pandas`, `matplotlib`, and `seaborn`.

2. **[Task 2: Predict Future Stock Prices (Short-Term)](./task2-stock-prediction)**  
   * **Domain:** Time-Series Forecasting & Regression  
   * **Focus:** Fetching historical stock data dynamically via `yfinance` and implementing machine learning models to predict short-term stock closing prices.

3. **[Task 3: Heart Disease Prediction](./task3-heart-disease)**  
   * **Domain:** Medical Diagnostics & Classification  
   * **Focus:** Developing robust binary classifiers (Logistic Regression and Decision Trees) to predict a patient's risk of heart disease based on clinical metrics.

4. **[Task 4: General Health Query Chatbot](./task4-health-chatbot)**  
   * **Domain:** Conversational AI & Prompt Engineering  
   * **Focus:** Building a conversational agent leveraging external LLM APIs (OpenAI/Mistral), featuring strict safety guardrails to ethically answer health-related queries.

5. **[Task 5: Mental Health Support Chatbot](./task5-empathetic-chatbot)**  
   * **Domain:** LLM Fine-Tuning & Retrieval-Augmented Generation (RAG)  
   * **Focus:** Fine-tuning base models (DistilGPT2) on empathetic dialogue datasets to create a supportive mental health chatbot, augmented with vector database knowledge retrieval and presented via a Streamlit UI.

6. **[Task 6: House Price Prediction](./task6-house-pricing)**  
   * **Domain:** Multivariable Regression  
   * **Focus:** Creating an end-to-end ML pipeline to predict real estate values based on property features, including thorough feature scaling, model evaluation (MAE/RMSE), and persistent model saving.

---

## Getting Started

Follow these instructions to clone the repository and run the projects locally on your machine.

### Prerequisites
* Python 3.8 or higher installed
* `pip` (Python package installer)
* Git

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MuhammadEhsan02/developershub-ai-projects.git
   cd developershub-ai-projects
   ```

2. **Navigate into a specific task directory:**
   Each task operates independently. It is recommended to navigate into the task folder you wish to execute. For example:
   ```bash
   cd task1-iris-eda
   ```

3. **Install the required dependencies:**
   Many tasks require specific libraries such as `scikit-learn`, `pandas`, `transformers`, or `streamlit`. Check the individual task's `README.md` or `pyproject.toml` files, or install the standard data science packages globally:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn yfinance transformers torch streamlit openai-agents faiss-cpu
   ```

4. **Run the project script:**
   Execute the primary Python file for the respective task. For example:
   ```bash
   python explore_iris.py
   ```
   *(Please refer to the `README.md` inside each specific task folder for detailed execution commands, especially for Streamlit apps or Docker deployments.)*

