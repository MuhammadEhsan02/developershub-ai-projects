# ============================================
# Mental Health Chatbot Fine-Tuning Code
# Fine-tune DistilGPT2 on EmpatheticDialogues Dataset
# ============================================


# -----------------------------------
# Step 1: Load Dataset
# -----------------------------------

# datasets library HuggingFace se dataset load karne ke liye use hoti hai
from datasets import load_dataset


# EmpatheticDialogues dataset load kar rahe hain
# Yeh dataset emotional conversations par based hai
dataset = load_dataset(
    "empathetic_dialogues",          # dataset name
    revision="refs/convert/parquet"  # optimized parquet version
)


# dataset structure print karega
# train, validation, test splits show karega
print(dataset)


# first training example print karega
print(dataset["train"][0])


# -----------------------------------
# Step 2: Format Dataset for Chat Training
# -----------------------------------

# is function ka purpose dataset ko chatbot format me convert karna hai
def format_chat(example):

    # user message
    user = example["prompt"]

    # bot response
    bot = example["utterance"]

    # chatbot format create karte hain
    # Example:
    # User: I feel sad
    # Bot: I understand how you feel
    example["text"] = f"User: {user}\nBot: {bot}"

    return example


# dataset ke har example par format_chat function apply karega
dataset = dataset.map(format_chat)


# formatted text print karega
print(dataset["train"][0]["text"])



# -----------------------------------
# Step 3: Load Tokenizer
# -----------------------------------

# tokenizer text ko numbers (tokens) me convert karta hai
from transformers import AutoTokenizer


# DistilGPT2 tokenizer load karega
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")


# GPT2 me default padding token nahi hota
# isliye EOS token ko padding token set karte hain
tokenizer.pad_token = tokenizer.eos_token



# -----------------------------------
# Step 4: Tokenize Dataset
# -----------------------------------

# tokenize function text ko input_ids me convert karega
def tokenize(example):

    return tokenizer(

        example["text"],

        # long text ko cut karega
        truncation=True,

        # fixed length padding karega
        padding="max_length",

        # maximum length
        max_length=128
    )


# batched=True means multiple examples ek sath tokenize honge
tokenized_dataset = dataset.map(tokenize, batched=True)


# first example ke tokens print karega
print(tokenized_dataset["train"][0]["input_ids"])



# -----------------------------------
# Step 5: Load Model
# -----------------------------------

# DistilGPT2 model load karega
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilgpt2")



# -----------------------------------
# Step 6: Add Labels for Training
# -----------------------------------

# labels training ke liye required hote hain
# causal LM me labels = input_ids
def add_labels(example):

    # input_ids copy karke labels bana rahe hain
    example["labels"] = example["input_ids"].copy()

    return example


# labels add kar rahe hain dataset me
tokenized_dataset = tokenized_dataset.map(add_labels)



# -----------------------------------
# Step 7: Training Setup
# -----------------------------------

# Trainer aur training arguments import karein
from transformers import Trainer, TrainingArguments


# training configuration
training_args = TrainingArguments(

    # model save location
    output_dir="./mental_health_model",

    # batch size per GPU/CPU
    per_device_train_batch_size=4,

    # number of epochs
    num_train_epochs=1,

    # logging interval
    logging_steps=100,

    # only 1 checkpoint save karega
    save_total_limit=1,
)



# -----------------------------------
# Step 8: Create Trainer
# -----------------------------------

trainer = Trainer(

    # model jo train hoga
    model=model,

    # training arguments
    args=training_args,

    # training dataset
    train_dataset=tokenized_dataset["train"].select(range(5000))
    # only 5000 examples use kar rahe hain testing ke liye
)



# -----------------------------------
# Step 9: Start Training
# -----------------------------------

# model training start karega
trainer.train()



# -----------------------------------
# Step 10: Save Fine-Tuned Model
# -----------------------------------

# trained model save karega
trainer.save_model("./mental_health_model")


# tokenizer bhi save karega
tokenizer.save_pretrained("./mental_health_model")


print("Training Complete ✅")



# ============================================
# Step 11: Test Fine-Tuned Model
# ============================================


# tokenizer load karega saved model ka
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./mental_health_model")


# trained model load karega
model = AutoModelForCausalLM.from_pretrained("./mental_health_model")



# test input
user_input = "I am feeling stressed about exams."


# input format create karte hain
inputs = tokenizer(

    f"User: {user_input}\nBot:",

    return_tensors="pt"
)



# model response generate karega
outputs = model.generate(

    **inputs,

    # maximum response length
    max_length=150,

    # padding token
    pad_token_id=tokenizer.eos_token_id,

    # randomness enable
    do_sample=True,

    # top tokens selection
    top_p=0.9,

    # creativity level
    temperature=0.7
)



# tokens → text conversion
response = tokenizer.decode(

    outputs[0],

    skip_special_tokens=True
)


# sirf bot ka response print karega
print(response.split("Bot:")[1].strip())