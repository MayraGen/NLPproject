# src/fine_tune_t5.py
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
import torch

# 1. Load Base Model & Tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 2. Prepare Dataset (Example: CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0")  # Replace with your PDF-derived dataset

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True, 
        padding="max_length"
    )
    
    labels = tokenizer(
        examples["highlights"], 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 3. Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,            # Adjust based on your dataset size
    per_device_train_batch_size=8, # Reduce if OOM errors occur
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 4. Create Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 6. Train
trainer.train()

# 7. Save Fine-Tuned Model
model.save_pretrained("models/fine_tuned_t5")
tokenizer.save_pretrained("models/fine_tuned_t5/tokenizer")

# 8. Optional: Test Inference
input_text = "summarize: Your input text here..."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))