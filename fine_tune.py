# 1. Import Libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# 2. Load and Prepare Dataset with an Evaluation Set
def load_and_prepare_dataset():
    dataset = load_dataset("ealvaradob/phishing-dataset", "texts", trust_remote_code=True)
    df = dataset['train'].to_pandas()
    
    # Split the original dataset into training and a temporary set (remaining)
    train_df, remaining_df = train_test_split(df, test_size=0.4, random_state=42)  # Adjust size based on your preference
    
    # Split the temporary set into evaluation and test sets
    eval_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)  # Split the remaining data equally
    
    # Convert DataFrames to Datasets
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    eval_ds = Dataset.from_pandas(eval_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    
    return train_ds, eval_ds, test_ds

# Tokenization and data preparation function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# 3. Model and Tokenizer Setup
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels accordingly

# 4. Data Preprocessing
train_ds, eval_ds, test_ds = load_and_prepare_dataset()
tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_eval = eval_ds.map(tokenize_function, batched=True)
tokenized_test = test_ds.map(tokenize_function, batched=True)

# 5. Training Setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",  # Add this to perform evaluation at the end of each epoch
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,  # Use the tokenized evaluation set here
    data_collator=data_collator,
)

# 6. Training
trainer.train()

# 7. Evaluation
trainer.evaluate(tokenized_test)  # Evaluate on the test set

# 8. Save Model
model.save_pretrained("./my_tinybert_phishing_model")
tokenizer.save_pretrained("./my_tinybert_phishing_model")
