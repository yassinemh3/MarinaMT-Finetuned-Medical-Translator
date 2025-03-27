from data_preprocessing import load_parallel_texts, preprocess_dataset
from model_training import train_model
from config import EN_FILE, DE_FILE, OUTPUT_DIR
from transformers import MarianTokenizer

# Load the tokenizer
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# Load and preprocess the dataset
dataset = load_parallel_texts(EN_FILE, DE_FILE)
tokenized_dataset = preprocess_dataset(dataset, tokenizer)  # Pass the tokenizer here

# Train the model
train_model(tokenized_dataset, output_dir=OUTPUT_DIR)