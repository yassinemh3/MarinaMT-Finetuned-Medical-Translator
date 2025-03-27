from transformers import MarianMTModel, MarianTokenizer

# Path to the fine-tuned model
model_path = r"C:\Users\yassi\Downloads\Bachelor_Project\Bachelor_Project\marian_finetuning\marian-mt-finetuned-medical" 

# Load the model and tokenizer
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Read the terms from the text file
input_file = r"C:\Users\yassi\Downloads\Bachelor_Project\Bachelor_Project\unique_terms.txt"
output_file = "translated_terms.txt"

with open(input_file, "r", encoding="utf-8") as f:
    terms = [line.strip() for line in f.readlines()]  # Remove extra spaces or newlines

# Translate each term and save the results
translated_terms = []

for term in terms:
    if term:  # Ensure the term is not empty
        # Tokenize the input
        inputs = tokenizer(term, return_tensors="pt", max_length=128, truncation=True)

        # Generate the translation
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        # Save the original and translated terms
        translated_terms.append((term, translated_text))

# Write the translations to a file
with open(output_file, "w", encoding="utf-8") as f:
    for original, translated in translated_terms:
        f.write(f"{original} -> {translated}\n")

print(f"Translated {len(translated_terms)} terms.")
print(f"Translations saved to: {output_file}")