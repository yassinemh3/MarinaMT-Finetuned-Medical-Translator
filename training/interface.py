from transformers import MarianMTModel, MarianTokenizer

def translate_text(input_text, model_path="./marian-mt-finetuned-medical"):
    # Load the fine-tuned model and tokenizer
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    # Generate translation
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_text

# Example usage
if __name__ == "__main__":
    input_text = "Dies ist eine medizinische Konfiguration."
    translated_text = translate_text(input_text)
    print(f"Translated Text: {translated_text}")