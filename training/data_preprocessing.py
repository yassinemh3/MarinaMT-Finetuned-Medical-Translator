from datasets import Dataset

def load_parallel_texts(en_file, de_file):
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    with open(de_file, 'r', encoding='utf-8') as f:
        de_lines = f.readlines()
    
    assert len(en_lines) == len(de_lines), "Mismatch in the number of lines between English and German files."
    
    data = {"translation": [{"en": en.strip(), "de": de.strip()} for en, de in zip(en_lines, de_lines)]}
    return Dataset.from_dict(data)

def preprocess_dataset(dataset, tokenizer):
    def preprocess_function(examples):
        inputs = [ex["de"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    return dataset.map(preprocess_function, batched=True)


