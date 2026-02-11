# update_training_format.py

import json
import os

def convert_to_gemma_format(data):
    """Convert from old format to Gemma chat format"""
    converted = []
    
    for item in data:
        # Gemma chat format
        text = (
            f"<start_of_turn>user\n{item['instruction']}<end_of_turn>\n"
            f"<start_of_turn>model\n{item['output']}<end_of_turn>"
        )
        
        converted.append({
            "text": text,
            "note_id": item['note_id']
        })
    
    return converted

# Convert all dimensions
# DIMENSIONS = ["completeness", "accuracy", "compliance", "risk", "clarity"]
DIMENSIONS = ["risk"]

for dim in DIMENSIONS:
    # Load old format
    train = json.load(open(f"data/finetuning/{dim}_train.json"))
    test = json.load(open(f"data/finetuning/{dim}_test.json"))
    
    # Convert
    train_gemma = convert_to_gemma_format(train)
    test_gemma = convert_to_gemma_format(test)
    
    os.makedirs("data/gemma_format/", exist_ok=True)
    # Save
    json.dump(train_gemma, open(f"data/gemma_format/{dim}_train.json", 'w'), indent=2)
    json.dump(test_gemma, open(f"data/gemma_format/{dim}_test.json", 'w'), indent=2)
    
    print(f"Converted {dim}: {len(train_gemma)} train, {len(test_gemma)} test")

print("\nSample:")
print(train_gemma[0]['text'][:300])