# finetune_agent.py
# Run this in Kaggle notebook with GPU enabled

# ============================================================================
# SETUP
# ============================================================================

# Install dependencies
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q --upgrade transformers datasets

import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DIMENSION = "completeness"  # Change this for each agent
MODEL_NAME = "google/medgemma-4b-it"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
OUTPUT_DIR = f"/kaggle/working/models/{DIMENSION}_agent"

# ============================================================================
# LOAD DATA
# ============================================================================

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

print(f"Loading {DIMENSION} training data...")
train_data = load_json(f"/kaggle/input/training-data/finetuning/{DIMENSION}_train.json")
test_data = load_json(f"/kaggle/input/training-data/finetuning/{DIMENSION}_test.json")

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# ============================================================================
# FORMAT DATA FOR TRAINING
# ============================================================================

def format_prompt(sample):
    """Format as instruction-following prompt"""
    return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list([
    {"text": format_prompt(item)} for item in train_data
])

test_dataset = Dataset.from_list([
    {"text": format_prompt(item)} for item in test_data
])

print("\nSample training example:")
print(train_dataset[0]['text'][:500])

# ============================================================================
# LOAD MODEL WITH LORA
# ============================================================================

print("\nLoading model with LoRA...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # 4-bit quantization for memory efficiency
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"\nTrainable parameters: {model.get_nb_trainable_parameters()}")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=200,  # Increase if needed
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
)

# ============================================================================
# FINE-TUNE
# ============================================================================

print("\nStarting fine-tuning...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

# Train
trainer.train()

print("\nTraining complete!")

# ============================================================================
# SAVE MODEL
# ============================================================================

print(f"\nSaving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Also save merged model (LoRA + base)
print("Saving merged model...")
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged",
    tokenizer,
    save_method="merged_16bit",
)

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "="*80)
print("EVALUATION ON TEST SET")
print("="*80)

# Load for inference
FastLanguageModel.for_inference(model)

def predict_score(note_text):
    """Predict score for a clinical note"""
    prompt = f"""### Instruction:
Score the {DIMENSION} of this clinical note (0-100):

{note_text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=False,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract number from response
    response = response.split("### Response:")[-1].strip()
    
    try:
        return int(response)
    except:
        # Try to extract first number
        import re
        numbers = re.findall(r'\d+', response)
        return int(numbers[0]) if numbers else 50  # Default fallback

# Evaluate on test set
predictions = []
actuals = []

print(f"\nEvaluating on {len(test_data)} test samples...")

for i, item in enumerate(test_data[:30]):  # Limit for speed
    # Get note from original split file
    test_split = load_json("/kaggle/input/training-data/splits/test_split.json")
    note = next(n for n in test_split if n['note_id'] == item['note_id'])
    
    pred = predict_score(note['note_text'])
    actual = int(item['output'])
    
    predictions.append(pred)
    actuals.append(actual)
    
    if i < 5:  # Show first 5
        print(f"\n{item['note_id']}")
        print(f"  Predicted: {pred}, Actual: {actual}, Error: {abs(pred-actual)}")

# Calculate metrics
predictions = np.array(predictions)
actuals = np.array(actuals)

mae = np.mean(np.abs(predictions - actuals))
rmse = np.sqrt(np.mean((predictions - actuals)**2))
max_error = np.max(np.abs(predictions - actuals))

print("\n" + "="*80)
print("METRICS")
print("="*80)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Max Error: {max_error:.2f}")
print(f"Accuracy within 10 points: {np.mean(np.abs(predictions - actuals) <= 10)*100:.1f}%")
print(f"Accuracy within 20 points: {np.mean(np.abs(predictions - actuals) <= 20)*100:.1f}%")

# Save results
results = {
    "dimension": DIMENSION,
    "mae": float(mae),
    "rmse": float(rmse),
    "max_error": float(max_error),
    "predictions": predictions.tolist(),
    "actuals": actuals.tolist(),
}

with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/evaluation_results.json")
print("\nDone!")