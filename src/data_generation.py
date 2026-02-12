"""
Contains common functions used during 
gold prompt creation, persona creations and 
label generation

Use the 5 synthetically generated scenarios and 2 personas
1) Generate 10 clinical notes per scenario (50 Gold prompts)
2) Generate 10 clinical notes per persona per scenario (50 Brooks; 50 Chen prompts)
3) Generate labels (5 dimensions) to represent each clinical note
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import gc
from datetime import datetime
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from kaggle_secrets import UserSecretsClient

from src.config import (
    model_name,
    scenarios,
    personas,
    base_dir,
    gold_prompt_dir,
    persona_prompt_dir,
    persona_generated_dir,
    gender_content,
    demographics_female,
    demographics_male,
)

def clear_gpu_memory():
    """Clear GPU cache"""
    torch.cuda.empty_cache()
    gc.collect()

def get_credentials(secret_name: str = "hf_token"):
    """Fetch GCP Credentials"""
    user_secrets =  UserSecretsClient()
    service_account_json = user_secrets.get_secret(secret_name)

    return service_account_json

def load_medgemma(model_name=model_name, token=None):
    """Load MedGemma model"""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # dtype=torch.float16,
        dtype=torch.bfloat16,
        device_map="auto",
        token=token
    )
    return tokenizer, model

def generate_note(prompt, tokenizer, model, max_new_tokens = 1000):
    """Generate clinical note"""
    clear_gpu_memory()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
        # eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("---")[0]] 
    )
    del inputs
    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def clean_note(generated_text):
    # Remove everything after the note ends
    # Stop at first occurrence of multiple backticks or dashes
    import re

    if "**Clinical Note**" in generated_text:
        generated_text = generated_text.split("**Clinical Note**")[1]
    
    # Split at patterns that indicate end of real content
    patterns = [r'===+', r'```+', r'---+', r'\n\n\n+']
    
    for pattern in patterns:
        parts = re.split(pattern, generated_text)
        if len(parts) > 1:
            return  max(parts, key=len).strip().strip('```\n')
    
    return generated_text.strip().strip('```\n')

def load_prompt(prompt_file: str):
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt

def load_json_prompt(prompt_file: str):
    with open(prompt_file, 'r') as f:
        prompt = json.load(f)
    return prompt

def save_note(
        clean_note: str, 
        scenario: str, 
        note_id: str, 
        persona: str = "gold",
        metadata=None,
    ):
    data = {
        "note_id": note_id.replace('.json', ''),
        "scenario": scenario,
        "persona": persona,
        "note_text": clean_note,
        "generated_at": datetime.now().isoformat()
    }
    
    if metadata:
        data['demographics'] = metadata['demographics']
    
    filename = note_id
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def save_note_with_labels(note_text, labels, persona, scenario, note_id):
    """Save note and its labels together"""
    
    data = {
        "note_id": note_id,
        "persona": persona,
        "scenario": scenario,
        "note_text": note_text,
        "ground_truth_scores": {
                    "completeness": labels["completeness"],
                    "accuracy": labels["accuracy"],
                    "compliance": labels["compliance"],
                    "risk": labels["risk"],
                    "clarity": labels["clarity"]
                },
        "quality_issues": labels["issues"],
        "generated_at": datetime.now().isoformat()
    }
    
    filename = f"{note_id}.json"
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def inject_demographics_and_gender_content(prompt_template, demographics, scenario):
    """Inject demographics AND gender-specific clinical content"""
    
    gender = demographics['gender']

    safe_dict = {
        'name': demographics['name'],
        'age': demographics['age'],
        'gender': demographics['gender'],
        'occupation': demographics['occupation']
    }
    
    # Base demographics
    filled_prompt = prompt_template
    for key, value in safe_dict.items():
        filled_prompt = filled_prompt.replace(f"{{{key}}}", str(value))
    
    # Gender-specific content (if scenario has it)
    if scenario in gender_content and gender in gender_content[scenario]:
        for key, value in gender_content[scenario][gender].items():
            placeholder = "{" + key + "}"
            filled_prompt = filled_prompt.replace(placeholder, value)
    
    return filled_prompt


def run_scenarios(
    tokenizer, 
    model, 
    base_dir: str = base_dir, 
    scenarios: list = scenarios, 
    n_notes_per_gender: int = 5,
):
    """Generate gold notes with gender-specific demographics and clinical content"""
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"GENERATING SCENARIO: {scenario}")
        print(f"{'='*80}")
        
        # Load base prompt template (with placeholders)
        prompt_template = load_prompt(f"{base_dir}/{scenario}_prompt.txt")
        
        note_counter = 0
        
        # Generate male notes
        for i in range(n_notes_per_gender):
            demographics = demographics_male[i]
            
            # Inject demographics and gender-specific content
            prompt = inject_demographics_and_gender_content(
                prompt_template, 
                demographics, 
                scenario
            )
            
            # Generate note
            note = generate_note(prompt, tokenizer, model)
            clean = clean_note(note)
            
            # Save
            note_id = f"gold_{scenario}_{note_counter:03d}"
            save_note(
                clean, 
                scenario=scenario, 
                note_id=f"{note_id}.json",
                metadata={'demographics': demographics}
            )
            
            clear_gpu_memory()
            
            print(f" Generated {note_id} ({demographics['name']}, {demographics['age']}{demographics['gender']})")
            note_counter += 1
        
        # Generate female notes
        for i in range(n_notes_per_gender):
            demographics = demographics_female[i]
            
            # Inject demographics and gender-specific content
            prompt = inject_demographics_and_gender_content(
                prompt_template, 
                demographics, 
                scenario
            )
            
            # Generate note
            note = generate_note(prompt, tokenizer, model)
            clean = clean_note(note)
            
            # Save
            note_id = f"gold_{scenario}_{note_counter:03d}"
            save_note(
                clean, 
                scenario=scenario, 
                note_id=f"{note_id}.json",
                metadata={'demographics': demographics}
            )
            
            clear_gpu_memory()
            
            print(f" Generated {note_id} ({demographics['name']}, {demographics['age']}{demographics['gender']})")
            note_counter += 1
        
        print(f"\n Completed {scenario}: {note_counter}/{2 * n_notes_per_gender} notes generated")

def transform_to_persona(tokenizer, model, gold_note, persona, path):
    """Transform gold standard note to persona style"""
    
    with open(f"{path}/{persona}_transform_prompt.txt", 'r') as f:
        transform_prompt = f.read()
    # else:
    #     with open(f"{path}/chen_transform_prompt.txt", 'r') as f:
    #         transform_prompt = f.read()
    
    full_prompt = transform_prompt.replace("{insert_gold_standard_note_here}", gold_note)
    
    transformed = generate_note(full_prompt, tokenizer, model)
    cleaned = clean_note(transformed)
    
    return cleaned


def run_personas(
        tokenizer, 
        model, 
        base_path: str = gold_prompt_dir, 
        scenarios: list = scenarios, 
        n_notes: int = 10,
        personas: list = personas,
        persona_path: str = persona_prompt_dir,
    ):
    for scenario in scenarios:
        for i in range(n_notes):  # 10 notes per scenario
            # Load gold note
            gold_note = load_prompt(f"{base_path}/gold_{scenario}_{i:03d}")
            
            for persona in personas:
                # Transform to persona
                note = transform_to_persona(tokenizer, model, gold_note, persona, persona_path)
                save_note(note, scenario=scenario, note_id=f"{persona}_{scenario}_{i:03d}.json", persona=persona)
            
            # # Transform to Chen  
            # chen_note = transform_to_persona(tokenizer, model, gold_note, persona="chen")
            # save_note(chen_note, scenario=scenario, note_id=f"chen_{scenario}_{i:03d}.json", persona="chen")
            
            clear_gpu_memory()
            
            print(f"{scenario}: Transformed {i+1}/{n_notes}")

def clean_label_note(generated_text):
    import re
    
    # Remove markdown code blocks
    generated_text = re.sub(r'```[\w]*\s*', '', generated_text)
    
    # Find first complete JSON object
    start = generated_text.find('{')
    if start == -1:
        return generated_text.strip()
    
    # Find matching closing brace
    count = 0
    for i in range(start, len(generated_text)):
        if generated_text[i] == '{':
            count += 1
        elif generated_text[i] == '}':
            count -= 1
            if count == 0:
                return generated_text[start:i+1]
    
    return generated_text.strip()

def generate_labels_for_persona(tokenizer, model, gold_note, transformed_note, persona, max_new_tokens):
    """Use MedGemma to compare and label quality issues"""
    
    prompt = f"""
Compare these two clinical notes and identify quality issues.

GOLD STANDARD (Complete):
{gold_note}

PHYSICIAN NOTE ({persona}):
{transformed_note}

Analyze the physician note and score these dimensions (0-100):
- Completeness: 90-100 (all elements), 70-89 (minor omissions), 50-69 (missing key elements), <50 (severely incomplete)
- Accuracy: 90-100 (all facts correct), 70-89 (minor errors), <70 (significant errors)
- Compliance: 90-100 (billing justified), 70-89 (minimal gaps), <70 (inadequate documentation)
- Risk: 90-100 (allergies, medications, return precautions all documented), 60-89 (missing 1-2 safety elements), <60 (missing 3+ safety elements)
- Clarity: 90-100 (clear and organized), 70-89 (adequate), <70 (confusing/disorganized)

Output JSON only:
{{
  "completeness": <score>,
  "accuracy": <score>,
  "compliance": <score>,
  "risk": <score>,
  "clarity": <score>,
  "issues": [
    {{"dimension": "completeness", "severity": "important", "description": "Missing patient education"}},
    {{"dimension": "compliance", "severity": "suggestion", "description": "No billing justification"}}
  ]
}}
"""
    
    output = generate_note(prompt, tokenizer, model, max_new_tokens=max_new_tokens)
    try:
        labels = json.loads(clean_label_note(output))
    except json.JSONDecodeError:
        print(f"Failed to parse: {output}")
        return None
    
    return labels

def run_generate_labels(
    tokenizer, 
    model,
    gold_prompt_path: str = gold_prompt_dir,
    persona_generated_path: str = persona_generated_dir,
    max_new_tokens: int = 1500,
    scenarios: list = scenarios,
    n_notes: int = 10,
    personas: list = personas,
):
    """Call the generate label function to run in batches"""
    for scenario in scenarios:
        for i in range(n_notes):
            seq = f"{i:03d}"
            for persona in personas:
                gold_file = f"{gold_prompt_path}/gold_{scenario}_{seq}"
                persona_file = f"{persona_generated_path}/{persona}_{scenario}_{seq}.json"
                gold_note = load_json_prompt(gold_file)["note"]
                persona_note = load_json_prompt(persona_file)["note"]
    
                # persona = persona_note.get("persona")
                labels = generate_labels_for_persona(tokenizer, model, gold_note, persona_note, persona, max_new_tokens)
                if labels:
                    labels["note_id"] = f"{persona}_{scenario}_{seq}"
                    save_note_with_labels(persona_note, labels, persona, scenario, labels["note_id"])
            clear_gpu_memory()
            
            print(f"{scenario}: Labeled {i+1}/{n_notes}")


if __name__ == "__main__":
    clear_gpu_memory()
    # Load model once
    tokenizer, model = load_medgemma(token=get_credentials())

    # run_scenarios(tokenizer, model, scenarios)
    # run_personas(tokenizer, model)
    # run_generate_labels(tokenizer, model)

