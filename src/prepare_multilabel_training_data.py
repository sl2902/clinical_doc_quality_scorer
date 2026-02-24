def create_multilabel_example(note_text: str, scores: dict) -> dict:
    """Convert scores to multi-label classification format"""
    
    # Convert each score to a bin (0-4)
    def score_to_class(score):
        if score < 20: return 0      # very_low
        elif score < 40: return 1    # low  
        elif score < 60: return 2    # medium
        elif score < 80: return 3    # high
        else: return 4               # very_high
    
    labels = {
        'completeness': score_to_class(scores['completeness']),
        'accuracy': score_to_class(scores['accuracy']),
        'compliance': score_to_class(scores['compliance']),
        'risk': score_to_class(scores['risk']),
        'clarity': score_to_class(scores['clarity'])
    }
    
    instruction = """Classify this clinical note on 5 quality dimensions. For each dimension, respond with one number (0-4):

0 = very_low (0-19)
1 = low (20-39) 
2 = medium (40-59)
3 = high (60-79)
4 = very_high (80-100)

Dimensions:
- completeness: Completeness measures if all required sections are present: Chief Complaint, HPI, ROS, Physical Exam, Past Medical History, Medications, Allergies, Assessment, and Plan.
- accuracy: Accuracy measures if medical facts, diagnoses, and clinical information are medically correct and appropriate.
- compliance: Compliance measures if adequate billing documentation is present, including time spent, complexity level, and appropriate ICD codes.  
- risk: Risk measures if critical safety elements are documented: allergies, current medications, return precautions, and red flag warnings.
- clarity: Clarity measures if the note is well-organized, clearly written, uses appropriate medical terminology, and is easy to understand.

Respond in this exact format:
completeness: [0-4]
accuracy: [0-4]
compliance: [0-4]
risk: [0-4]
clarity: [0-4]"""
    
    output = f"""completeness: {labels['completeness']}
accuracy: {labels['accuracy']}
compliance: {labels['compliance']}
risk: {labels['risk']}
clarity: {labels['clarity']}"""
    
    text = (
        f"<start_of_turn>user\n"
        f"{instruction}\n\n"
        f"{note_text[:1000]}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{output}<end_of_turn>"
    )
    
    return {"text": text}


def prepare_multilabel_training_data():
    """Generate single training file for multi-label model"""
    
    from pathlib import Path
    import json
    import random
    
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "data" / "finetuning"
    
    # Load all labeled notes
    all_notes = []
    personas = ["gold", "brooks", "chen", "minimal"]
    
    for persona in personas:
        persona_dir = DATA_DIR / persona
        for filepath in sorted(persona_dir.glob("*.json")):
            with open(filepath) as f:
                data = json.load(f)
                if "ground_truth_scores" in data:
                    all_notes.append(data)
    
    # Stratified split by persona + scenario
    by_group = {}
    for note in all_notes:
        key = (note['persona'], note['scenario'])
        if key not in by_group:
            by_group[key] = []
        by_group[key].append(note)
    
    train_notes = []
    test_notes = []
    
    random.seed(42)
    for group_notes in by_group.values():
        shuffled = group_notes.copy()
        random.shuffle(shuffled)
        
        n_test = max(1, int(len(shuffled) * 0.2))
        test = shuffled[:n_test]
        train = shuffled[n_test:]
        
        test_notes.extend(test)
        train_notes.extend(train)
    
    # Create training examples
    train_examples = []
    for note in train_notes:
        example = create_multilabel_example(
            note['note_text'], 
            note['ground_truth_scores']
        )
        train_examples.append(example)
    
    test_examples = []
    for note in test_notes:
        example = create_multilabel_example(
            note['note_text'],
            note['ground_truth_scores']
        )
        example['note_id'] = note['note_id']  # For evaluation
        test_examples.append(example)
    
    # Save files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_DIR / "multilabel_train.json", 'w') as f:
        json.dump(train_examples, f, indent=2)
    
    with open(OUTPUT_DIR / "multilabel_test.json", 'w') as f:
        json.dump(test_examples, f, indent=2)
    
    print(f"Created multilabel training data:")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")
    print(f"  Files: multilabel_train.json, multilabel_test.json")
    
    return train_examples, test_examples

if __name__ == "__main__":
    prepare_multilabel_training_data()