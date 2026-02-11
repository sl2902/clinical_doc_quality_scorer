# prepare_training_data.py

import json
import random
import os
from datetime import datetime

from src.config import(
    data_dir,
    dimensions,
    max_notes,
    scenarios,
    personas,
)

def load_note(filepath):
    """Load a single note"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save JSON data"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def fix_inconsistent_labels(note):
    """Adjust scores based on issues to resolve conflicts"""
    for issue in note.get('quality_issues', []):
        dimension = issue.get('dimension')
        severity = issue.get('severity')

        if dimension is None or severity is None:
            # no change
            return note
        
        if dimension  in ["complaint", "compliancy"]:
            dimension = 'compliance'
        
        # Deduct points based on severity
        if severity in ['major', 'important']:
            note['ground_truth_scores'][dimension] = max(
                note['ground_truth_scores'][dimension] - 40, 0
            )
        elif severity in ['moderate']:
            note['ground_truth_scores'][dimension] = max(
                note['ground_truth_scores'][dimension] - 30, 0
            )
        elif severity in ['minor', 'suggestion']:
            note['ground_truth_scores'][dimension] = max(
                note['ground_truth_scores'][dimension] - 10, 0
            )
    
    return note

def load_all_notes(scenarios, personas):
    """Load all 150 labeled notes"""
    all_notes = []
    
    for persona in personas:
        persona_dir = f"{data_dir}/{persona}"
        
        for scenario in scenarios:
            for i in range(10):
                note_id = f"{persona}_{scenario}_{i:03d}"
                filepath = f"{persona_dir}/{note_id}.json"
                
                try:
                    note = load_note(filepath)
                    # Fix inconsistencies
                    note = fix_inconsistent_labels(note)
                    all_notes.append(note)
                    print(f"Loaded: {note_id}")
                except FileNotFoundError:
                    print(f"Missing: {filepath}")
    
    print(f"\nTotal notes loaded: {len(all_notes)}")
    return all_notes

def split_dataset(notes, train_ratio=0.8):
    """Split into train and test sets"""
    random.seed(42)
    random.shuffle(notes)
    
    split_idx = int(len(notes) * train_ratio)
    train_notes = notes[:split_idx]
    test_notes = notes[split_idx:]
    
    print(f"Train: {len(train_notes)}, Test: {len(test_notes)}")
    return train_notes, test_notes

def format_for_finetuning(data):
    """Convert note to Gemma chat format"""
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

def create_training_datasets(
        scenarios: list = scenarios, 
        personas: list = personas, 
        dimensions: list = dimensions
    ):
    """Main function to prepare all training data"""
    
    print("="*80)
    print("PREPARING TRAINING DATA")
    print("="*80)
    
    # Load all notes
    print("\n1. Loading notes...")
    all_notes = load_all_notes(scenarios, personas)
    
    if len(all_notes) != max_notes:
        print(f"\nWARNING: Expected {max_notes} notes, got {len(all_notes)}")
    
    # Split dataset
    print("\n2. Splitting dataset...")
    train_notes, test_notes = split_dataset(all_notes)
    
    # Save splits
    save_json(train_notes, f"{data_dir}/splits/train_split.json")
    save_json(test_notes, f"{data_dir}/splits/test_split.json")
    print("Saved train/test splits")
    
    # Format for each dimension
    print("\n3. Formatting for fine-tuning...")
    for dimension in dimensions:
        # Training data
        train_data = format_for_finetuning(train_notes)
        save_json(train_data, f"{data_dir}/finetuning/{dimension}_train.json")
        
        # Test data
        test_data = format_for_finetuning(test_notes)
        save_json(test_data, f"{data_dir}/finetuning/{dimension}_test.json")
        
        print(f"  {dimension}: {len(train_data)} train, {len(test_data)} test")
    
    # Summary
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - {data_dir}/splits/train_split.json ({len(train_notes)} notes)")
    print(f"  - {data_dir}/splits/test_split.json ({len(test_notes)} notes)")
    print(f"  - {data_dir}/finetuning/[dimension]_train.json (5 files)")
    print(f"  - {data_dir}/finetuning/[dimension]_test.json (5 files)")
    
    return train_notes, test_notes

if __name__ == "__main__":
    train_notes, test_notes = create_training_datasets()