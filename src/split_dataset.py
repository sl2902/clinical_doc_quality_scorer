import random
import os

from generate_notes import (
    DATA_DIR,
    DIMENSIONS,
    SCENARIOS, 
    load_json_note, 
    save_json, 
    fix_inconsistent_labels
)


def creat_train_test_split():
    all_notes = []
    for persona in ["gold", "brooks", "chen"]:
        for scenario in SCENARIOS:
            for i in range(10):
                note = load_json_note(f"data/{persona}/{persona}_{scenario}_{i:03d}.json")
                all_notes.append(note)

    # Shuffle and split (80/20)
    random.shuffle(all_notes)
    train_notes = all_notes[:120]
    test_notes = all_notes[120:]

    # Save splits
    save_json(train_notes, f"{DATA_DIR}/train_split.json")
    save_json(test_notes, f"{DATA_DIR}/test_split.json")

def adjust_inconsistent_label_scores(filename):

    notes = load_json_note(filename)

    fixed_note = [
        fix_inconsistent_labels(note)
        for note in notes
    ]
    save_json(fixed_note, filename)

def format_for_finetuning(notes, dimension):
    """Convert to instruction-response pairs for fine-tuning"""
    dataset = []
    
    for note in notes:
        instruction = f"Score the {dimension} of this clinical note (0-100):\n\n{note['note_text']}"
        output = str(note['ground_truth_scores'][dimension])
        
        dataset.append({
            "instruction": instruction,
            "output": output,
            "note_id": note['note_id'],
        })
    
    return dataset

# train_data = format_for_finetuning(train_notes, "completeness")
# save_json(train_data, "data/completeness_train.json")

# creat_train_test_split()
# adjust_inconsistent_label_scores(f"{DATA_DIR}/train_split.json")
# adjust_inconsistent_label_scores(f"{DATA_DIR}/test_split.json")
train_notes = load_json_note(f"{DATA_DIR}/splits/train_split.json")
test_notes = load_json_note(f"{DATA_DIR}/splits/test_split.json")
for dim in DIMENSIONS:
    filename = f"{DATA_DIR}/finetuning/{dim}_train.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dataset = format_for_finetuning(train_notes, dim)
    save_json(dataset, f"{DATA_DIR}/finetuning/{dim}_train.json")

    dataset = format_for_finetuning(test_notes, dim)
    save_json(dataset, f"{DATA_DIR}/finetuning/{dim}_test.json")