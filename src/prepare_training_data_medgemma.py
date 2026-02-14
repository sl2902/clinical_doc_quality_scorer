import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "finetuning"

# Set seed for reproducibility
random.seed(42)

# Dimension prompts
DIMENSION_PROMPTS = {
    "completeness": "Score the completeness of this clinical note (0-100):\n\nCompleteness measures if all required sections are present: Chief Complaint, HPI, ROS, Physical Exam, Past Medical History, Medications, Allergies, Assessment, and Plan.",
    
    "accuracy": "Score the accuracy of this clinical note (0-100):\n\nAccuracy measures if medical facts, diagnoses, and clinical information are medically correct and appropriate.",
    
    "compliance": "Score the compliance of this clinical note (0-100):\n\nCompliance measures if adequate billing documentation is present, including time spent, complexity level, and appropriate ICD codes.",
    
    "risk": "Score the risk documentation of this clinical note (0-100):\n\nRisk measures if critical safety elements are documented: allergies, current medications, return precautions, and red flag warnings.",
    
    "clarity": "Score the clarity of this clinical note (0-100):\n\nClarity measures if the note is well-organized, clearly written, uses appropriate medical terminology, and is easy to understand."
}


def load_all_labeled_notes() -> List[Dict]:
    """Load all notes with ground_truth_scores"""
    
    personas = ["gold", "brooks", "chen", "minimal"]
    all_notes = []
    
    for persona in personas:
        persona_dir = DATA_DIR / persona
        
        if not persona_dir.exists():
            print(f"Warning: Directory not found: {persona_dir}")
            continue
        
        for filepath in sorted(persona_dir.glob("*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    
                    # Check if labeled
                    if "ground_truth_scores" not in data:
                        print(f"Warning: {filepath.name} missing ground_truth_scores, skipping")
                        continue
                    
                    all_notes.append(data)
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return all_notes


def split_train_test(notes: List[Dict], test_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
    """Split notes into train and test sets, stratified by persona AND scenario"""
    
    # Group by (persona, scenario)
    by_group = {}
    for note in notes:
        persona = note['persona']
        scenario = note['scenario']
        key = (persona, scenario)
        
        if key not in by_group:
            by_group[key] = []
        by_group[key].append(note)
    
    train_notes = []
    test_notes = []
    
    print("\nStratified split by persona and scenario:")
    print(f"{'Persona':<10} {'Scenario':<20} {'Train':<8} {'Test':<8}")
    print("-" * 50)
    
    # Split each group proportionally
    for (persona, scenario), group_notes in sorted(by_group.items()):
        # Shuffle
        shuffled = group_notes.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        n_test = int(len(shuffled) * test_ratio)
        if n_test == 0 and len(shuffled) > 0:
            n_test = 1  # Ensure at least 1 test example if group exists
        
        # Split
        test = shuffled[:n_test]
        train = shuffled[n_test:]
        
        test_notes.extend(test)
        train_notes.extend(train)
        
        print(f"{persona:<10} {scenario:<20} {len(train):<8} {len(test):<8}")
    
    return train_notes, test_notes


def create_training_example(note_text: str, dimension: str, score: int, note_id: str = None) -> Dict:
    """Create single training example in MedGemma format"""
    
    instruction = DIMENSION_PROMPTS[dimension]
    
    # Truncate note if too long (keep first 1000 chars to stay within context)
    if len(note_text) > 1000:
        note_text = note_text[:1000] + "..."
    
    # Format in Gemma chat format
    text = (
        f"<start_of_turn>user\n"
        f"{instruction}\n\n"
        f"{note_text}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{score}<end_of_turn>"
    )
    
    example = {"text": text, "note_id": note_id}
    
    # # Add note_id for test set (to track predictions)
    # if note_id:
        # example["note_id"] = note_id
    
    return example


def prepare_training_data():
    """Convert labeled notes to training format for each dimension"""
    
    print("Loading labeled notes...")
    notes = load_all_labeled_notes()
    print(f"Loaded {len(notes)} labeled notes\n")
    
    if len(notes) == 0:
        print("ERROR: No labeled notes found! Run label_all_notes.py first.")
        return
    
    # Count by persona
    persona_counts = {}
    for note in notes:
        persona = note['persona']
        persona_counts[persona] = persona_counts.get(persona, 0) + 1
    
    print("Breakdown by persona:")
    for persona, count in sorted(persona_counts.items()):
        print(f"  {persona}: {count} notes")
    
    # Split into train/test (80/20)
    print(f"\nSplitting data (80% train, 20% test)...")
    train_notes, test_notes = split_train_test(notes, test_ratio=0.2)
    
    print(f"\nTotal: {len(train_notes)} train, {len(test_notes)} test")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each dimension
    dimensions = ["completeness", "accuracy", "compliance", "risk", "clarity"]
    
    for dimension in dimensions:
        print(f"\n{'='*80}")
        print(f"Processing {dimension.upper()}")
        print(f"{'='*80}")
        
        # Create training examples
        train_examples = []
        for note in train_notes:
            note_text = note['note_text']
            score = note['ground_truth_scores'][dimension]
            note_id = note['note_id']
            example = create_training_example(note_text, dimension, score, note_id)
            train_examples.append(example)
        
        # Create test examples (with note_id for evaluation)
        test_examples = []
        for note in test_notes:
            note_text = note['note_text']
            score = note['ground_truth_scores'][dimension]
            note_id = note['note_id']
            example = create_training_example(note_text, dimension, score, note_id)
            test_examples.append(example)
        
        # Save train file
        train_file = OUTPUT_DIR / f"{dimension}_train.json"
        with open(train_file, 'w') as f:
            json.dump(train_examples, f, indent=2)
        
        # Save test file
        test_file = OUTPUT_DIR / f"{dimension}_test.json"
        with open(test_file, 'w') as f:
            json.dump(test_examples, f, indent=2)
        
        print(f"Created {len(train_examples)} train examples")
        print(f"Created {len(test_examples)} test examples")
        print(f"Saved to: {train_file.name}, {test_file.name}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total notes: {len(notes)}")
    print(f"Train split: {len(train_notes)} notes ({len(train_notes)/len(notes)*100:.1f}%)")
    print(f"Test split: {len(test_notes)} notes ({len(test_notes)/len(notes)*100:.1f}%)")
    print(f"Dimensions: {len(dimensions)}")
    print(f"Files created: {len(dimensions)*2} (train + test per dimension)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    print("Next step: Upload these files to Kaggle/Colab and fine-tune MedGemma agents")
    
    # Summary of files
    print("\nFiles created:")
    for dimension in dimensions:
        print(f"  {dimension}_train.json")
        print(f"  {dimension}_test.json")


if __name__ == "__main__":
    prepare_training_data()