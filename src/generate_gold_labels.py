"""Generate labels for gold dataset"""

from generate_notes import SCENARIOS, load_json_note, save_note_with_labels

def label_gold_notes(n_files: int = 10, base_path: str = "/"):
    """Assign high scores to gold standard notes"""
    for scenario in SCENARIOS:
        for i in range(n_files):
            note_id = f"{base_path}/gold_{scenario}_{i:03d}"
            note = load_json_note(note_id)["note"]
            
            labels = {
                "completeness": 95,
                "accuracy": 95,
                "compliance": 90,
                "risk": 95,
                "clarity": 90,
                "issues": []
            }
            save_note_with_labels(note, labels, "gold", scenario, note_id.split("/")[-1])

base_path = "generated_prompts"
label_gold_notes(base_path=base_path)
