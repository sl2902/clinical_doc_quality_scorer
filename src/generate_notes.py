"""Generate Clinical notes from scenarios"""
import os
import json
from datetime import datetime
from typing import Any


DATA_DIR = "data"
SCENARIOS = ["uri", "htn_followup", "t2dm", "back_pain", "annual_physical"]
DIMENSIONS = ["completeness", "accuracy", "compliance", "risk", "clarity"]

def load_json_note(prompt_file: str):
    with open(prompt_file, 'r') as f:
        prompt = json.load(f)
    return prompt

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
    
    filename = f"data/gold/{note_id}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def save_json(json_note: list[dict[str], Any], filename: str):

    with open(filename, 'w') as f:
        json.dump(json_note, f, indent=2)

def fix_inconsistent_labels(note):
    """Adjust scores based on issues"""
    for issue in note['quality_issues']:
        dimension = issue.get('dimension')
        severity = issue.get('severity')

        if dimension is None or severity is None:
            # no change
            return note
        
        if dimension  == "complaint":
            dimension = 'compliance'
        
        # Deduct points based on severity
        if severity == 'major' or severity == "important":
            note['ground_truth_scores'][dimension] -= 40
        elif severity == 'moderate':
            note['ground_truth_scores'][dimension] -= 30
        else:
            note['ground_truth_scores'][dimension] -= 10
        
    
    return note
