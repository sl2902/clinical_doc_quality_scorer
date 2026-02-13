import asyncio
import aiofiles
from openai import AsyncOpenAI
import json
import time
from datetime import datetime
import os
from pathlib import Path

from rate_limiter import RateLimiter
from config import (
    model_name_gpt,
    MAX_REQUESTS_PER_MINUTE,
    MAX_TOKENS_PER_MINUTE,
    SEMAPHORE_SIZE,
)

# Setup
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tracking
total_input_tokens = 0
total_output_tokens = 0

# Rate limiting
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE)
semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)

# Personas to label
PERSONAS = ["gold", "brooks", "chen", "minimal"]

# System prompt for labeling
LABELING_SYSTEM_PROMPT = """You are a clinical documentation quality auditor. You objectively score clinical notes on 5 dimensions (completeness, accuracy, compliance, risk, clarity) using a 0-100 scale.

You provide scores as JSON only, with no explanations or preamble."""


def check_if_already_labeled(filepath: Path) -> bool:
    """Check if note already has ground_truth_scores"""
    try:
        with open(filepath) as f:
            data = json.load(f)
            if "ground_truth_scores" in data:
                print(f"  {filepath.name} already labeled, skipping")
                return True
    except Exception as e:
        print(f" Error reading {filepath}: {e}")
    
    return False


def load_all_notes():
    """Load all notes from all persona directories"""
    all_notes = []
    
    for persona in PERSONAS:
        persona_dir = DATA_DIR / persona
        
        if not persona_dir.exists():
            print(f" Directory not found: {persona_dir}")
            continue
        
        for filepath in sorted(persona_dir.glob("*.json")):
            # Skip if already labeled
            if not check_if_already_labeled(filepath):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                        data['_filepath'] = filepath  # Store path for later
                        all_notes.append(data)
                except Exception as e:
                    print(f" Error loading {filepath}: {e}")
    
    return all_notes


async def label_note_gpt4_async(note_data: dict):
    """Label one note with quality scores"""
    
    global total_input_tokens, total_output_tokens
    
    async with semaphore:
        await rate_limiter.wait_if_needed(estimated_tokens=500)
        
        filepath = note_data['_filepath']
        note_id = note_data['note_id']
        persona = note_data['persona']
        note_text = note_data['note_text']
        
        # Create labeling prompt
        prompt = f"""Score this clinical note on 5 dimensions (0-100):

**Dimensions:**
- **Completeness**: All required elements present (CC, HPI, ROS, PE, PMH, Meds, Allergies, Assessment, Plan)
- **Accuracy**: Medical facts and diagnoses are correct
- **Compliance**: Adequate billing documentation (time spent, complexity, ICD codes)
- **Risk**: Safety elements documented (allergies, medications, return precautions, red flags)
- **Clarity**: Clear, well-organized, professional documentation

**Expected score ranges by note type:**
- **gold** (complete notes): 88-98
- **brooks** (rushed notes): 50-75
- **chen** (variable notes): 70-88
- **minimal** (incomplete notes): 15-45

**This is a {persona} note.**

**Clinical note to score:**
{note_text[:1500]}

Respond with JSON only (no other text):
{{
  "completeness": <score>,
  "accuracy": <score>,
  "compliance": <score>,
  "risk": <score>,
  "clarity": <score>
}}"""
        
        # Retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name_gpt,
                    messages=[
                        {
                            "role": "system",
                            "content": LABELING_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=150,
                    response_format={"type": "json_object"},
                    timeout=60.0
                )
                
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                
                # Parse scores
                scores = json.loads(response.choices[0].message.content)
                
                # Validate scores
                required_keys = ["completeness", "accuracy", "compliance", "risk", "clarity"]
                if not all(key in scores for key in required_keys):
                    raise ValueError(f"Missing required score keys: {required_keys}")
                
                # Add scores to note data
                note_data['ground_truth_scores'] = scores
                
                # Remove temporary filepath before saving
                del note_data['_filepath']
                
                # Save back to file
                async with aiofiles.open(filepath, 'w') as f:
                    await f.write(json.dumps(note_data, indent=2))
                
                print(f" {note_id}: {scores}")
                return note_data
                
            except Exception as e:
                error_msg = str(e)
                
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = (2 ** attempt) * 5
                    print(f" Rate limit on {note_id}, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f" Error on {note_id}, retrying in {wait_time}s: {error_msg[:100]}")
                    await asyncio.sleep(wait_time)
                else:
                    print(f" Failed {note_id} after {max_retries} attempts: {error_msg[:100]}")
                    return None


async def label_all_notes():
    """Label all unlabeled notes"""
    
    # Load all unlabeled notes
    print("Loading unlabeled notes...")
    notes_to_label = load_all_notes()
    
    if not notes_to_label:
        print(" All notes already labeled!")
        return []
    
    print(f"Found {len(notes_to_label)} unlabeled notes\n")
    
    # Count by persona
    persona_counts = {}
    for note in notes_to_label:
        persona = note['persona']
        persona_counts[persona] = persona_counts.get(persona, 0) + 1
    
    print("Breakdown by persona:")
    for persona, count in sorted(persona_counts.items()):
        print(f"  {persona}: {count} notes")
    
    print(f"\n{'='*80}")
    print(f"LABELING CLINICAL NOTES")
    print(f"{'='*80}")
    print(f"Total notes to label: {len(notes_to_label)}")
    print(f"Estimated time: {len(notes_to_label) * 2 / 60:.1f} minutes")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Create tasks
    tasks = [label_note_gpt4_async(note) for note in notes_to_label]
    
    # Execute all tasks
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Separate successes and failures
    labeled_notes = []
    failures = []
    
    for note, result in zip(notes_to_label, results):
        if isinstance(result, Exception):
            failures.append(f"{note['note_id']}: {str(result)[:50]}")
        elif result is not None:
            labeled_notes.append(result)
    
    elapsed = time.time() - start_time
    cost = (total_input_tokens * 0.01 + total_output_tokens * 0.03) / 1000
    
    print(f"\n{'='*80}")
    print(f" LABELING COMPLETE")
    print(f"{'='*80}")
    print(f"Labeled: {len(labeled_notes)} notes")
    print(f"Failed: {len(failures)} notes")
    if failures:
        print(f"\nFailed notes:")
        for failure in failures[:10]:
            print(f"  - {failure}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")
    print(f"\nTime: {elapsed/60:.1f} minutes")
    print(f"Cost: ${cost:.2f}")
    print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
    print(f"{'='*80}\n")
    
    if failures:
        print("Rerun script to retry failed notes (already labeled notes will be skipped)")
    
    return labeled_notes


if __name__ == "__main__":
    asyncio.run(label_all_notes())