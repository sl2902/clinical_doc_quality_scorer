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
    data_dir_gold,
    scenarios,
    personas,
    MAX_REQUESTS_PER_MINUTE,
    MAX_TOKENS_PER_MINUTE,
    SEMAPHORE_SIZE,
)

# Setup
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
PROMPTS_DIR = SCRIPT_DIR / "personas"
DATA_DIR = SCRIPT_DIR / "data"

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tracking
total_input_tokens = 0
total_output_tokens = 0

# Rate limiting
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE)
semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)


PERSONA_SYSTEM_PROMPTS = {
    "brooks": "You are Dr. Marcus Brooks, a busy physician who documents quickly (6-10 minutes per note) with minimal detail. You skip non-essential information while capturing key clinical facts.",
    
    "chen": "You are Dr. Lisa Chen, whose documentation quality varies based on time pressure. Some sections are thorough, others are rushed. Your notes are inconsistent in completeness and detail.",
    
    "minimal": "You are a resident documenting during a mass casualty incident at 3 AM. You have 2 minutes per patient. Your notes are extremely brief and would be flagged for quality review."
}


def check_if_file_exists(persona: str, note_id: str) -> bool:
    """Check if persona note already exists"""
    filepath = DATA_DIR / persona / f"{note_id}.json"
    
    if filepath.exists():
        print(f"  {note_id} already exists, skipping")
        return True
    return False


def load_all_gold_notes():
    """Load all gold notes from disk"""
    gold_dir = DATA_DIR / "gold"
    
    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold notes directory not found: {gold_dir}")
    
    gold_notes = []
    
    for filepath in sorted(gold_dir.glob("*.json")):
        with open(filepath) as f:
            gold_notes.append(json.load(f))
    
    return gold_notes


async def transform_to_persona_gpt4_async(gold_note_text: str, persona_type: str, 
                                          demographics: dict, scenario: str, note_num: int):
    """Transform one gold note to persona style"""
    
    global total_input_tokens, total_output_tokens
    
    async with semaphore:
        await rate_limiter.wait_if_needed(estimated_tokens=2000)
        
        note_id = f"{persona_type}_{scenario}_{note_num:03d}"
        
        # Check if already exists
        if check_if_file_exists(persona_type, note_id):
            filepath = DATA_DIR / persona_type / f"{note_id}.json"
            with open(filepath) as f:
                return json.load(f)
        
        # Load transformation prompt
        prompt_file = PROMPTS_DIR / f"{persona_type}_transform_prompt.txt"
        
        if not prompt_file.exists():
            print(f" Prompt file not found: {prompt_file}")
            return None
        
        prompt_template = prompt_file.read_text()
        prompt = prompt_template.replace("{gold_note}", gold_note_text)
        
        # Retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name_gpt,
                    messages=[
                        {
                            "role": "system",
                            "content": PERSONA_SYSTEM_PROMPTS[persona_type]
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.6,
                    max_tokens=1500,
                    timeout=60.0
                )
                
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                
                # Prepend demographics header
                demo_header = f"**Patient:** {demographics['name']}\n**Age:** {demographics['age']}\n**Gender:** {demographics['gender']}\n**Occupation:** {demographics['occupation']}\n\n"
                note_text = demo_header + response.choices[0].message.content
                
                # Prepare note data
                note_data = {
                    "note_id": note_id,
                    "scenario": scenario,
                    "persona": persona_type,
                    "note_text": note_text,
                    "demographics": demographics,
                    "source_gold_note": f"gold_{scenario}_{note_num:03d}",
                    "generated_at": datetime.now().isoformat()
                }
                
                # Save
                save_dir = DATA_DIR / persona_type
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / f"{note_id}.json"
                async with aiofiles.open(save_path, 'w') as f:
                    await f.write(json.dumps(note_data, indent=2))
                
                print(f" {note_id}")
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


async def generate_all_persona_notes(personas: list = personas):
    """Transform all gold notes to personas"""
    
    # Load gold notes
    print("Loading gold notes...")
    gold_notes = load_all_gold_notes()
    print(f"Loaded {len(gold_notes)} gold notes\n")
    
    # Prepare all transformation tasks
    tasks = []
    
    for gold_note in gold_notes:
        scenario = gold_note['scenario']
        demographics = gold_note['demographics']
        gold_text = gold_note['note_text']
        note_num = int(gold_note['note_id'].split('_')[-1])
        
        for persona in personas:
            # Check if already exists before creating task
            note_id = f"{persona}_{scenario}_{note_num:03d}"
            
            if not check_if_file_exists(persona, note_id):
                task = transform_to_persona_gpt4_async(
                    gold_text, persona, demographics, scenario, note_num
                )
                tasks.append((persona, scenario, note_num, task))
    
    if not tasks:
        print(" All persona notes already exist!")
        return []
    
    print(f"{'='*80}")
    print(f"PERSONA NOTE GENERATION")
    print(f"{'='*80}")
    print(f"Total notes to generate: {len(tasks)}")
    print(f"Estimated time: {len(tasks) * 3 / 60:.1f} minutes")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Execute all tasks
    results = await asyncio.gather(*[task for _, _, _, task in tasks], return_exceptions=True)
    
    # Separate successes and failures
    persona_notes = []
    failures = []
    
    for (persona, scenario, note_num, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            failures.append(f"{persona}_{scenario}_{note_num:03d}: {str(result)[:50]}")
        elif result is not None:
            persona_notes.append(result)
    
    elapsed = time.time() - start_time
    cost = (total_input_tokens * 0.01 + total_output_tokens * 0.03) / 1000
    
    print(f"\n{'='*80}")
    print(f" PERSONA NOTES COMPLETE")
    print(f"{'='*80}")
    print(f"Generated: {len(persona_notes)} new notes")
    print(f"Failed: {len(failures)} notes")
    if failures:
        print(f"\nFailed notes:")
        for failure in failures[:10]:  # Show first 10
            print(f"  - {failure}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")
    print(f"\nTime: {elapsed/60:.1f} minutes")
    print(f"Cost: ${cost:.2f}")
    print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
    print(f"{'='*80}\n")
    
    if failures:
        print("Rerun script to retry failed notes (existing notes will be skipped)")
    
    return persona_notes


if __name__ == "__main__":
    asyncio.run(generate_all_persona_notes())