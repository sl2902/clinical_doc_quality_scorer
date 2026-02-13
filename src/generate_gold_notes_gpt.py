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
    demographics_male,
    demographics_female,
    gender_content,
    scenarios,
    model_name_gpt,
    data_dir_gold,
    scenario_variations,
    MAX_REQUESTS_PER_MINUTE,
    MAX_TOKENS_PER_MINUTE,
    SEMAPHORE_SIZE,
    BATCH_DELAY,
    BATCH_SIZE,
)

SCRIPT_DIR = Path(__file__).parent.parent.absolute()
PROMPTS_DIR = SCRIPT_DIR / "prompts"

# Initialize
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Tracking
total_input_tokens = 0
total_output_tokens = 0


rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE)
semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)

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

async def generate_gold_note_gpt4_async(scenario, demographics, note_num):
    global total_input_tokens, total_output_tokens

    prompt_file = PROMPTS_DIR / f"{scenario}_prompt.txt"
    
    async with semaphore:
        await rate_limiter.wait_if_needed(estimated_tokens=2500)
        
        prompt_template = prompt_file.read_text()
        prompt = inject_demographics_and_gender_content(
            prompt_template, 
            demographics, 
            scenario
        )

        variation_index = note_num % 5
        variation = scenario_variations[scenario][variation_index]
        
        variation_text = "\n\nCLINICAL VARIATION FOR THIS NOTE:\n"
        for key, value in variation.items():
            variation_text += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        prompt = prompt + variation_text
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are Dr. Elizabeth Martinez, a board-certified family medicine physician with 20 years of experience. You create thorough, well-structured clinical documentation following best practices."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=2000,
                        timeout=60.0,
                    )
                
                total_input_tokens += response.usage.prompt_tokens
                total_output_tokens += response.usage.completion_tokens
                
                note_text = response.choices[0].message.content
                note_id = f"gold_{scenario}_{note_num:03d}"
                
                note_data = {
                    "note_id": note_id,
                    "scenario": scenario,
                    "persona": "gold",
                    "note_text": note_text,
                    "demographics": demographics,
                    "generated_at": datetime.now().isoformat()
                }
                
                os.makedirs(f"{data_dir_gold}", exist_ok=True)
                async with aiofiles.open(f"{data_dir_gold}/{note_id}.json", 'w') as f:
                    await f.write(json.dumps(note_data, indent=2))
                
                print(f" {note_id} ({demographics['name']}, {demographics['age']}{demographics['gender']})")
                return note_data
                
            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    wait_time = (2 ** attempt) * 5  # Longer backoff for rate limits
                    print(f" Rate limit on {note_id}, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f" Error on {note_id}, retrying in {wait_time}s: {error_msg[:100]}")
                    await asyncio.sleep(wait_time)
                else:
                    print(f" Failed {note_id} after {max_retries} attempts: {error_msg[:100]}")
                    return None

def check_if_file_exists(note_id: str) -> bool:
    """Checks whether file exists or not"""

    if os.path.isfile(f"{data_dir_gold}/{note_id}.json"):
        print(f"File {note_id} exists. Skipping request")
        return True
    return False


# async def generate_all_gold_notes(scenarios: list = scenarios, n_notes: int = 5):
#     tasks = []
    
#     for scenario in scenarios:
#         print(f"\n{'='*80}")
#         print(f"QUEUING: {scenario}")
#         print(f"{'='*80}")
        
#         # 5 male notes
#         for note_num in range(n_notes):
#             note_id = f"gold_{scenario}_{note_num:03d}"

#             if not check_if_file_exists(note_id):
#                 task = generate_gold_note_gpt4_async(
#                     scenario, 
#                     demographics_male[i], 
#                     note_num
#                 )
#                 tasks.append(task)
        
#         # 5 female notes
#         for i in range(n_notes):
#             note_num = n_notes + i
#             note_id = f"gold_{scenario}_{note_num:03d}"

#             if not check_if_file_exists(note_id):
#                 task = generate_gold_note_gpt4_async(
#                     scenario, 
#                     demographics_female[i], 
#                     note_num
#                 )
#                 tasks.append(task)
    
#     print(f"\n{'='*80}")
#     print(f" Starting generation of {len(tasks)} gold notes...")
#     print(f"{'='*80}\n")
    
#     start_time = time.time()
#     results = await asyncio.gather(*tasks)
#     gold_notes = [r for r in results if r is not None]
    
#     elapsed = time.time() - start_time
#     cost = (total_input_tokens * 0.01 + total_output_tokens * 0.03) / 1000
    
#     print(f"\n{'='*80}")
#     print(f" GOLD NOTES COMPLETE")
#     print(f"{'='*80}")
#     print(f"Generated: {len(gold_notes)}/{len(tasks)} notes")
#     print(f"Time: {elapsed/60:.1f} minutes")
#     print(f"Cost: ${cost:.2f}")
#     print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
#     print(f"{'='*80}\n")
    
#     return gold_notes

async def generate_all_gold_notes_batched():
    """Process in batches to avoid rate limits"""
    
    # Prepare all requests
    all_requests = []
    for scenario in scenarios:
        note_counter = 0
        for i in range(5):  # Male
            all_requests.append((scenario, demographics_male[i], note_counter))
            note_counter += 1
        for i in range(5):  # Female
            all_requests.append((scenario, demographics_female[i], note_counter))
            note_counter += 1
    
    print(f"{'='*80}")
    print(f"GOLD NOTE GENERATION")
    print(f"{'='*80}")
    print(f"Total notes to generate: {len(all_requests)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated time: {len(all_requests) * 4 / 60:.1f} minutes")
    print(f"{'='*80}\n")
    
    all_results = []
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(all_requests), BATCH_SIZE):
        batch = all_requests[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(all_requests) - 1) // BATCH_SIZE + 1
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_num}/{total_batches} - Notes {i+1} to {min(i+BATCH_SIZE, len(all_requests))}")
        print(f"{'='*80}")
        
        # Create tasks for this batch
        tasks = [
            generate_gold_note_gpt4_async(scenario, demo, num)
            for scenario, demo, num in batch
        ]
        
        # Process batch
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results.extend(batch_results)
        
        # Count successes
        successful = sum(1 for r in batch_results if r is not None and not isinstance(r, Exception))
        failed = len(batch_results) - successful
        
        print(f"\nBatch {batch_num} complete: {successful}/{len(batch)} successful")
        if failed > 0:
            print(f" {failed} failed - will retry at end")
        
        # Wait between batches (except last batch)
        if i + BATCH_SIZE < len(all_requests):
            print(f"Waiting {BATCH_DELAY}s before next batch...")
            await asyncio.sleep(BATCH_DELAY)
    
    # Filter results
    gold_notes = [r for r in all_results if r is not None and not isinstance(r, Exception)]
    failed_count = len(all_results) - len(gold_notes)
    
    elapsed = time.time() - start_time
    cost = (total_input_tokens * 0.01 + total_output_tokens * 0.03) / 1000
    
    print(f"\n{'='*80}")
    print(f" GOLD NOTES COMPLETE")
    print(f"{'='*80}")
    print(f"Generated: {len(gold_notes)}/{len(all_requests)} notes")
    if failed_count > 0:
        print(f" Failed: {failed_count} notes")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Cost: ${cost:.2f}")
    print(f"Tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
    print(f"{'='*80}\n")
    
    # Show which ones failed
    if failed_count > 0:
        print("Failed notes:")
        for req, result in zip(all_requests, all_results):
            if result is None or isinstance(result, Exception):
                scenario, demo, num = req
                print(f"  - gold_{scenario}_{num:03d}")
        print("\nRerun script to retry failed notes (existing notes will be skipped)")
    
    return gold_notes

if __name__ == "__main__":
    asyncio.run(generate_all_gold_notes_batched())