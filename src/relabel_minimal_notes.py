# relabel_minimal_notes.py

import asyncio
import json
from pathlib import Path
from openai import AsyncOpenAI
import os

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data" / "minimal"

async def relabel_minimal_note(filepath):
    """Relabel with STRICT scoring prompt"""
    
    with open(filepath) as f:
        data = json.load(f)
    
    note_text = data['note_text']
    
    # STRICT prompt for minimal notes
    prompt = f"""Score this INCOMPLETE clinical note. This note is from a rushed emergency scenario and is severely inadequate.

Be STRICT. Minimal notes should score 15-45, NOT 60-80.

Missing sections = LOW completeness (20-40)
Missing safety elements = LOW risk (20-40)  
Missing billing = LOW compliance (15-30)

Score on 5 dimensions (0-100):
- Completeness: Sections present vs missing
- Accuracy: Medical facts correct (can still be high even if incomplete)
- Compliance: Billing documentation present
- Risk: Safety elements (allergies, meds, return precautions)
- Clarity: Organization and readability

This is an INCOMPLETE minimal note. Score strictly.

Note:
{note_text[:1500]}

JSON only:
{{
  "completeness": <score>,
  "accuracy": <score>,
  "compliance": <score>,
  "risk": <score>,
  "clarity": <score>
}}"""
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a STRICT clinical documentation auditor. Incomplete notes get low scores (15-45 range). Be harsh on missing elements."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=150,
        response_format={"type": "json_object"}
    )
    
    scores = json.loads(response.choices[0].message.content)
    
    # Update and save
    data['ground_truth_scores'] = scores
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"{filepath.name}: {scores}")
    return scores

async def relabel_all_minimal():
    tasks = []
    
    for filepath in sorted(DATA_DIR.glob("*.json")):
        tasks.append(relabel_minimal_note(filepath))
    
    results = await asyncio.gather(*tasks)
    
    # Check new distribution
    import numpy as np
    avg_scores = [sum(s.values())/len(s) for s in results]
    print(f"\nNew minimal average: {np.mean(avg_scores):.1f}")
    print(f"Range: {min(avg_scores):.1f} - {max(avg_scores):.1f}")

if __name__ == "__main__":
    asyncio.run(relabel_all_minimal())