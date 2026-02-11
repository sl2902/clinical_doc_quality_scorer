model_name = "google/medgemma-4b-it"

scenarios = ["uri", "htn_followup", "t2dm", "back_pain", "annual_physical"]
personas = ["brooks", "chen"]
dimensions = ["completeness", "accuracy", "compliance", "risk", "clarity"]

data_dir = "data"
base_dir = "/kaggle/input/prompts"
gold_prompt_dir = "/kaggle/input/gold-prompts/generated_prompts"
persona_prompt_dir = "/kaggle/input/personas"
persona_generated_dir = "/kaggle/input/personas-generated/generated_personas"

# 50 gold; 50 brooks; 50 chen
max_notes = 150