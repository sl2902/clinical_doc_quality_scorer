model_name = "google/medgemma-4b-it"

scenarios = ["uri", "htn_followup", "t2dm", "back_pain", "annual_physical"]
personas = ["brooks", "chen"]

base_dir = "/kaggle/input/prompts/"
gold_prompt_dir = "/kaggle/input/gold-prompts/generated_prompts"
persona_prompt_dir = "/kaggle/input/personas"
persona_generated_dir = "/kaggle/input/personas-generated/generated_personas"