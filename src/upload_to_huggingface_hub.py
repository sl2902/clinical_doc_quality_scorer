from huggingface_hub import login, upload_folder, create_repo
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv('hf_token')

login(token=hf_token)

BASE_MODEL = "google/medgemma-4b-it"

dimensions = ["completeness", "accuracy", "compliance", "risk", "clarity"]

for dim in dimensions:
    print(f"Uploading {dim}...")

    repo_id = f"sl02/clinical-doc-{dim}-agent"

    try:
        create_repo(repo_id, token=hf_token, repo_type="model", exist_ok=True)
        print(f" Created repo: {repo_id}")
    except Exception as e:
        print(f"Repo might already exist: {e}")
    
    upload_folder(
        folder_path=f"model_weights/{dim}/models/{dim}_agent",
        repo_id=repo_id, 
        token=hf_token,
        repo_type="model",
        create_pr=False,
    )
    
    print(f" Done: https://huggingface.co/{repo_id}")

print("\nAll agents uploaded!")