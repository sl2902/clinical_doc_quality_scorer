# Clinical Documentation Quality Scorer

Multi-agent MedGemma system for real-time clinical note quality assessment across 5 dimensions.

## 🚀 Quick Demo

### Prerequisites
- **ngrok account**: Sign up at [ngrok.com](https://ngrok.com) and get your auth token
- **HuggingFace token**: Create token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Demo Setup
1. **Install ngrok**:
   ```bash
   # Download from ngrok.com or:
   pip install pyngrok
   
   # Set your ngrok token on Colab secrets
   ```

2. **Run Demo**:
   ```bash
   # Open notebooks/06-demo.ipynb in Google Colab
   # Run all the cells sequentially.
   # If successful, the demo will create a public URL via ngrok
   # Set your HF_TOKEN in the Streamlit configuration box
   # Explore the app
   ```

3. **Access**: Demo creates a public URL for testing the multi-agent scoring system

## 📊 System Architecture

### 5 Specialized MedGemma Agents
Each agent scores clinical notes (0-100) across:

| Dimension | Measures |
|-----------|----------|
| **Completeness** | All required sections present (CC, HPI, ROS, PE, Assessment, Plan) |
| **Accuracy** | Medical facts and diagnoses clinically appropriate |
| **Compliance** | Billing documentation adequate (time, ICD codes, complexity) |
| **Risk** | Safety elements documented (allergies, medications, precautions) |
| **Clarity** | Organization, terminology, professional presentation |

### Model Details
- **Base Model**: MedGemma-4B-it
- **Training**: QLoRA fine-tuning with 4-bit quantization
- **Data**: 200 synthetic clinical notes across quality levels
- **Available at**: https://huggingface.co/sl02/clinical-doc-{dimension}-agent

## 🔬 Development Workflow

### For Developers: Full Replication Steps

#### 1. Data Generation (Local with GPT-4)
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Generate synthetic clinical notes
python src/generate_gold_notes_gpt.py      # 50 complete notes
python src/generate_persona_notes_gpt.py   # 150 quality-degraded notes
python src/generate_labeled_notes_gpt.py   # GPT-4 labeling across 5 dimensions
```

**Requirements**: OpenAI API access (~$15-20 for full dataset generation)

#### 2. Training Data Preparation
```bash
python src/prepare_training_data_medgemma.py
# Creates 80/20 train/test splits for each dimension
# Output: data/finetuning/{dimension}_train.json, {dimension}_test.json
```

#### 3. Model Fine-tuning (Kaggle)
- **Notebook**: `notebooks/05-finetune-agents.ipynb`
- **Platform**: Kaggle (GPU required)
- **Process**: QLoRA fine-tuning of MedGemma-4B-it
- **Output**: 5 LoRA adapters (one per dimension)

```python
# Key fine-tuning config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

#### 4. Orchestrator Setup
```python
from src.orchestrator import ClinicalQualityOrchestrator

orchestrator = ClinicalQualityOrchestrator(hf_token="your-token")
scores = orchestrator.score_note(clinical_note_text)
```

## 📈 Performance Results

### Validation Metrics
- **MAE**: 16.5 points (on 0-100 scale)
- **Accuracy**: 85% predictions within 20 points of ground truth
- **Input Validation**: 100% rejection of non-clinical text

### Technical Challenges
- **Training Data Scale**: 200 examples insufficient for consistent fine-tuning
- **Model Behavior**: Base MedGemma too lenient in zero-shot scoring
- **Solution**: Hybrid GPT-4 labeling + MedGemma inference approach

## 🛠️ Installation

### Dependencies
```bash
pip install transformers peft torch bitsandbytes streamlit
pip install openai aiofiles python-dotenv  # For data generation only
```

### Hardware Requirements
- **Inference**: 6GB+ GPU memory (with 4-bit quantization)
- **Training**: 16GB+ GPU memory (Kaggle/Colab recommended)

## 📁 Project Structure

```
├── src/
│   ├── generate_gold_notes_gpt.py      # GPT-4 note generation
│   ├── generate_persona_notes_gpt.py   # Quality degradation
│   ├── generate_labeled_notes_gpt.py   # GPT-4 scoring
│   ├── prepare_training_data_medgemma.py    # Training splits
│   └── orchestrator.py            # Multi-agent coordinator
├── notebooks/
│   ├── 05-finetune-agents.ipynb   # Kaggle training
│   └── 06-demo.ipynb             # Interactive demo
├── data/
│   ├── gold/                      # Complete clinical notes
│   ├── brooks/                    # Rushed notes  
│   ├── chen/                      # Variable quality
│   ├── minimal/                   # Inadequate notes
│   └── finetuning/               # Training splits
└── prompts/                       # Generation templates
```

## 🏥 Clinical Impact

**Target Users**: Clinical quality managers, department heads, healthcare administrators

**Benefits**:
- Real-time quality feedback (seconds vs weeks)
- Consistent documentation standards
- HIPAA-compliant on-premises deployment
- Reduced manual review burden

## 📝 Citation

```
HAI-DEF Healthcare AI Competition 2026
Clinical Documentation Quality Scorer
Using MedGemma for Multi-Dimensional Quality Assessment
```

## ⚠️ Limitations

- Trained on synthetic data only
- Limited training examples (200 notes)
- Requires larger dataset for production deployment
- Some dimensions (e.g., compliance) show training instability

## 🔮 Future Work

- Partner with health systems for real clinical note datasets (5000+ examples)
- Specialty-specific model variants (Emergency Medicine, Surgery, etc.)
- Integration with major EMR systems
- Multi-language support for international deployment

---

**Demo**: Access via ngrok tunnel in `notebooks/06-demo.ipynb`  
**Models**: https://huggingface.co/sl02  
**Competition**: HAI-DEF 2026