import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gc

st.set_page_config(
    page_title="Clinical Documentation Quality Scorer",
    page_icon="🏥",
    layout="wide"
)

# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================

@st.cache_resource
def load_base_model(hf_token):
    """Load base model once and cache"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-4b-it",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "google/medgemma-4b-it",
        token=hf_token
    )
    
    return model, tokenizer

def score_dimension(base_model, tokenizer, note_text, dimension, hf_token):
    """Score single dimension - load adapter temporarily"""
    try:
        # Load adapter
        model = PeftModel.from_pretrained(
            base_model,
            f"sl02/clinical-doc-{dimension}-agent",
            token=hf_token
        )
        model.eval()
        
        # Create prompt
        prompt = (
            f"<start_of_turn>user\n"
            f"Score the {dimension} of this clinical note (0-100):\n\n"
            f"{note_text[:500]}\n"  # Check if truncation is issue
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                # eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract score
        prompt_length = inputs.input_ids.shape[1]
        response = tokenizer.decode(
            output[0][prompt_length:], 
            skip_special_tokens=True
        ).strip()
        
        # DEBUG: Print what model actually returns
        # st.write(f"**{dimension} raw response:** `{response}`")
        # st.write(f"Risk raw output: '{response}'")
        # if dimension == "risk":
        #     st.write(f"**DEBUG Risk Agent:**")
        #     st.write(f"Generated tokens: {output[0][prompt_length:]}")
            
        #     # Decode with special tokens visible
        #     response_with_special = tokenizer.decode(
        #         output[0][prompt_length:], 
        #         skip_special_tokens=False  # Don't skip
        #     )
        #     st.write(f"Raw (with special tokens): `{response_with_special}`")
            
        #     # Decode without special tokens
        #     response_without_special = tokenizer.decode(
        #         output[0][prompt_length:], 
        #         skip_special_tokens=True
        #     )
        #     st.write(f"Raw (without special tokens): `{response_without_special}`")
            
        #     # Also decode individual tokens
        #     st.write(f"Token 236771: `{tokenizer.decode([236771])}`")
        #     st.write(f"Token 106: `{tokenizer.decode([106])}`")
        
        # Clean up immediately
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Parse score
        try:
            return int(response)
        except:
            import re
            nums = re.findall(r'\d+', response)
            return int(nums[0]) if nums else 50
            
    except Exception as e:
        st.error(f"Error scoring {dimension}: {str(e)}")
        return None

def validate_input(note_text: str) -> tuple[bool, str]:
        """Validate if input is a clinical note"""
        
        if len(note_text.strip()) < 50:
            return False, "Input too short (minimum 50 characters)"
        
        clinical_keywords = [
            'patient', 'chief complaint', 'cc:', 'history', 'hpi',
            'physical exam', 'pe:', 'vital', 'assessment', 'plan',
            'diagnosis', 'medication', 'treatment', 'symptom'
        ]
        
        text_lower = note_text.lower()
        matches = sum(1 for kw in clinical_keywords if kw in text_lower)
        
        if matches < 2:
            return False, "Input does not appear to be a clinical note"
        
        return True, "Valid"


# ============================================================================
# UI
# ============================================================================

st.title("🏥 Clinical Documentation Quality Scorer")
st.markdown("**Powered by MedGemma 4B** | Multi-Agent Quality Assessment System")

# Sidebar for token
with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        help="Required to load models from HF Hub"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system uses 5 specialized agents to evaluate clinical notes:
    - **Completeness**: All required elements present
    - **Accuracy**: Medical facts correct
    - **Compliance**: Billing documentation adequate
    - **Risk**: Safety elements documented
    - **Clarity**: Clear and well-organized
    """)

# Sample notes
SAMPLE_NOTES = {
    "Complete Note": """# Clinical Note

**Patient Name:** Carlos Rodriguez  
**Date of Service:** [Insert Date]  
**Age:** 33 years  
**Gender:** Male  
**Occupation:** Restaurant Manager  

---

## Chief Complaint
“Back pain for 5 days”

---

## History of Present Illness
Carlos Rodriguez is a 33-year-old male presenting with acute low back pain localized to the lumbar region, primarily left-sided. The pain is described as sharp with movement and a dull ache at rest. Severity is **7/10 with movement** and **4/10 at rest**.

Symptoms began 5 days ago after lifting a 50-pound box at work with a twisting motion. Pain is constant and worse in the morning and after prolonged sitting.

**Exacerbating factors:**
- Bending forward  
- Standing >30 minutes  
- Sneezing  

**Relieving factors:**
- Lying flat  
- Heat application  
- Ibuprofen  

Associated symptom: tingling in the right leg.  
Denies numbness, weakness, saddle anesthesia, or bowel/bladder dysfunction.

---

## Review of Systems
- **Musculoskeletal:** Lower back pain as described; no other joint pain.  
- **Neurological:** Tingling in right leg; no weakness, numbness, or balance issues.  
- **Constitutional:** No fever, chills, or weight loss.  
- **Genitourinary:** No urinary retention or incontinence; normal bowel function.  

---

## Past Medical History
- No history of chronic back problems  
- Last physical examination 3 months ago  

---

## Medications
- Ibuprofen 600 mg as needed for pain  

---

## Allergies
- No known drug allergies  

---

## Social History
- Occupation: Restaurant Manager  
- No tobacco use  
- Occasional alcohol use  

---

## Physical Examination

**Vital Signs**
- BP: 128/82 mmHg  
- HR: 74 bpm  
- Temperature: 98.4°F  
- RR: 14  

**General**
- Alert, uncomfortable with movement, not in severe distress  

**Back**
- Tenderness over left paraspinal muscles at L3–L5  
- No midline tenderness  
- Muscle spasm present  

**Range of Motion**
- Flexion limited to 45° due to pain  
- Extension 15°  
- Lateral bending 20° bilaterally  

**Neurological**
- **Straight Leg Raise:** Negative bilaterally (local discomfort only at 70°)  
- **Motor:** 5/5 strength in bilateral hip flexors, knee extensors, ankle dorsiflexion, plantar flexion, and great toe extension  
- **Sensory:** Intact to light touch L1–S1 bilaterally  
- **Reflexes:** 2+ patellar and Achilles reflexes bilaterally, symmetric, no clonus  
- **Gait:** Antalgic but ambulates independently; able to heel and toe walk  

**Abdomen**
- Soft, non-tender, no costovertebral angle tenderness  

---

## Assessment
1. **Acute mechanical low back pain / lumbar strain** (ICD-10: M54.5)  
2. Mild radiculopathy (right leg tingling)  
3. No red flags for serious pathology  

---

## Plan

### Conservative Management
- No imaging indicated at this time  

### Medications
- Ibuprofen 600 mg three times daily with food for 7–10 days  
- Cyclobenzaprine 5 mg at bedtime for 3–5 days (avoid if excessive sedation)  

### Activity Modification
- Avoid lifting >20 lbs for 2 weeks  
- Avoid prolonged sitting (>1 hour without break)  
- Return to normal activities as tolerated  

### Physical Therapy
- Referral for core strengthening and body mechanics training  
- Initiate in 1 week if symptoms improving  

### Adjunct Therapy
- Alternate heat and ice for 15–20 minutes several times daily  

### Education
- Discussed expected course (most improve within 2–4 weeks)  
- Reviewed red flag symptoms requiring urgent evaluation:
  - Progressive leg weakness  
  - Numbness in groin/inner thighs  
  - Loss of bowel/bladder control  
  - Worsening or severe pain  

### Work Note
- Light duty with 20-pound lifting restriction for 2 weeks  

### Follow-Up
- Return in 2 weeks if not significantly improved  
- Earlier evaluation if symptoms worsen or new symptoms develop  
- Imaging (X-ray or MRI) if no improvement at 4–6 weeks or if red flags develop  

---

## Billing Information
- **CPT Code:** 99214 – Established patient, moderate complexity  
- **ICD-10:** M54.5  
- **Total Time Spent:** 20 minutes  

---

## Medical Decision Making
Comprehensive red flag assessment performed and negative. Clinical presentation consistent with acute mechanical low back pain/lumbar 
strain with mild radicular symptoms. No indication for imaging per guidelines. Medication risks/benefits and activity modification discussed. 
Patient educated on warning signs requiring urgent evaluation.""",

"Incomplete Note": """CC: Back pain

HPI: Pt reports low back pain x 3 days. Started after lifting.

PE: Tenderness lower back. ROM limited."""
}

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Clinical Note Input")
    
    # Sample selector
    selected_sample = st.selectbox(
        "Load sample note:",
        ["", "Complete Note", "Incomplete Note"]
    )
    
    if selected_sample:
        default_text = SAMPLE_NOTES[selected_sample]
    else:
        default_text = ""
    
    note_text = st.text_area(
        "Enter clinical note:",
        value=default_text,
        height=400,
        placeholder="Paste clinical note here..."
    )
    
    analyze_button = st.button("🔍 Analyze Quality", type="primary", use_container_width=True)

with col2:
    st.subheader("Quality Scores")
    score_placeholder = st.empty()

# Analysis
if analyze_button:
    if not hf_token:
        st.error("Please enter your Hugging Face token in the sidebar")
    elif not note_text.strip():
        st.warning("Please enter a clinical note to analyze")
    else:
        is_valid, msg = validate_input(note_text.strip())
        if is_valid:
            with st.spinner("Loading models..."):
                try:
                    base_model, tokenizer = load_base_model(hf_token)
                    st.success(" Models loaded")
                except Exception as e:
                    st.error(f"Failed to load models: {str(e)}")
                    st.stop()
            
            # Score each dimension
            dimensions = ["completeness", "accuracy", "compliance", "risk", "clarity"]
            scores = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, dim in enumerate(dimensions):
                status_text.text(f"Scoring {dim}...")
                score = score_dimension(base_model, tokenizer, note_text, dim, hf_token)
                # st.write(f"Risk extracted: {score}")
                
                if score is not None:
                    scores[dim] = score
                
                progress_bar.progress((i + 1) / len(dimensions))
            
            status_text.text("Analysis complete!")
            
            # Calculate overall
            if scores:
                scores['overall'] = sum(scores.values()) / len(scores)
            
            # Display results
            with score_placeholder.container():
                # Overall score
                st.metric(
                    label="Overall Quality",
                    value=f"{scores.get('overall', 0):.0f}/100",
                    delta=None
                )
                
                st.markdown("---")
                
                # Individual scores with color coding
                for dim in dimensions:
                    score = scores.get(dim, 0)
                    
                    # Color coding
                    if score >= 90:
                        color = "🟢"
                    elif score >= 70:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.metric(
                        label=f"{color} {dim.capitalize()}",
                        value=f"{score}/100"
                    )
            
            # Feedback section
            st.markdown("---")
            st.subheader("📋 Recommendations")
            
            issues = []
            if scores.get('completeness', 100) < 70:
                issues.append("- Missing required documentation elements")
            if scores.get('accuracy', 100) < 70:
                issues.append("- Review medical facts and diagnoses")
            if scores.get('compliance', 100) < 70:
                issues.append("- Insufficient billing documentation")
            if scores.get('risk', 100) < 70:
                issues.append("- Missing critical safety elements")
            if scores.get('clarity', 100) < 70:
                issues.append("- Improve organization and clarity")
            
            if issues:
                st.markdown("\n".join(issues))
            else:
                st.success(" Documentation meets quality standards")
        else:
            st.warning(msg)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>HAI-DEF Competition 2026 | "
    "Built with MedGemma 4B</div>",
    unsafe_allow_html=True
)

def main():
    """Entry point for launching Streamlit app"""
    import sys

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())