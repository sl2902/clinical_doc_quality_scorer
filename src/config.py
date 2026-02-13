model_name = "google/medgemma-4b-it"

model_name_gpt = "gpt-4o"
MAX_REQUESTS_PER_MINUTE = 30
MAX_TOKENS_PER_MINUTE = 60000
SEMAPHORE_SIZE = 5
BATCH_SIZE = 5
BATCH_DELAY = 3 

scenarios = ["uri", "htn_followup", "t2dm", "back_pain", "annual_physical"]
personas = ["brooks", "chen", "minimal"]
dimensions = ["completeness", "accuracy", "compliance", "risk", "clarity"]

data_dir = "data"
base_dir = "/kaggle/input/prompts"
gold_prompt_dir = "/kaggle/input/gold-prompts/generated_prompts"
persona_prompt_dir = "/kaggle/input/personas"
persona_generated_dir = "/kaggle/input/personas-generated/generated_personas"

data_dir_gold = "data/gold"

# 50 gold; 50 brooks; 50 chen
max_notes = 150

# training prompt
prompt = "Score the accuracy of this clinical note (0-100):"

# demographic details used in the scenario prompts
demographics_male = [
    {"age": 42, "gender": "M", "occupation": "software engineer", "name": "Robert Chen"},
    {"age": 51, "gender": "M", "occupation": "sales manager", "name": "David Kim"},
    {"age": 38, "gender": "M", "occupation": "warehouse worker", "name": "James Wilson"},
    {"age": 33, "gender": "M", "occupation": "restaurant manager", "name": "Carlos Rodriguez"},
    {"age": 47, "gender": "M", "occupation": "high school teacher", "name": "Michael Brown"}
]

demographics_female = [
    {"age": 35, "gender": "F", "occupation": "elementary teacher", "name": "Jane Doe"},
    {"age": 28, "gender": "F", "occupation": "registered nurse", "name": "Maria Garcia"},
    {"age": 45, "gender": "F", "occupation": "accountant", "name": "Lisa Anderson"},
    {"age": 29, "gender": "F", "occupation": "graphic designer", "name": "Emily Taylor"},
    {"age": 39, "gender": "F", "occupation": "project manager", "name": "Sarah Johnson"}
]

# gender-specific content block
gender_content = {
    "annual_physical": {
        "M": {
            "gender_specific_screening": """* Prostate: No urinary symptoms, PSA discussed (age <50, deferred per patient preference)
  * Testicular self-exam: Reviewed technique""",
            
            "gu_ros": "No dysuria, no urinary frequency/urgency, no erectile dysfunction, no hematuria",
            
            "breast_exam": "Breast: Normal male breast tissue, no gynecomastia, no masses",
            
            "gu_exam": "Normal male external genitalia, both testes descended and normal size/consistency, no masses, no hernias, circumcised, no lesions",
            
            "gender_specific_cancer_screening": """* Prostate cancer: PSA discussed, deferred per shared decision-making (age <50, average risk)
  * Testicular cancer: Self-exam technique reviewed"""
        },
        
        "F": {
            "gender_specific_screening": """* Last mammogram: 1 year ago (normal, due for repeat today)
  * Last Pap smear: 3 years ago (normal, not due yet - every 3 years per guidelines)
  * Menstrual history: Regular periods, cycle 28 days""",
            
            "gu_ros": "No dysuria, no urinary frequency/urgency, menstrual periods regular (still menstruating), no abnormal vaginal bleeding",
            
            "breast_exam": "Breast: No masses, no skin changes, no nipple discharge, no lymphadenopathy (axillary, supraclavicular, infraclavicular)",
            
            "gu_exam": "External genitalia normal, no lesions, pelvic exam deferred (recent normal Pap smear)",
            
            "gender_specific_cancer_screening": """* Mammogram: Ordered today (annual for ages 40-75)
  * Cervical cancer: Pap smear not due (last 3 years ago, continue every 3 years per age and HPV status)"""
        }
    },
    "htn_followup": {
        "M": {
            "gender_specific_htn_questions": "- Sexual function: No erectile dysfunction, no concerns related to medication",
            
            "gender_specific_htn_counseling": "- Sexual health: Discussed that BP medications can affect erectile function, instructed to report any concerns"
        },
        
        "F": {
            "gender_specific_htn_questions": "- Menstrual status: Regular periods / Postmenopausal\n- Contraception: Using [method] / Not applicable\n- Pregnancy screening: Not pregnant (LMP 2 weeks ago) / Not applicable if postmenopausal",
            
            "gender_specific_htn_counseling": "- Contraception counseling: ACE inhibitors contraindicated in pregnancy, importance of reliable contraception discussed / Not applicable if postmenopausal"
        }
    },
    "t2dm": {
        "M": {
            "gender_specific_diabetes_questions": "- Sexual function: Reports occasional erectile dysfunction, may be related to diabetes",
            
            "gender_specific_diabetes_counseling": "- Erectile dysfunction: Discussed relationship to diabetes and vascular health, may improve with better glucose control, consider PDE5 inhibitors if persistent"
        },
        
        "F": {
            "gender_specific_diabetes_questions": "- Gestational diabetes history: None / Had gestational diabetes with pregnancy\n- Pregnancy planning: Not currently planning pregnancy / Actively trying to conceive\n- Menstrual status: Regular periods / Irregular cycles related to obesity and insulin resistance",
            
            "gender_specific_diabetes_counseling": "- Pregnancy planning: If planning pregnancy, HbA1c should be <6.5% prior to conception, discussed preconception counseling, GLP-1 agonists contraindicated in pregnancy (discontinue if pregnancy planned or occurs)\n- Contraception: Discussed need for reliable contraception while on diabetes medications"
        }
    }
}

scenario_variations = {
    "uri": [
        {"severity": "mild", "duration": "3 days", "complications": "none"},
        {"severity": "moderate", "duration": "5 days", "complications": "none"},
        {"severity": "moderate", "duration": "7 days", "complications": "mild sinusitis"},
        {"severity": "mild", "duration": "4 days", "complications": "none"},
        {"severity": "severe", "duration": "10 days", "complications": "secondary bacterial infection suspected"},
    ],
    
    "htn_followup": [
        {"control": "well-controlled", "adherence": "excellent", "labs": "all normal"},
        {"control": "uncontrolled", "adherence": "good", "labs": "borderline lipids"},
        {"control": "improving", "adherence": "variable", "labs": "elevated creatinine"},
        {"control": "uncontrolled", "adherence": "excellent", "labs": "normal"},
        {"control": "well-controlled", "adherence": "good", "labs": "low potassium"},
    ],
    
    "t2dm": [
        {"control": "uncontrolled", "a1c": "8.2%", "complications": "neuropathy symptoms"},
        {"control": "poorly controlled", "a1c": "9.5%", "complications": "none yet"},
        {"control": "improving", "a1c": "7.8%", "complications": "mild retinopathy"},
        {"control": "uncontrolled", "a1c": "8.5%", "complications": "none"},
        {"control": "very poor", "a1c": "10.2%", "complications": "foot ulcer concern"},
    ],
    
    "back_pain": [
        {"severity": "moderate", "radiation": "no radiculopathy", "duration": "3 days"},
        {"severity": "severe", "radiation": "L leg radiculopathy", "duration": "2 weeks"},
        {"severity": "mild", "radiation": "no radiculopathy", "duration": "1 day"},
        {"severity": "moderate", "radiation": "R leg tingling", "duration": "5 days"},
        {"severity": "severe", "radiation": "bilateral leg weakness", "duration": "1 week"},
    ],
    
    "annual_physical": [
        {"focus": "routine", "concerns": "none", "findings": "all normal"},
        {"focus": "new hypertension", "concerns": "family history", "findings": "elevated BP"},
        {"focus": "weight management", "concerns": "prediabetes", "findings": "obesity"},
        {"focus": "routine", "concerns": "fatigue", "findings": "low vitamin D"},
        {"focus": "preventive", "concerns": "cancer screening", "findings": "all normal"},
    ]
}
