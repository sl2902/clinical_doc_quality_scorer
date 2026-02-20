# Clinical Documentation Quality Scorer

Multi-agent system for real-time clinical note quality assessment using MedGemma 4B.

## Quick Start

1. Install dependencies:
```bash
   pip install transformers peft torch bitsandbytes
```

2. Run demo:
```bash
   # Open notebooks/06-demo.ipynb in Colab
```

3. Models available at: https://huggingface.co/sl02

## Architecture

5 specialized agents score clinical notes across:
- Completeness, Accuracy, Compliance, Risk, Clarity

## Results

- MAE: 16.5 points (on 0-100 scale)
- 85% predictions within 20 points of ground truth

## Citation
HAI-DEF Healthcare AI Competition 2026