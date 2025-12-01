# Inpatient Data Inference Scripts

This directory contains scripts for running QoQ-Med-VL-7B inference on inpatient encounter data.

## Quick Start

### Option 1: Prepare Prompts Only (No GPU Required)
```bash
python3 prepare_inpatient_prompts.py
```
This creates formatted prompts from the inpatient dataset without running inference.

**Outputs:**
- `prepared_prompts_[timestamp].txt` - Human-readable prompts
- `prepared_prompts_[timestamp].json` - Structured JSON data

### Option 2: Full Inference with GPU
```bash
python3 inpatient_inference.py
```
This runs the QoQ-Med-VL-7B model on formatted inpatient encounters.

**Requirements:**
- GPU with CUDA (A100 or newer recommended)
- All packages from `requirements.txt`

**Output:**
- `inpatient_analysis_results.txt` - Encounter summaries with model analysis

## Files

| File | Description |
|------|-------------|
| `inpatient_inference.py` | Main inference script (requires GPU) |
| `prepare_inpatient_prompts.py` | Prompt formatter (no GPU needed) |
| `INPATIENT_INFERENCE_GUIDE.md` | Comprehensive documentation |
| `simple_inference_example.py` | Original image-based inference example |

## Dataset

- **File:** `datasets/dbo.accm_inpatient.csv`
- **Records:** 8.3M+ patient encounters
- **Columns:** 33 features including admission details, departments, providers, outcomes

## Key Features

### Automated Encounter Formatting
- Extracts relevant clinical information
- Calculates length of stay
- Maps coded values to readable labels
- Creates structured clinical narratives

### Multiple Analysis Tasks
- `analysis` - General clinical analysis
- `prediction` - LOS and readmission risk
- `risk` - Safety and quality concerns
- `recommendations` - Process improvements
- `summary` - Concise summaries

### Flexible Filtering
Filter by:
- Department specialty
- Admission type
- ED visits
- Any column in the dataset

## Example Configuration

Edit the `config` dictionary in either script:

```python
config = {
    'num_samples': 5,              # Number of encounters
    'task': 'analysis',            # Analysis type
    'filter_criteria': {           # Optional filters
        'dep_speciality': 'Emergency Medicine',
        'ed_visit_yn': 'Y'
    },
    'output_file': 'results.txt'   # Output filename
}
```

## Sample Output Format

```
PATIENT ENCOUNTER SUMMARY

Encounter ID: 1085008483.0
Patient ID: 0028bcb6...

ADMISSION DETAILS:
- Admission Type: Urgent
- ED Visit: Yes
- Length of Stay: 1.33 hours

CLINICAL SERVICE:
- Department Specialty: Emergency Medicine
- Hospital Service: Emergency Medicine

OUTCOME:
- Discharge Disposition: Home

[Model Analysis Follows...]
```

## Documentation

See **[INPATIENT_INFERENCE_GUIDE.md](INPATIENT_INFERENCE_GUIDE.md)** for:
- Detailed usage instructions
- Dataset schema and code mappings
- Computational requirements
- Example workflows
- Troubleshooting guide

## Quick Reference: Analysis Tasks

| Task | Use Case |
|------|----------|
| `analysis` | General clinical review and observations |
| `prediction` | Length of stay and readmission analysis |
| `risk` | Safety concerns and quality assessment |
| `recommendations` | Care improvement suggestions |
| `summary` | Brief clinical summaries |

## Common Workflows

### 1. ED Analysis
```python
filter_criteria = {'dep_speciality': 'Emergency Medicine'}
task = 'analysis'
```

### 2. Readmission Risk
```python
filter_criteria = None  # All departments
task = 'prediction'
```

### 3. Quality Review
```python
filter_criteria = {'hosp_admsn_type_c': 3.0}  # Elective admissions
task = 'recommendations'
```

## Data Privacy Notice

⚠️ **Important:** This dataset contains patient encounter data. Ensure:
- HIPAA compliance
- Institutional IRB approval
- Proper data access controls
- Secure output storage

## Tips

1. **Start small:** Test with `num_samples=3` first
2. **Use filters:** Focus on specific specialties or case types
3. **Review outputs:** Validate model responses for clinical accuracy
4. **Monitor GPU:** Use `nvidia-smi` to track VRAM usage
5. **Batch processing:** For large analyses, use `prepare_inpatient_prompts.py` first

## Requirements

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn torch transformers qwen-vl-utils
```

Or use the project requirements:
```bash
pip install -r requirements.txt
```

## Support

For detailed information, see:
- [INPATIENT_INFERENCE_GUIDE.md](INPATIENT_INFERENCE_GUIDE.md) - Complete documentation
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Original inference guide
- Project README.md - General project information
