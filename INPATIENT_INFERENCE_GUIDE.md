# Inpatient Data Inference Guide

This guide explains how to use the QoQ-Med-VL-7B model with the inpatient encounter dataset (`dbo.accm_inpatient.csv`).

## Overview

The inpatient dataset contains clinical encounter data with the following key information:
- **Patient identifiers**: osler_id, pat_enc_csn_id
- **Admission details**: admission type, arrival/admission/discharge times
- **Clinical services**: department specialty, hospital service, providers
- **Outcomes**: discharge disposition, length of stay
- **ED information**: ED visit flag, ED departure time

## Scripts

### 1. `inpatient_inference.py` (Full Inference with GPU)

Main script that runs the QoQ-Med-VL-7B model on formatted inpatient encounters.

**Requirements:**
- GPU with CUDA support (preferably A100 or newer for flash_attention_2)
- All dependencies from `requirements.txt`
- Sufficient VRAM (16GB+)

**Usage:**
```bash
# Basic usage (analyzes 3 random encounters)
python3 inpatient_inference.py

# Edit the config section in the script to customize:
config = {
    'num_samples': 5,              # Number of encounters to analyze
    'task': 'analysis',            # Analysis type
    'filter_criteria': None,       # Filter by specialty, etc.
    'output_file': 'results.txt'   # Output filename
}
```

**Analysis Tasks:**
- `analysis` - General clinical analysis and observations
- `prediction` - LOS factors and readmission risk prediction
- `risk` - Clinical risk assessment and safety concerns
- `recommendations` - Care improvement recommendations
- `summary` - Concise clinical summary

**Filtering Examples:**
```python
# Analyze only Emergency Medicine encounters
filter_criteria = {'dep_speciality': 'Emergency Medicine'}

# Analyze only elective admissions (type 3)
filter_criteria = {'hosp_admsn_type_c': 3.0}

# Combine multiple filters
filter_criteria = {
    'dep_speciality': 'Emergency Medicine',
    'ed_visit_yn': 'Y'
}
```

### 2. `prepare_inpatient_prompts.py` (Prompt Preparation without GPU)

Lightweight script that formats encounter data into structured prompts without running inference. Useful for:
- Testing prompt formatting
- Preparing batch inference data
- Reviewing data before expensive GPU runs

**Usage:**
```bash
# Run the formatter
python3 prepare_inpatient_prompts.py

# Outputs:
# - prepared_prompts_YYYYMMDD_HHMMSS.txt  (human-readable)
# - prepared_prompts_YYYYMMDD_HHMMSS.json (structured data)
```

**Customization:**
Edit the config in the script:
```python
config = {
    'num_samples': 5,
    'task': 'analysis',
    'filter_criteria': None,
    'output_format': 'both'  # 'text', 'json', or 'both'
}
```

## Dataset Schema

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `osler_id` | String | Patient identifier |
| `pat_enc_csn_id` | Float | Encounter identifier |
| `hosp_admsn_type_c` | Float | Admission type code (1=Emergency, 2=Urgent, 3=Elective, etc.) |
| `adt_arrival_time` | DateTime | Patient arrival time |
| `hosp_admsn_time` | DateTime | Hospital admission time |
| `hosp_disch_time` | DateTime | Hospital discharge time |
| `dep_speciality` | String | Department specialty |
| `hospital_service` | String | Hospital service name |
| `admission_prov` | String | Admitting provider |
| `discharge_prov` | String | Discharge provider |
| `disch_disp_c` | Float | Discharge disposition code |
| `ed_visit_yn` | String | ED visit flag (Y/N) |
| `serv_area_name` | String | Service area |

### Code Mappings

**Admission Types (`hosp_admsn_type_c`):**
- 1.0 = Emergency
- 2.0 = Urgent
- 3.0 = Elective
- 4.0 = Newborn
- 5.0 = Trauma

**Discharge Disposition (`disch_disp_c`):**
- 1.0 = Home
- 2.0 = Transfer
- 3.0 = Skilled Nursing Facility
- 4.0 = Expired
- 5.0 = Left Against Medical Advice
- 6.0 = Home Health

## Example Formatted Prompt

```
PATIENT ENCOUNTER SUMMARY

Encounter ID: 1085008483.0
Patient ID: 0028bcb6...

ADMISSION DETAILS:
- Admission Type: Urgent
- ED Visit: Yes
- Arrival Time: 2015-10-20 15:13:00
- Admission Time: 2015-10-20 21:49:00
- Discharge Time: 2015-10-20 23:09:00
- Length of Stay: 1.33 hours

CLINICAL SERVICE:
- Department Specialty: Emergency Medicine
- Hospital Service: Emergency Medicine
- Service Area: JHM CLINICAL

PROVIDERS:
- Admitting Provider: Not recorded
- Discharge Provider: Not recorded

OUTCOME:
- Discharge Disposition: Home

Based on this patient encounter summary, please provide:
1. A brief clinical analysis of the admission pattern
2. Notable observations about the care delivery (length of stay, department, etc.)
3. Any potential areas of concern or interest from a quality improvement perspective
```

## Output Files

### `inpatient_analysis_results.txt`
Contains:
- Timestamp and configuration
- For each encounter:
  - Formatted encounter summary
  - Model-generated analysis

### `prepared_prompts_*.json`
Structured JSON with:
```json
[
  {
    "encounter_id": "1085008483.0",
    "patient_id": "0028bcb6...",
    "specialty": "Emergency Medicine",
    "admission_type": 2.0,
    "los_hours": 1.33,
    "encounter_summary": "...",
    "full_prompt": "...",
    "task": "analysis"
  }
]
```

## Tips and Best Practices

### 1. Start Small
Begin with `num_samples=3` to test the pipeline before scaling up.

### 2. Filter Strategically
Use filters to focus on specific specialties or case types:
```python
# High-value analysis targets
filter_criteria = {'dep_speciality': 'Internal Medicine'}
filter_criteria = {'hosp_admsn_type_c': 1.0}  # Emergency admissions only
```

### 3. Choose Appropriate Tasks
- Use `summary` for quick overviews
- Use `risk` for quality/safety reviews
- Use `prediction` for readmission analysis
- Use `recommendations` for process improvement

### 4. Batch Processing
For large-scale analysis:
1. Use `prepare_inpatient_prompts.py` to create prompt batches
2. Review the formatted prompts
3. Run `inpatient_inference.py` on filtered subsets
4. Combine results for aggregate analysis

### 5. Monitor GPU Usage
The model requires significant VRAM:
- Monitor with: `nvidia-smi`
- Reduce `max_new_tokens` if running out of memory
- Process in smaller batches

### 6. Validation
Always review a sample of model outputs to ensure:
- Relevant clinical insights
- Appropriate medical language
- Accurate interpretation of encounter data

## Computational Requirements

### Minimum:
- GPU: NVIDIA A100 (40GB) or equivalent
- RAM: 32GB
- Storage: 50GB for model + data

### Recommended:
- GPU: NVIDIA A100 (80GB)
- RAM: 64GB
- Storage: 100GB

## Common Issues

### 1. CUDA Out of Memory
**Solution:** Reduce `num_samples`, `max_new_tokens`, or use gradient checkpointing

### 2. Missing Data in Encounters
**Solution:** Script handles missing values gracefully with "Unknown" or "Not recorded"

### 3. Slow Inference
**Solution:** 
- Ensure flash_attention_2 is properly installed
- Check GPU utilization
- Consider batch processing

### 4. Model Not Found
**Solution:** 
- Check internet connection
- Verify HuggingFace access
- May need HuggingFace token for some models

## Example Workflows

### Workflow 1: Emergency Department Analysis
```python
# In inpatient_inference.py
config = {
    'num_samples': 10,
    'task': 'analysis',
    'filter_criteria': {
        'dep_speciality': 'Emergency Medicine',
        'ed_visit_yn': 'Y'
    },
    'output_file': 'ed_analysis.txt'
}
```

### Workflow 2: Readmission Risk Assessment
```python
config = {
    'num_samples': 20,
    'task': 'prediction',
    'filter_criteria': None,  # All specialties
    'output_file': 'readmission_risk.txt'
}
```

### Workflow 3: Quality Improvement Review
```python
config = {
    'num_samples': 15,
    'task': 'recommendations',
    'filter_criteria': {'hosp_admsn_type_c': 3.0},  # Elective only
    'output_file': 'quality_review.txt'
}
```

## Data Privacy

**Important:** This dataset contains patient encounter data. Ensure:
- Compliance with HIPAA and institutional policies
- Proper data access controls
- Secure storage of outputs
- De-identification where required
- Approved research protocols

## Citation

If using this code for research, please cite:
- QoQ-Med-VL-7B model paper
- Your institution's data governance policies
- Relevant clinical research protocols

## Support

For issues or questions:
1. Check this documentation
2. Review error messages in detail
3. Verify GPU/CUDA setup
4. Check data file permissions
5. Contact research computing support
