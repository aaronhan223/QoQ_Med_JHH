#!/usr/bin/env python3
"""
Inpatient Data Formatter for Medical LLM Inference - MCQ-5: Length of Stay Category

This script formats inpatient encounter data into structured clinical narratives
for Length of Stay prediction tasks (MCQ-5 from designed_qa_questions.txt).
Generates QA pairs following the format from jeannieshe/multimodal repository.

MCQ-5 Categories:
- A. Short stay (0-2 days)
- B. Moderate stay (3-7 days)
- C. Extended stay (8-14 days)
- D. Long-term stay (>14 days)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import uuid


def calculate_length_of_stay(row):
    """Calculate length of stay in hours and days"""
    try:
        if pd.notna(row['hosp_admsn_time']) and pd.notna(row['hosp_disch_time']):
            adm = pd.to_datetime(row['hosp_admsn_time'])
            disch = pd.to_datetime(row['hosp_disch_time'])
            los_hours = (disch - adm).total_seconds() / 3600
            los_days = los_hours / 24
            return round(los_hours, 2), round(los_days, 2) if los_hours >= 0 else (None, None)
    except:
        pass
    return None, None


def categorize_los(los_days):
    """
    Categorize length of stay into MCQ-5 categories
    Returns: (category_letter, category_description)
    """
    if los_days is None:
        return None, None
    
    if los_days <= 2:
        return "A", "Short stay (0-2 days)"
    elif los_days <= 7:
        return "B", "Moderate stay (3-7 days)"
    elif los_days <= 14:
        return "C", "Extended stay (8-14 days)"
    else:
        return "D", "Long-term stay (>14 days)"


def qa_id():
    """Generate a unique QA ID"""
    return str(uuid.uuid4())


def format_patient_encounter(row):
    """
    Format a patient encounter record into a structured clinical narrative
    for Length of Stay prediction (MCQ-5)
    """
    
    # Extract key information
    patient_id = row.get('osler_id', 'Unknown')
    encounter_id = row.get('pat_enc_csn_id', 'Unknown')
    
    # Department and service information
    specialty = row.get('dep_speciality', 'Unknown Specialty')
    hospital_service = row.get('hospital_service', 'Unknown Service')
    service_area = row.get('serv_area_name', 'Unknown Area')
    
    # Admission information
    admission_type_code = row.get('hosp_admsn_type_c')
    admission_type_map = {1.0: 'Emergency', 2.0: 'Urgent', 3.0: 'Elective', 
                          4.0: 'Newborn', 5.0: 'Trauma'}
    admission_type = admission_type_map.get(admission_type_code, 'Unknown')
    
    # Time information
    admission_time = row.get('hosp_admsn_time', 'Unknown')
    arrival_time = row.get('adt_arrival_time', 'Unknown')
    
    # ED visit
    ed_visit = row.get('ed_visit_yn', 'N')
    ed_text = "Yes" if ed_visit == 'Y' else "No"
    
    # Providers
    admission_prov = row.get('admission_prov', 'Not recorded')
    
    # Demographics (if available from other sources)
    # For now, we'll include basic encounter info
    
    # Create structured narrative for LOS prediction
    # Following the style from the GitHub repo examples
    narrative = f"""Patient is admitted through {admission_type.lower()} admission.

Admission Details:
- ED Visit: {ed_text}
- Arrival Time: {arrival_time}
- Admission Time: {admission_time}
- Admitting Provider: {admission_prov}

Clinical Service:
- Department Specialty: {specialty}
- Hospital Service: {hospital_service}
- Service Area: {service_area}

The patient is being admitted for further evaluation and treatment."""
    
    return narrative


def create_los_qa_prompt(encounter_text):
    """
    Create Length of Stay (MCQ-5) prompt following the format from 
    jeannieshe/multimodal repository.
    
    Based on create_qa_pairs_with_metadata.py question_type == 6
    """
    
    problem = (
        "Below is a history of a patient:\n"
        f"They have the following medical history: {encounter_text}\n"
        f"How long will the patient stay in the hospital?\n"
        f"A. Short stay (0-2 days)\n"
        f"B. Moderate stay (3-7 days)\n"
        f"C. Extended stay (8-14 days)\n"
        f"D. Long-term stay (>14 days)"
    )
    
    return problem


def prepare_inference_data(
    data_path,
    num_samples=5,
    filter_criteria=None,
    output_format="jsonl",
    exclude_missing_discharge=True
):
    """
    Prepare formatted QA pairs from the inpatient data for MCQ-5: Length of Stay Category
    Following the format from jeannieshe/multimodal repository
    
    Args:
        data_path: Path to CSV file
        num_samples: Number of encounters to prepare
        filter_criteria: Dict of filters
        output_format: 'jsonl', 'text', or 'both'
        exclude_missing_discharge: If True, only use completed encounters with discharge times
    
    Returns:
        List of prepared QA pairs in JSONL format
    """
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total encounters: {len(df):,}")
    
    # Filter out encounters without discharge times (can't calculate LOS)
    if exclude_missing_discharge:
        initial_count = len(df)
        df = df[pd.notna(df['hosp_disch_time']) & pd.notna(df['hosp_admsn_time'])]
        print(f"Encounters with complete admission/discharge times: {len(df):,} (filtered {initial_count - len(df):,})")
    
    # Apply additional filters
    if filter_criteria:
        print(f"\nApplying filters: {filter_criteria}")
        for col, value in filter_criteria.items():
            if col in df.columns:
                df = df[df[col] == value]
        print(f"Encounters after filtering: {len(df):,}")
    
    # Sample encounters
    if len(df) > num_samples:
        df_sample = df.sample(n=num_samples, random_state=42)
    else:
        df_sample = df.head(num_samples)
    
    print(f"\nPreparing {len(df_sample)} encounters for Length of Stay prediction (MCQ-5)...")
    print("=" * 80)
    
    prepared_data = []
    
    for idx, (_, row) in enumerate(df_sample.iterrows(), 1):
        print(f"\n{'='*80}")
        print(f"ENCOUNTER {idx}/{len(df_sample)}")
        print(f"{'='*80}")
        
        # Calculate actual LOS
        los_hours, los_days = calculate_length_of_stay(row)
        
        if los_days is None:
            print(f"Skipping encounter - unable to calculate LOS")
            continue
        
        # Categorize LOS
        los_category, los_description = categorize_los(los_days)
        
        # Format encounter information (WITHOUT discharge info - this is for prediction)
        encounter_text = format_patient_encounter(row)
        
        # Create QA prompt
        problem = create_los_qa_prompt(encounter_text)
        
        print(f"\nEncounter ID: {row.get('pat_enc_csn_id')}")
        print(f"Actual LOS: {los_days:.2f} days ({los_hours:.2f} hours)")
        print(f"LOS Category: {los_category} - {los_description}")
        print(f"\n{problem}")
        
        # Create QA pair following the GitHub repo format
        qa_pair = {
            'qa_id': qa_id(),
            'qa_type': 6,  # Type 6 corresponds to Length of Stay question
            'format': 'Multiple Choice',
            'question': problem,
            'images': [],  # No images in our dataset
            'time-series': [],  # No time-series data
            'choices': [
                'A. Short stay (0-2 days)',
                'B. Moderate stay (3-7 days)',
                'C. Extended stay (8-14 days)',
                'D. Long-term stay (>14 days)'
            ],
            'correct_choice': los_category,
            'answer': los_category,
            'encounter_id': str(row.get('pat_enc_csn_id')),
            'patient_id': str(row.get('osler_id')),
            'specialty': str(row.get('dep_speciality')),
            'hospital_service': str(row.get('hospital_service')),
            'admission_type_c': int(row.get('hosp_admsn_type_c')) if pd.notna(row.get('hosp_admsn_type_c')) else None,
            'admission_time': str(row.get('hosp_admsn_time')),
            'discharge_time': str(row.get('hosp_disch_time')),
            'los_days': float(los_days),
            'los_hours': float(los_hours),
            'ed_visit': str(row.get('ed_visit_yn', 'N'))
        }
        
        prepared_data.append(qa_pair)
    
    # Save outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format in ['jsonl', 'both']:
        jsonl_file = f'los_qa_pairs_{timestamp}.jsonl'
        with open(jsonl_file, 'w') as f:
            for qa in prepared_data:
                f.write(json.dumps(qa) + '\n')
        
        print(f"\n✓ JSONL file saved: {jsonl_file}")
    
    if output_format in ['text', 'both']:
        text_file = f'los_qa_pairs_{timestamp}.txt'
        with open(text_file, 'w') as f:
            f.write(f"Length of Stay Prediction QA Pairs (MCQ-5)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total QA pairs: {len(prepared_data)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, qa in enumerate(prepared_data, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"QA PAIR {i}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Question:\n{qa['question']}\n\n")
                f.write(f"Correct Answer: {qa['correct_choice']}\n")
                f.write(f"Actual LOS: {qa['los_days']:.2f} days\n")
                f.write(f"Encounter ID: {qa['encounter_id']}\n")
                f.write(f"Specialty: {qa['specialty']}\n\n")
        
        print(f"✓ Text file saved: {text_file}")
    
    print(f"\n{'='*80}")
    print(f"✓ Prepared {len(prepared_data)} QA pairs for inference")
    print(f"✓ Format: Multiple Choice (MCQ-5: Length of Stay Category)")
    print(f"{'='*80}\n")
    
    return prepared_data


def main():
    """Main execution"""
    
    print("="*80)
    print("Length of Stay QA Pair Generator (MCQ-5)")
    print("Based on designed_qa_questions.txt")
    print("="*80)
    
    # Configuration
    data_path = Path(__file__).parent / "datasets" / "dbo.accm_inpatient.csv"
    
    if not data_path.exists():
        print(f"✗ Error: Data file not found at {data_path}")
        return
    
    config = {
        'num_samples': 10,  # Number of QA pairs to generate
        'filter_criteria': None,  # Example: {'dep_speciality': 'Emergency Medicine'}
        'output_format': 'both',  # Options: 'jsonl', 'text', 'both'
        'exclude_missing_discharge': True  # Only use completed encounters
    }
    
    print(f"\nConfiguration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Number of samples: {config['num_samples']}")
    print(f"  - Filters: {config['filter_criteria']}")
    print(f"  - Output format: {config['output_format']}")
    print(f"  - Exclude incomplete encounters: {config['exclude_missing_discharge']}")
    
    # Prepare QA pairs
    prepared_data = prepare_inference_data(
        data_path=data_path,
        num_samples=config['num_samples'],
        filter_criteria=config['filter_criteria'],
        output_format=config['output_format'],
        exclude_missing_discharge=config['exclude_missing_discharge']
    )
    
    print(f"\n✓ QA pair generation completed!")
    print(f"✓ {len(prepared_data)} QA pairs ready for model inference")
    print(f"\nQA Pair Format:")
    print(f"  - Question Type: MCQ-5 (Length of Stay Category)")
    print(f"  - Format: Multiple Choice (4 options)")
    print(f"  - Categories: A (0-2 days), B (3-7 days), C (8-14 days), D (>14 days)")
    print(f"  - Metadata: Includes encounter details, actual LOS, specialty, etc.")


if __name__ == "__main__":
    main()
