#!/usr/bin/env python3
"""
Inpatient Data Formatter for Medical LLM Inference

This script formats inpatient encounter data into structured clinical narratives
without running the actual model inference. Useful for preparing prompts and
testing data formatting before running expensive GPU inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json


def calculate_length_of_stay(row):
    """Calculate length of stay in hours"""
    try:
        if pd.notna(row['hosp_admsn_time']) and pd.notna(row['hosp_disch_time']):
            adm = pd.to_datetime(row['hosp_admsn_time'])
            disch = pd.to_datetime(row['hosp_disch_time'])
            los_hours = (disch - adm).total_seconds() / 3600
            return round(los_hours, 2) if los_hours >= 0 else None
    except:
        pass
    return None


def format_patient_encounter(row):
    """Format a patient encounter record into a structured clinical narrative"""
    
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
    
    # Discharge disposition
    disch_disp_code = row.get('disch_disp_c')
    disch_disp_map = {1.0: 'Home', 2.0: 'Transfer', 3.0: 'Skilled Nursing Facility',
                      4.0: 'Expired', 5.0: 'Left Against Medical Advice', 6.0: 'Home Health'}
    discharge_disposition = disch_disp_map.get(disch_disp_code, 'Unknown')
    
    # Time information
    admission_time = row.get('hosp_admsn_time', 'Unknown')
    discharge_time = row.get('hosp_disch_time', 'Unknown')
    arrival_time = row.get('adt_arrival_time', 'Unknown')
    
    # Calculate length of stay
    los = calculate_length_of_stay(row)
    los_text = f"{los} hours" if los else "Unknown"
    if los and los >= 24:
        los_text = f"{los} hours ({los/24:.1f} days)"
    
    # ED visit
    ed_visit = row.get('ed_visit_yn', 'N')
    ed_text = "Yes" if ed_visit == 'Y' else "No"
    
    # Providers
    admission_prov = row.get('admission_prov', 'Not recorded')
    discharge_prov = row.get('discharge_prov', 'Not recorded')
    
    # Create structured narrative
    narrative = f"""PATIENT ENCOUNTER SUMMARY

Encounter ID: {encounter_id}
Patient ID: {patient_id[:8]}...

ADMISSION DETAILS:
- Admission Type: {admission_type}
- ED Visit: {ed_text}
- Arrival Time: {arrival_time}
- Admission Time: {admission_time}
- Discharge Time: {discharge_time}
- Length of Stay: {los_text}

CLINICAL SERVICE:
- Department Specialty: {specialty}
- Hospital Service: {hospital_service}
- Service Area: {service_area}

PROVIDERS:
- Admitting Provider: {admission_prov}
- Discharge Provider: {discharge_prov}

OUTCOME:
- Discharge Disposition: {discharge_disposition}
"""
    
    return narrative


def create_clinical_prompt(encounter_text, task="analysis"):
    """Create different types of clinical prompts based on the task"""
    
    prompts = {
        "analysis": f"""{encounter_text}

Based on this patient encounter summary, please provide:
1. A brief clinical analysis of the admission pattern
2. Notable observations about the care delivery (length of stay, department, etc.)
3. Any potential areas of concern or interest from a quality improvement perspective""",
        
        "prediction": f"""{encounter_text}

Based on this encounter data, please analyze:
1. Factors that may have influenced the length of stay
2. Whether the discharge disposition seems appropriate given the admission type
3. Any patterns that might predict readmission risk""",
        
        "risk": f"""{encounter_text}

From a clinical risk management perspective, please evaluate:
1. High-risk indicators in this encounter
2. Appropriateness of care transitions
3. Potential quality or safety concerns""",
        
        "recommendations": f"""{encounter_text}

Based on this encounter, please provide:
1. Recommendations for similar cases
2. Care coordination considerations
3. Opportunities for improving patient flow or outcomes""",
        
        "summary": f"""{encounter_text}

Please provide a concise clinical summary highlighting the key aspects of this inpatient encounter."""
    }
    
    return prompts.get(task, prompts["analysis"])


def prepare_inference_data(
    data_path,
    num_samples=5,
    task="analysis",
    filter_criteria=None,
    output_format="text"
):
    """
    Prepare formatted prompts from the inpatient data
    
    Args:
        data_path: Path to CSV file
        num_samples: Number of encounters to prepare
        task: Type of analysis prompt
        filter_criteria: Dict of filters
        output_format: 'text', 'json', or 'both'
    
    Returns:
        List of prepared prompts
    """
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total encounters: {len(df):,}")
    
    # Apply filters
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
    
    print(f"\nPreparing {len(df_sample)} encounters for inference...")
    print(f"Task: {task}")
    print("=" * 80)
    
    prepared_data = []
    
    for idx, (_, row) in enumerate(df_sample.iterrows(), 1):
        print(f"\n{'='*80}")
        print(f"ENCOUNTER {idx}/{len(df_sample)}")
        print(f"{'='*80}")
        
        # Format encounter
        encounter_text = format_patient_encounter(row)
        prompt = create_clinical_prompt(encounter_text, task=task)
        
        print(prompt)
        
        prepared_data.append({
            'encounter_id': str(row.get('pat_enc_csn_id')),
            'patient_id': str(row.get('osler_id'))[:8] + '...',
            'specialty': str(row.get('dep_speciality')),
            'admission_type': row.get('hosp_admsn_type_c'),
            'los_hours': calculate_length_of_stay(row),
            'encounter_summary': encounter_text,
            'full_prompt': prompt,
            'task': task
        })
    
    # Save outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format in ['text', 'both']:
        text_file = f'prepared_prompts_{timestamp}.txt'
        with open(text_file, 'w') as f:
            f.write(f"Prepared Prompts for Medical LLM Inference\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task: {task}\n")
            f.write(f"Total encounters: {len(prepared_data)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, data in enumerate(prepared_data, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"ENCOUNTER {i}\n")
                f.write(f"{'='*80}\n\n")
                f.write(data['full_prompt'])
                f.write("\n\n")
        
        print(f"\n✓ Text file saved: {text_file}")
    
    if output_format in ['json', 'both']:
        json_file = f'prepared_prompts_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(prepared_data, f, indent=2)
        
        print(f"✓ JSON file saved: {json_file}")
    
    print(f"\n{'='*80}")
    print(f"✓ Prepared {len(prepared_data)} prompts for inference")
    print(f"{'='*80}\n")
    
    return prepared_data


def main():
    """Main execution"""
    
    print("="*80)
    print("Inpatient Data Formatter for Medical LLM")
    print("="*80)
    
    # Configuration
    data_path = Path(__file__).parent / "datasets" / "dbo.accm_inpatient.csv"
    
    if not data_path.exists():
        print(f"✗ Error: Data file not found at {data_path}")
        return
    
    config = {
        'num_samples': 5,
        'task': 'analysis',  # Options: 'analysis', 'prediction', 'risk', 'recommendations', 'summary'
        'filter_criteria': None,  # Example: {'dep_speciality': 'Emergency Medicine'}
        'output_format': 'both'  # Options: 'text', 'json', 'both'
    }
    
    print(f"\nConfiguration:")
    print(f"  - Data path: {data_path}")
    print(f"  - Number of samples: {config['num_samples']}")
    print(f"  - Task type: {config['task']}")
    print(f"  - Filters: {config['filter_criteria']}")
    print(f"  - Output format: {config['output_format']}")
    
    # Prepare data
    prepared_data = prepare_inference_data(
        data_path=data_path,
        num_samples=config['num_samples'],
        task=config['task'],
        filter_criteria=config['filter_criteria'],
        output_format=config['output_format']
    )
    
    print(f"\n✓ Data preparation completed!")
    print(f"✓ {len(prepared_data)} prompts ready for model inference")


if __name__ == "__main__":
    main()
