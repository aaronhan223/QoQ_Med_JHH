#!/usr/bin/env python3
"""
QoQ-Med-VL-7B Inference for Inpatient Data

This script processes inpatient encounter data from dbo.accm_inpatient.csv
and uses the QoQ-Med-VL-7B model to analyze clinical scenarios and provide insights.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path


# ============================================================================
# Utility Functions
# ============================================================================

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
    """
    Format a patient encounter record into a structured clinical narrative
    suitable for the medical language model.
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
    """
    Create different types of clinical prompts based on the task.
    
    Args:
        encounter_text: Formatted patient encounter text
        task: Type of analysis ("analysis", "prediction", "risk", "recommendations")
    """
    
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


# ============================================================================
# Main Inference Function
# ============================================================================

def run_inference_on_encounters(
    data_path,
    model,
    processor,
    num_samples=5,
    task="analysis",
    filter_criteria=None,
    output_file="inpatient_analysis_results.txt"
):
    """
    Run inference on patient encounters from the CSV file.
    
    Args:
        data_path: Path to the CSV file
        model: Loaded model
        processor: Loaded processor
        num_samples: Number of encounters to analyze
        task: Type of analysis to perform
        filter_criteria: Dict of column:value pairs to filter data
        output_file: Path to save results
    """
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total encounters in dataset: {len(df):,}")
    
    # Apply filters if specified
    if filter_criteria:
        print(f"\nApplying filters: {filter_criteria}")
        for col, value in filter_criteria.items():
            if col in df.columns:
                df = df[df[col] == value]
        print(f"Encounters after filtering: {len(df):,}")
    
    # Sample encounters
    if len(df) > num_samples:
        # Sample diverse cases
        df_sample = df.sample(n=num_samples, random_state=42)
    else:
        df_sample = df.head(num_samples)
    
    print(f"\nAnalyzing {len(df_sample)} encounters...")
    print(f"Task: {task}")
    print("=" * 80)
    
    results = []
    
    for idx, (_, row) in enumerate(df_sample.iterrows(), 1):
        print(f"\n{'='*80}")
        print(f"ENCOUNTER {idx}/{len(df_sample)}")
        print(f"{'='*80}")
        
        # Format the encounter data
        encounter_text = format_patient_encounter(row)
        print(encounter_text)
        
        # Create prompt
        prompt = create_clinical_prompt(encounter_text, task=task)
        
        # Prepare messages (text-only, no image)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
        
        # Process input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda")
        
        # Generate response
        print("\nGenerating model response...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Display result
        print("\n" + "-" * 80)
        print("MODEL ANALYSIS:")
        print("-" * 80)
        print(output_text)
        print("-" * 80)
        
        # Store result
        results.append({
            'encounter_id': row.get('pat_enc_csn_id'),
            'specialty': row.get('dep_speciality'),
            'los_hours': calculate_length_of_stay(row),
            'encounter_summary': encounter_text,
            'model_analysis': output_text
        })
    
    # Save results to file
    with open(output_file, 'w') as f:
        f.write(f"QoQ-Med-VL-7B Inpatient Analysis Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Total encounters analyzed: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"ENCOUNTER {i}\n")
            f.write(f"{'='*80}\n\n")
            f.write(result['encounter_summary'])
            f.write(f"\n\nMODEL ANALYSIS:\n{'-'*80}\n")
            f.write(result['model_analysis'])
            f.write(f"\n\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("QoQ-Med-VL-7B Inpatient Data Inference")
    print("="*80)
    
    # ========================================================================
    # Step 1: Load the model and processor
    # ========================================================================
    print("\n[1/4] Loading QoQ-Med-VL-7B model...")
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "ddvd233/QoQ-Med-VL-7B",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        
        processor = AutoProcessor.from_pretrained("ddvd233/QoQ-Med-VL-7B")
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return
    
    # ========================================================================
    # Step 2: Define data path and parameters
    # ========================================================================
    print("\n[2/4] Setting up data paths and parameters...")
    
    data_path = Path(__file__).parent / "datasets" / "dbo.accm_inpatient.csv"
    
    if not data_path.exists():
        print(f"✗ Error: Data file not found at {data_path}")
        return
    
    print(f"✓ Data file located: {data_path}")
    
    # ========================================================================
    # Step 3: Configure analysis parameters
    # ========================================================================
    print("\n[3/4] Configuring analysis parameters...")
    
    # You can customize these parameters:
    config = {
        'num_samples': 5,  # Number of encounters to analyze
        'task': 'analysis',  # Options: 'analysis', 'prediction', 'risk', 'recommendations', 'summary'
        'filter_criteria': None,  # Example: {'dep_speciality': 'Emergency Medicine'}
        'output_file': 'inpatient_analysis_results.txt'
    }
    
    print(f"  - Number of samples: {config['num_samples']}")
    print(f"  - Analysis task: {config['task']}")
    print(f"  - Filters: {config['filter_criteria']}")
    print(f"  - Output file: {config['output_file']}")
    
    # ========================================================================
    # Step 4: Run inference
    # ========================================================================
    print("\n[4/4] Running inference on patient encounters...")
    
    try:
        results = run_inference_on_encounters(
            data_path=data_path,
            model=model,
            processor=processor,
            num_samples=config['num_samples'],
            task=config['task'],
            filter_criteria=config['filter_criteria'],
            output_file=config['output_file']
        )
        
        print("\n✓ Analysis completed successfully!")
        print(f"✓ Analyzed {len(results)} encounters")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
