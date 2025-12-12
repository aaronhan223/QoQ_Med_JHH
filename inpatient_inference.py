#!/usr/bin/env python3
"""
QoQ-Med-VL-7B Inference for Inpatient Data - MCQ-5: Length of Stay Category

This script processes inpatient encounter data from dbo.accm_inpatient.csv
and uses the QoQ-Med-VL-7B model to predict Length of Stay categories.

MCQ-5 from designed_qa_questions.txt:
"Based on the patient's admission and discharge times, which category best
describes their hospital length of stay?
A. Short stay (0-2 days)
B. Moderate stay (3-7 days)
C. Extended stay (8-14 days)
D. Long-term stay (>14 days)"
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
import json
import uuid

# ============================================================================
# Utility Functions
# ============================================================================

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
    suitable for Length of Stay prediction (MCQ-5).
    
    NOTE: Does NOT include discharge information, as this is for prediction.
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


# ============================================================================
# Main Inference Function
# ============================================================================

def run_inference_on_encounters(
    data_path,
    model,
    processor,
    num_samples=5,
    filter_criteria=None,
    output_file="los_inference_results.jsonl",
    exclude_missing_discharge=True
):
    """
    Run inference on patient encounters for Length of Stay prediction (MCQ-5).
    
    Args:
        data_path: Path to the CSV file
        model: Loaded model
        processor: Loaded processor
        num_samples: Number of encounters to analyze
        filter_criteria: Dict of column:value pairs to filter data
        output_file: Path to save results (JSONL format)
        exclude_missing_discharge: Only use completed encounters
    """
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Total encounters in dataset: {len(df):,}")
    
    # Filter out encounters without discharge times (can't calculate actual LOS)
    if exclude_missing_discharge:
        initial_count = len(df)
        df = df[pd.notna(df['hosp_disch_time']) & pd.notna(df['hosp_admsn_time'])]
        print(f"Encounters with complete admission/discharge times: {len(df):,} (filtered {initial_count - len(df):,})")
    
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
    
    print(f"\nAnalyzing {len(df_sample)} encounters for Length of Stay prediction...")
    print(f"Task: MCQ-5 - Length of Stay Category")
    print("=" * 80)
    
    results = []
    correct_predictions = 0
    
    for idx, (_, row) in enumerate(df_sample.iterrows(), 1):
        print(f"\n{'='*80}")
        print(f"ENCOUNTER {idx}/{len(df_sample)}")
        print(f"{'='*80}")
        
        # Calculate actual LOS
        los_hours, los_days = calculate_length_of_stay(row)
        
        if los_days is None:
            print(f"Skipping encounter - unable to calculate LOS")
            continue
        
        # Get ground truth category
        true_category, true_description = categorize_los(los_days)
        
        # Format the encounter data (without discharge info)
        encounter_text = format_patient_encounter(row)
        print(f"\n{encounter_text}")
        
        # Create LOS prediction prompt
        prompt = create_los_qa_prompt(encounter_text)
        
        print(f"\n{'='*80}")
        print(f"QUESTION:")
        print(f"{'='*80}")
        print(prompt)
        print(f"\nGround Truth: {true_category} - {true_description}")
        print(f"Actual LOS: {los_days:.2f} days ({los_hours:.2f} hours)")
        
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
        print(f"\nGenerating model prediction...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,  # Shorter for MCQ
                temperature=0.1,  # Lower temperature for more deterministic MCQ answers
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
        
        # Extract predicted category (A, B, C, or D)
        predicted_category = None
        output_upper = output_text.upper().strip()
        
        # Try to extract the answer
        for choice in ['A', 'B', 'C', 'D']:
            if output_upper.startswith(choice) or f"ANSWER IS {choice}" in output_upper or f"ANSWER: {choice}" in output_upper:
                predicted_category = choice
                break
        
        # If no clear match, look for the first occurrence
        if predicted_category is None:
            for choice in ['A', 'B', 'C', 'D']:
                if choice in output_upper[:50]:  # Look in first 50 chars
                    predicted_category = choice
                    break
        
        # Check if prediction is correct
        is_correct = (predicted_category == true_category) if predicted_category else False
        if is_correct:
            correct_predictions += 1
        
        # Display result
        print("\n" + "-" * 80)
        print("MODEL PREDICTION:")
        print("-" * 80)
        print(f"Raw Output: {output_text}")
        print(f"Extracted Answer: {predicted_category if predicted_category else 'UNABLE TO EXTRACT'}")
        print(f"Ground Truth: {true_category}")
        print(f"Correct: {'✓ YES' if is_correct else '✗ NO'}")
        print("-" * 80)
        
        # Store result in JSONL format (following GitHub repo structure)
        result = {
            'qa_id': qa_id(),
            'qa_type': 6,
            'format': 'Multiple Choice',
            'question': prompt,
            'images': [],
            'time-series': [],
            'choices': [
                'A. Short stay (0-2 days)',
                'B. Moderate stay (3-7 days)',
                'C. Extended stay (8-14 days)',
                'D. Long-term stay (>14 days)'
            ],
            'correct_choice': true_category,
            'answer': true_category,
            'generated_answer': output_text,
            'extracted_prediction': predicted_category,
            'is_correct': is_correct,
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
        
        results.append(result)
        
        # Save result incrementally (append to JSONL)
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    # Calculate accuracy
    accuracy = (correct_predictions / len(results) * 100) if results else 0
    
    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETED")
    print(f"{'='*80}")
    print(f"Total encounters analyzed: {len(results)}")
    print(f"Correct predictions: {correct_predictions}/{len(results)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Also save a summary file
    summary_file = output_file.replace('.jsonl', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"QoQ-Med-VL-7B Length of Stay Prediction Results (MCQ-5)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total encounters: {len(results)}\n")
        f.write(f"Correct predictions: {correct_predictions}/{len(results)}\n")
        f.write(f"Accuracy: {accuracy:.1f}%\n")
        f.write("=" * 80 + "\n\n")
        
        # Category-wise breakdown
        category_stats = {'A': {'total': 0, 'correct': 0}, 
                         'B': {'total': 0, 'correct': 0},
                         'C': {'total': 0, 'correct': 0},
                         'D': {'total': 0, 'correct': 0}}
        
        for result in results:
            cat = result['correct_choice']
            category_stats[cat]['total'] += 1
            if result['is_correct']:
                category_stats[cat]['correct'] += 1
        
        f.write("Category-wise Performance:\n")
        f.write("-" * 80 + "\n")
        for cat in ['A', 'B', 'C', 'D']:
            total = category_stats[cat]['total']
            correct = category_stats[cat]['correct']
            acc = (correct / total * 100) if total > 0 else 0
            cat_desc = {
                'A': 'Short stay (0-2 days)',
                'B': 'Moderate stay (3-7 days)',
                'C': 'Extended stay (8-14 days)',
                'D': 'Long-term stay (>14 days)'
            }
            f.write(f"{cat}. {cat_desc[cat]}: {correct}/{total} ({acc:.1f}%)\n")
    
    print(f"Summary saved to: {summary_file}\n")
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("QoQ-Med-VL-7B Length of Stay Prediction (MCQ-5)")
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
        'num_samples': 10,  # Number of encounters to analyze
        'filter_criteria': None,  # Example: {'dep_speciality': 'Emergency Medicine'}
        'output_file': 'los_inference_results.jsonl',
        'exclude_missing_discharge': True  # Only use completed encounters
    }
    
    print(f"  - Number of samples: {config['num_samples']}")
    print(f"  - Analysis task: MCQ-5 (Length of Stay Category)")
    print(f"  - Filters: {config['filter_criteria']}")
    print(f"  - Output file: {config['output_file']}")
    print(f"  - Exclude incomplete encounters: {config['exclude_missing_discharge']}")
    
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
            filter_criteria=config['filter_criteria'],
            output_file=config['output_file'],
            exclude_missing_discharge=config['exclude_missing_discharge']
        )
        
        print("\n✓ Inference completed successfully!")
        print(f"✓ Analyzed {len(results)} encounters")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
