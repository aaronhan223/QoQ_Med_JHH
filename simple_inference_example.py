#!/usr/bin/env python3
"""
Simple QoQ-Med-VL-7B Inference Example

A minimal example for quick testing.
Based on the official README at: https://huggingface.co/ddvd233/QoQ-Med-VL-7B
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ============================================================================
# Step 1: Load the model and processor
# ============================================================================
print("Loading QoQ-Med-VL-7B model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "ddvd233/QoQ-Med-VL-7B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Requires Ampere (A100) or newer GPUs
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("ddvd233/QoQ-Med-VL-7B")

print("Model loaded successfully!")


# ============================================================================
# Step 2: Prepare your input
# ============================================================================
# Replace with your actual image path
image_path = "/projects/LCICM/Xing_Scripts/QoQ_Med_JHH/datasets/sample_cxr.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text",
                "text": "Describe this medical image and identify any abnormalities."
            },
        ],
    }
]


# ============================================================================
# Step 3: Process the input
# ============================================================================
print("Processing input...")

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


# ============================================================================
# Step 4: Generate output
# ============================================================================
print("Generating response...")

generated_ids = model.generate(**inputs, max_new_tokens=1024)

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)


# ============================================================================
# Step 5: Display the result
# ============================================================================
print("\n" + "="*70)
print("MODEL RESPONSE:")
print(output_text[0])
print("="*70)
