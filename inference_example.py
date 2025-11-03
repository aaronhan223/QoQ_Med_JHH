#!/usr/bin/env python3
"""
QoQ-Med-VL-7B Inference Example Script

This script demonstrates how to load the QoQ-Med-VL-7B model from HuggingFace
and perform inference on a medical image.

Requirements:
    pip install transformers qwen-vl-utils torch pillow

Usage:
    python inference_example.py --image_path path/to/medical/image.jpg
"""

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model(model_name="ddvd233/QoQ-Med-VL-7B", use_flash_attention=True):
    """
    Load the QoQ-Med model and processor from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        use_flash_attention: Whether to use flash attention for better performance

    Returns:
        model: The loaded model
        processor: The loaded processor
    """
    print(f"Loading model: {model_name}")

    try:
        if use_flash_attention:
            # Load with flash attention for better performance
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
            print("Model loaded with flash attention")
        else:
            # Load with automatic dtype selection
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            print("Model loaded with automatic dtype")

    except Exception as e:
        print(f"Failed to load with flash attention: {e}")
        print("Falling back to standard loading...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    # Configure visual token range for optimal performance
    # Balance between visual detail and computational cost
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    print("Model and processor loaded successfully!")
    return model, processor


def prepare_messages(image_path, question):
    """
    Prepare the multimodal message format for the model.

    Args:
        image_path: Path to the medical image
        question: The question or prompt about the image

    Returns:
        messages: Formatted messages for the model
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    return messages


def run_inference(model, processor, messages, max_new_tokens=512):
    """
    Run inference on the prepared messages.

    Args:
        model: The loaded QoQ-Med model
        processor: The processor for input/output handling
        messages: Formatted messages with image and text
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        output_text: The generated response
    """
    print("Processing input...")

    # Apply chat template to format the messages
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision information (images/videos)
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to GPU if available
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    print("Generating response...")

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim the input tokens from the generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated tokens to text
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]


def main():
    """Main function to run the inference example."""
    parser = argparse.ArgumentParser(
        description="QoQ-Med-VL-7B Inference Example"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="path/to/your/medical/image.jpg",
        help="Path to the medical image"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe this medical image and provide a detailed analysis of any abnormalities or findings.",
        help="Question or prompt about the image"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ddvd233/QoQ-Med-VL-7B",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable flash attention"
    )

    args = parser.parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU (inference will be slow)")

    # Load model and processor
    model, processor = load_model(
        model_name=args.model_name,
        use_flash_attention=not args.no_flash_attention
    )

    # Prepare messages
    messages = prepare_messages(args.image_path, args.question)

    # Run inference
    print("\n" + "="*70)
    print("QUESTION:")
    print(args.question)
    print("="*70)

    output = run_inference(model, processor, messages, max_new_tokens=args.max_tokens)

    print("\n" + "="*70)
    print("MODEL RESPONSE:")
    print(output)
    print("="*70)


if __name__ == "__main__":
    main()
