# QoQ-Med-VL-7B Inference Guide

This guide shows how to use the QoQ-Med-VL-7B model for inference on medical images.

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_inference.txt
```

Or install minimal dependencies:

```bash
pip install transformers qwen-vl-utils torch pillow
```

### 2. (Optional) Install Flash Attention

For better performance, install flash attention (requires CUDA):

```bash
pip install flash-attn
```

## Usage Options

### Option 1: Simple Script (Recommended for Quick Testing)

Use `simple_inference_example.py` for a straightforward example:

1. Edit the script and replace the image path:
   ```python
   image_path = "path/to/your/medical/image.jpg"
   ```

2. Run the script:
   ```bash
   python simple_inference_example.py
   ```

### Option 2: CLI Script (Recommended for Batch Processing)

Use `inference_example.py` for command-line usage with more options:

```bash
# Basic usage
python inference_example.py --image_path path/to/image.jpg

# Custom question
python inference_example.py \
    --image_path path/to/image.jpg \
    --question "What abnormalities are present in this chest X-ray?"

# Adjust generation length
python inference_example.py \
    --image_path path/to/image.jpg \
    --max_tokens 1024

# Use different model (32B version)
python inference_example.py \
    --image_path path/to/image.jpg \
    --model_name ddvd233/QoQ-Med-VL-32B

# Disable flash attention (if not available)
python inference_example.py \
    --image_path path/to/image.jpg \
    --no_flash_attention
```

### Option 3: Python Interactive Usage

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "ddvd233/QoQ-Med-VL-7B",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("ddvd233/QoQ-Med-VL-7B")

# Prepare message
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "your_image.jpg"},
        {"type": "text", "text": "Describe this medical image."}
    ]
}]

# Process and generate
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
print(output)
```

## Example Prompts for Medical Images

### General Analysis
```
"Describe this medical image and identify any abnormalities."
```

### Specific Findings
```
"What pathological findings are present in this image? Please provide a detailed analysis."
```

### Diagnosis
```
"Based on this medical image, what is the most likely diagnosis? Explain your reasoning."
```

### Comparison
```
"Compare the left and right sides of this image. Are there any asymmetries or differences?"
```

### Region of Interest
```
"Identify and describe the abnormal regions in this image. Provide bounding box coordinates if possible."
```

## Model Variants

| Model | HuggingFace Path | Avg. Accuracy | Memory Required |
|-------|------------------|---------------|-----------------|
| QoQ-Med-VL-7B | `ddvd233/QoQ-Med-VL-7B` | 68.6% | ~16 GB VRAM |
| QoQ-Med-VL-32B | `ddvd233/QoQ-Med-VL-32B` | 70.7% | ~64 GB VRAM |

## Hardware Requirements

### Minimum (7B Model)
- GPU: NVIDIA GPU with 16 GB VRAM (e.g., RTX 4090, A100)
- RAM: 32 GB system RAM
- Storage: ~15 GB for model weights

### Recommended (7B Model)
- GPU: NVIDIA GPU with 24+ GB VRAM (e.g., RTX 4090, A6000)
- RAM: 64 GB system RAM
- CUDA: 11.8 or higher
- Flash Attention support for faster inference

### For 32B Model
- GPU: NVIDIA GPU with 80 GB VRAM (e.g., A100-80GB) or multi-GPU setup
- RAM: 128 GB system RAM

## Troubleshooting

### Flash Attention Not Available
If you get an error about flash attention:
```bash
# Remove flash_attention_2 from the script or use --no_flash_attention flag
python inference_example.py --image_path image.jpg --no_flash_attention
```

### Out of Memory (OOM)
If you run out of GPU memory:

1. Use smaller max_pixels:
   ```python
   processor = AutoProcessor.from_pretrained(
       "ddvd233/QoQ-Med-VL-7B",
       min_pixels=128 * 28 * 28,
       max_pixels=512 * 28 * 28  # Reduced from 1280
   )
   ```

2. Reduce max_new_tokens:
   ```bash
   python inference_example.py --image_path image.jpg --max_tokens 256
   ```

3. Use CPU offloading:
   ```python
   model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
       "ddvd233/QoQ-Med-VL-7B",
       torch_dtype="auto",
       device_map="auto",
       offload_folder="offload"  # Offload to CPU when needed
   )
   ```

### Image Format Issues
Supported image formats: JPG, PNG, TIFF, DICOM (with preprocessing)

For DICOM files, convert to PNG first:
```python
import pydicom
from PIL import Image
import numpy as np

# Load DICOM
ds = pydicom.dcmread("image.dcm")
pixel_array = ds.pixel_array

# Normalize and convert
pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
img = Image.fromarray(pixel_array.astype(np.uint8))
img.save("image.png")
```

## Performance Tips

1. **Use Flash Attention**: Install `flash-attn` for 2-4x speedup
2. **Use BF16**: Set `torch_dtype=torch.bfloat16` for better performance
3. **Batch Processing**: Process multiple images in a batch when possible
4. **Adjust Visual Tokens**: Balance detail vs. speed by tuning `min_pixels` and `max_pixels`

## Limitations

⚠️ **Important**: This model is for **research purposes only**. Not approved for clinical use without extensive real-world testing and regulatory approval.

### Current Limitations
- No time-series (ECG) support via standard HuggingFace loading (see README note)
- Limited to medical images (2D/3D) and text
- English language primarily
- Performance varies by clinical domain

## Additional Resources

- **Model Card**: https://huggingface.co/ddvd233/QoQ-Med-VL-7B
- **Paper**: https://arxiv.org/abs/2506.00711
- **GitHub**: https://github.com/yourusername/QoQ_Med
- **GGUF Versions**: For use with LM Studio, Ollama, etc.
  - https://huggingface.co/mradermacher/QoQ-Med-VL-7B-GGUF

## Example Output

```
Input: "Describe this chest X-ray and identify any abnormalities."

Output: "This is a frontal chest X-ray showing the lungs, heart, and surrounding
structures. The image reveals several notable findings:

1. Cardiomegaly: The cardiac silhouette appears enlarged, with a
   cardiothoracic ratio exceeding 0.5, suggesting possible cardiac
   enlargement.

2. Pulmonary Edema: There is evidence of bilateral interstitial opacity
   with Kerley B lines visible in the lower lung fields, consistent with
   pulmonary edema.

3. Pleural Effusion: A small right-sided pleural effusion is visible,
   indicated by blunting of the right costophrenic angle.

The findings suggest congestive heart failure as a likely diagnosis.
Clinical correlation and comparison with prior imaging would be beneficial."
```
