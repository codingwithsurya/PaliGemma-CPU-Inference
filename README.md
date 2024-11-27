# PaliGemma Inference Pipeline

Replication and efficient inference for the PaliGemma model, a state-of-the-art vision-language model combining a SigLIP vision encoder with a Gemma language decoder. Optimized for both CPU and Apple Silicon (MPS) devices.

## Features

- Efficient inference with automatic device selection (MPS/CPU)
- Advanced optimizations for both CPU and MPS:
  - **MPS (Apple Silicon) Optimizations:**
    - Automatic mixed precision (float16)
    - Metal-specific memory layout optimizations
    - Optimized memory format for Metal Performance Shaders
    - Automatic fallback to CPU if MPS is unavailable
  
  - **CPU Optimizations:**
    - Dynamic quantization of linear layers (int8)
    - Optimized memory layout
    - CPU-specific automatic mixed precision
    - Inference mode optimizations

- Support for image and text inputs
- Customizable inference parameters
- Efficient memory management with proper context handling

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/codingwithsurya/PaliGemma-Inference-Pipeline.git
   cd paligemma-inference
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face token:
   - Copy `.env.template` to `.env`:
     ```bash
     cp .env.template .env
     ```
   - Edit `.env` and replace `your_token_here` with your Hugging Face token
   - You can get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

## Usage

Run inference using the following command:
```bash
# For MPS (Apple Silicon GPU) - Recommended for Mac users
python inference.py --prompt "Describe this image" --image_file_path "path/to/your/image.jpg"

# For CPU-only inference
python inference.py --prompt "Describe this image" --image_file_path "path/to/your/image.jpg" --only_cpu

# Using the sample dog image
python inference.py --prompt Describe this image in detail --image_file_path dog.jpg --max_tokens_to_generate 300

```

### Parameters

- `--prompt`: The text prompt for the model
- `--image_file_path`: Path to the input image
- `--only_cpu`: Flag to force CPU-only inference (default: False, will use MPS if available)
- `--max_tokens_to_generate`: Maximum number of tokens to generate (default: 300)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Top-p sampling parameter (default: 0.9)

## Technical Details

This project leverages several advanced deep learning concepts and optimizations:

- **Architecture:**
  - Vision Transformer (ViT) for image processing
  - Transformer architecture with multi-head attention
  - Rotary positional embeddings
  - Grouped query attention

- **Optimizations:**
  - Device-specific optimizations (MPS/CPU)
  - Automatic mixed precision training
  - Dynamic quantization
  - Optimized memory layouts
  - Inference mode optimizations
  - Proper context management for optimal performance

- **Memory Management:**
  - Efficient tensor operations
  - Automatic device selection and fallback
  - Optimized memory formats for each device

## Performance

The implementation automatically selects the best available device:
- On Apple Silicon Macs: Uses MPS (Metal Performance Shaders) for GPU acceleration
- On other systems: Falls back to optimized CPU inference with quantization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Umar Jamil's VLM Tutorial](https://www.youtube.com/watch?v=vAmKB7iPkWw)
- [PaLiGemma Model](https://huggingface.co/docs/transformers/main/en/model_doc/paligemma)
- [PyTorch](https://pytorch.org/)
