# PaliGemma Inference Pipeline

Replication and efficient CPU-compatible inference for the PaliGemma model, a state-of-the-art vision-language model combining a SigLIP vision encoder with a Gemma language decoder.

## Features

- Efficient CPU-based inference for PaLiGemma
- Dynamic quantization for optimized performance
- Support for image and text inputs
- Customizable inference parameters

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/codingwithsurya/PaliGemma-CPU-Inference.git
   cd paligemma-inference
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run inference using the following command:
```
python inference.py --prompt "Describe this image" --image_file_path "path/to/your/image.jpg" --only_cpu
```

### Parameters

- `--prompt`: The text prompt for the model
- `--image_file_path`: Path to the input image
- `--only_cpu`: Flag to ensure CPU-only inference
- `--max_tokens_to_generate`: Maximum number of tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.7)


## Technical Details

This project leverages several advanced deep learning concepts:

- Transformer architecture with multi-head attention and feed-forward layers
- Vision Transformer (ViT) for image processing
- Contrastive learning techniques inspired by CLIP and SigLIP
- Rotary positional embeddings and grouped query attention
- KV-cache for efficient token generation
- Dynamic quantization for optimized CPU inference

## Acknowledgements

- [Umar Jamil's VLM Tutorial](https://www.youtube.com/watch?v=vAmKB7iPkWw)
- [PaLiGemma Model](https://huggingface.co/docs/transformers/main/en/model_doc/paligemma)
- [PyTorch](https://pytorch.org/)
