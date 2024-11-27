import torch
import torch.nn as nn
from typing import Optional, Dict
import psutil
from contextlib import contextmanager
from PIL import Image

class InferenceOptimizer:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.is_mps = device == "mps"
        self.is_cpu = device == "cpu"
        
    def optimize_model(self):
        """Apply device-specific optimizations"""
        if self.is_cpu:
            return self._apply_cpu_optimizations()
        elif self.is_mps:
            return self._apply_mps_optimizations()
        return self.model
            
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations"""
        # Apply dynamic quantization to linear layers
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Optimize memory layout
        self.model = self.model.to(memory_format=torch.channels_last)
        
        # Enable inference mode
        torch.inference_mode(True)
        
        return self.model
        
    def _apply_mps_optimizations(self):
        """Apply MPS-specific optimizations"""
        if not torch.backends.mps.is_available():
            print("Warning: MPS device requested but not available. Falling back to CPU.")
            self.device = "cpu"
            self.is_mps = False
            self.is_cpu = True
            return self._apply_cpu_optimizations()

        # Convert model to float16 for better MPS performance
        self.model = self.model.half()
        
        # Move model to MPS device
        self.model = self.model.to(self.device)
        
        # Optimize memory layout for Metal
        self.model = self.model.to(memory_format=torch.channels_last)
        
        # Enable inference mode
        torch.inference_mode(True)
        
        return self.model

    @contextmanager
    def inference_context(self):
        """Context manager for optimized inference"""
        if self.is_cpu:
            # For CPU: Use inference mode and channels last
            with torch.inference_mode(), \
                 torch.cpu.amp.autocast(enabled=True), \
                 torch.no_grad():
                yield
        elif self.is_mps:
            # For MPS: Use automatic mixed precision and inference mode
            with torch.inference_mode(), \
                 torch.autocast(device_type='mps', dtype=torch.float16), \
                 torch.no_grad():
                yield

class OptimizedInference:
    def __init__(self, model, processor, device: str):
        self.device = device
        self.optimizer = InferenceOptimizer(model, device)
        self.model = self.optimizer.optimize_model()
        self.processor = processor
        self.setup_caching()

    def setup_caching(self):
        """Setup caching mechanisms"""
        self.kv_cache = {}
        self.cached_embeddings = {}

    def prepare_inputs(self, image_path: str, prompt: str) -> Dict[str, torch.Tensor]:
        """Prepare and optimize inputs"""
        image = Image.open(image_path)
        inputs = self.processor(text=[prompt], images=[image])
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _extract_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features"""
        with self.optimizer.inference_context():
            features = self.model.vision_tower(pixel_values)
            return features

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Optimized text generation"""
        inputs = self.prepare_inputs(image_path, prompt)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return generated_text

    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """Optimized top-p sampling"""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        probs[sorted_indices_to_remove] = 0.0
        probs.div_(probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token