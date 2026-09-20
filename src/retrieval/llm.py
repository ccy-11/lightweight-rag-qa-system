# -*- coding: utf-8 -*-
"""Qwen-1.8B-Chat LLM wrapper with 4-bit quantization support."""
import logging
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen-1.8B-Chat"


class QwenLLM:
    """Qwen-1.8B-Chat with greedy decoding for reproducible outputs.

    Attempts 4-bit quantization (bitsandbytes) on GPU. Falls back to
    8-bit on GPU or fp32 on CPU if quantization is unavailable.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        use_quantization: bool = True,
        cache_dir: str = "./model_cache",
    ):
        self.model_id = model_id
        self.device = device
        self.use_quantization = use_quantization
        self.cache_dir = cache_dir

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = self._load_model(
            model_id=model_id,
            device=device,
            use_quantization=use_quantization,
            cache_dir=cache_dir,
        )

    def _load_model(self, model_id, device, use_quantization, cache_dir):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # Try 4-bit quantization first
        if use_quantization and device == "cuda":
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                logger.info("Loaded Qwen-1.8B-Chat in 4-bit (nf4) mode")
                return model
            except Exception as exc:
                logger.warning(
                    "4-bit quantization failed (%s), falling back to 8-bit", exc
                )

        # Fallback: 8-bit on GPU
        if device == "cuda":
            try:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                logger.info("Loaded Qwen-1.8B-Chat in 8-bit mode (fallback)")
                return model
            except Exception as exc:
                logger.warning("8-bit failed (%s), falling back to fp32 CPU", exc)

        # Fallback: fp32 CPU
        logger.warning("Loading Qwen-1.8B-Chat in fp32 on CPU (slow but reliable)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
        model.eval()
        return model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response using greedy decoding (reproducible).

        Returns the generated text (excluding the prompt itself).
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device if hasattr(self.model, "device") else "cpu")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy decoding for reproducibility
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @property
    def generation_time(self) -> float:
        return getattr(self, "_last_gen_time", 0.0)
