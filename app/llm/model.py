import os
from typing import Dict, List, Optional, Union
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    A wrapper for running local LLMs using the Hugging Face Transformers library.
    Optimized for creative prompt expansion and interpretation.
    """

    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        device_map: str = "auto",
        torch_dtype=None,
    ):
        """
        Initialize the local LLM.

        Args:
            model_path: Path to model or HuggingFace model ID
            device_map: Device mapping strategy (default: "auto")
            torch_dtype: Torch data type (default: bfloat16 if available, otherwise float16)
        """
        self.model_path = model_path
        self.device_map = device_map

        if torch_dtype is None:
            # Set default dtype based on device
            if device_map == "mps":
                # Apple Silicon uses float16
                self.torch_dtype = torch.float16
            elif (
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            ):
                # Modern NVIDIA GPUs use bfloat16
                self.torch_dtype = torch.bfloat16
            else:
                # Default to float16 for other cases
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch_dtype

        logger.info(f"Loading LLM from {model_path}")
        logger.info(f"Using device: {device_map}, dtype: {self.torch_dtype}")

        try:
            # Load model and tokenizer directly instead of using pipeline
            # This gives us more control over the configuration

            # First, load and fix the config
            config = AutoConfig.from_pretrained(model_path)

            # Fix the rope_scaling issue for Llama models
            if hasattr(config, "rope_scaling") and isinstance(
                config.rope_scaling, dict
            ):
                # Ensure the type key exists and is set to linear
                config.rope_scaling["type"] = "linear"
                logger.info("Fixed rope_scaling configuration with type=linear")
            elif not hasattr(config, "rope_scaling"):
                # If no rope_scaling exists, add a basic one
                config.rope_scaling = {"type": "linear", "factor": 1.0}
                logger.info("Added default rope_scaling configuration")

            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load the model with our fixed config
            if device_map == "mps":
                # For Apple Silicon, load to device directly
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=self.torch_dtype,
                    device_map={"": "mps"},  # Map all modules to MPS device
                )
            else:
                # For other devices, use the device_map parameter
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map,
                )

            # Create the pipeline with our pre-loaded model and tokenizer
            self.pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, framework="pt"
            )

            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text based on a prompt with the local LLM.

        Args:
            prompt: The user prompt to generate from
            system_prompt: Optional system prompt to guide the generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter

        Returns:
            The generated text
        """
        # Format messages for chat-style models
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Generating with prompt: {prompt[:100]}...")

        try:
            # Generate response using the pipeline
            outputs = self.pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

            # Extract the assistant's response
            response = outputs[0]["generated_text"][-1]["content"]
            return response

        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return ""

    def expand_creative_prompt(self, prompt: str) -> str:
        """
        Specifically designed to expand a user prompt into a more detailed,
        creative description suitable for image generation.

        Args:
            prompt: The user's original prompt

        Returns:
            An expanded, detailed creative prompt
        """
        system_prompt = """You are a creative assistant specializing in enhancing text prompts for image and 3D model generation.
When given a simple prompt, expand it with rich, vivid details about:
- Visual elements and composition
- Lighting, colors, and atmosphere
- Style, mood, and artistic influence
- Textures and materials
- Perspective and framing

Keep your response focused only on the enhanced visual description without explanations or comments.
Limit to 3-4 sentences maximum, ensuring it's concise yet richly detailed."""

        # Generate the expanded prompt
        expanded = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=256,
            temperature=0.8,  # Slightly higher temperature for creativity
        )

        logger.info(f"Expanded prompt: {expanded[:100]}...")
        return expanded


def get_llm_instance(model_path: Optional[str] = None) -> LocalLLM:
    """
    Factory function to get a LocalLLM instance with default settings.

    Args:
        model_path: Optional path to model or HuggingFace model ID

    Returns:
        A LocalLLM instance
    """
    # If model path not provided, first check for MODEL_PATH, then MODEL_ID from environment
    if not model_path:
        model_path = os.environ.get("MODEL_PATH") or os.environ.get(
            "MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct"
        )

    # Check if the provided path is a local directory
    if os.path.isdir(model_path):
        logger.info(f"Using local model directory: {model_path}")
    else:
        logger.info(f"Using model ID from Hugging Face: {model_path}")

    # Check available device backends
    device_map = "auto"
    torch_dtype = None

    # Check for Apple Silicon (M1/M2/M3) MPS support
    if torch.backends.mps.is_available():
        logger.info(
            "Apple Silicon MPS is available. Using MPS backend for accelerated inference."
        )
        device_map = "mps"
        torch_dtype = torch.float16
    # Otherwise check if CUDA is available
    elif torch.cuda.is_available():
        logger.info(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
        if torch.cuda.get_device_capability()[0] >= 8:
            # For Ampere architecture (30XX, A100, etc.) use bfloat16
            torch_dtype = torch.bfloat16
        else:
            # For older architectures use float16
            torch_dtype = torch.float16
    else:
        logger.warning(
            "No GPU acceleration available. Using CPU. This may be slow for inference."
        )

    return LocalLLM(
        model_path=model_path, device_map=device_map, torch_dtype=torch_dtype
    )
