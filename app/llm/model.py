import os
from typing import Dict, List, Optional, Union
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
from huggingface_hub import login
from pathlib import Path

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
        token: Optional[str] = None,
    ):
        """
        Initialize the local LLM.

        Args:
            model_path: Path to model or HuggingFace model ID
            device_map: Device mapping strategy (default: "auto")
            torch_dtype: Torch data type (default: bfloat16 if available, otherwise float16)
            token: HuggingFace token for accessing gated models
        """
        self.model_path = model_path
        self.device_map = device_map

        # Handle HuggingFace authentication for remote models
        if not os.path.isdir(model_path):
            auth_token = token or os.environ.get("HF_TOKEN")
            if auth_token:
                logger.info("Authenticating with HuggingFace")
                login(token=auth_token, write_permission=False)

        # Set appropriate dtype if not provided
        if torch_dtype is None:
            if device_map == "mps":
                self.torch_dtype = torch.float16
            elif (
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            ):
                self.torch_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch_dtype

        logger.info(
            f"Loading LLM from {model_path} with device: {device_map}, dtype: {self.torch_dtype}"
        )

        # Load model with appropriate error handling
        try:
            # Load and prepare configuration
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # Fix common rope_scaling issues with Llama models
            if not hasattr(config, "rope_scaling") or not isinstance(
                config.rope_scaling, dict
            ):
                config.rope_scaling = {"type": "linear", "factor": 1.0}
                logger.info("Added default rope_scaling configuration")
            elif (
                isinstance(config.rope_scaling, dict)
                and "type" not in config.rope_scaling
            ):
                config.rope_scaling["type"] = "linear"
                logger.info("Fixed rope_scaling configuration with type=linear")

            # Load tokenizer with fallback options
            tokenizer = self._load_tokenizer(model_path)

            # Load model with appropriate device mapping
            model = self._load_model(model_path, config)

            # Create the pipeline
            self.pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer, framework="pt"
            )
            logger.info("LLM loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _load_tokenizer(self, model_path: str):
        """Helper method to load tokenizer with fallbacks"""
        try:
            logger.info(f"Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            # Set pad token if needed
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")

            # Handle local model with tokenizer config but no tokenizer files
            if os.path.isdir(model_path):
                tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
                if tokenizer_config_path.exists():
                    logger.info(
                        "Found tokenizer config but loading failed, trying fallback tokenizer"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        "meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True
                    )
                    logger.info("Successfully loaded fallback tokenizer")
                    return tokenizer

            # Check for auth errors with remote models
            if "401 Client Error" in str(e) or "403 Client Error" in str(e):
                raise ValueError(
                    f"Authentication error: You need a valid HuggingFace token to access {model_path}. "
                    f"Set the HF_TOKEN environment variable."
                )
            raise

    def _load_model(self, model_path: str, config):
        """Helper method to load model with appropriate device settings"""
        logger.info(f"Loading model with device_map={self.device_map}")

        # Special handling for Apple Silicon
        if self.device_map == "mps":
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=self.torch_dtype,
                device_map={"": "mps"},  # Map all modules to MPS
                trust_remote_code=True,
            )
        else:
            # Standard loading for other devices
            return AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                trust_remote_code=True,
            )

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

    # Check if the path points to a local directory
    is_local_model = os.path.isdir(model_path)

    if is_local_model:
        logger.info(f"Using local model directory: {model_path}")
    else:
        logger.info(f"Using model ID from Hugging Face: {model_path}")

        # For gated models, verify token availability
        if (
            "/" in model_path
            and "meta-llama" in model_path
            and not os.environ.get("HF_TOKEN")
        ):
            logger.warning(
                f"Using gated model '{model_path}' without HF_TOKEN may fail"
            )

    # Determine optimal device settings
    device_map = "auto"
    torch_dtype = None

    # Check available hardware and set appropriate device
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS available. Using MPS acceleration.")
        device_map = "mps"
        torch_dtype = torch.float16
    elif torch.cuda.is_available():
        logger.info(f"CUDA available on {torch.cuda.get_device_name(0)}")
        # Use bfloat16 for newer NVIDIA GPUs (Ampere+)
        torch_dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
    else:
        logger.warning("No GPU acceleration available. Using CPU (slow).")

    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN")

    # Standard loading attempt
    try:
        return LocalLLM(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
    except Exception as e:
        error_str = str(e).lower()

        # Handle tokenizer errors for local models
        if "tokenizer" in error_str and is_local_model:
            logger.warning(f"Tokenizer error with local model: {str(e)}")

            try:
                # Try specific Llama tokenizer for local models
                from transformers import LlamaTokenizer, LlamaForCausalLM

                tokenizer = None
                # First attempt: direct loading with LlamaTokenizer
                try:
                    logger.info("Attempting to load with LlamaTokenizer...")
                    tokenizer = LlamaTokenizer.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    model = LlamaForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )
                except Exception:
                    # Second attempt: load tokenizer from HF and model locally
                    logger.info("Trying with HuggingFace tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        "meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )

                if tokenizer:
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        framework="pt",
                    )
                    llm = LocalLLM(model_path=model_path, device_map=device_map)
                    llm.pipe = pipe
                    logger.info("Successfully loaded model with tokenizer workaround")
                    return llm

            except Exception as workaround_error:
                logger.error(
                    f"All local model workarounds failed: {str(workaround_error)}"
                )

        # Handle authentication errors for remote models
        elif (
            "401" in error_str or "403" in error_str or "authentication" in error_str
        ) and "meta-llama" in model_path:
            logger.warning("Authentication error. Trying fallback open models...")

            # Ordered list of fallback models
            fallback_models = [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "microsoft/Phi-3-mini-4k-instruct",
                "google/gemma-2b-it",
            ]

            # Try each fallback model
            for fallback in fallback_models:
                try:
                    logger.info(f"Attempting to load fallback model: {fallback}")
                    return get_llm_instance(fallback)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback model {fallback} failed: {str(fallback_error)}"
                    )

        # Re-raise the original exception if all fallbacks fail
        raise
