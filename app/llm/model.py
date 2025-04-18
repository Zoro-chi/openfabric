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

        # Authenticate with HuggingFace if token is provided and the model is not local
        if not os.path.isdir(model_path) and token:
            logger.info("Authenticating with HuggingFace using provided token")
            login(token=token, write_permission=False)
        elif not os.path.isdir(model_path) and os.environ.get("HF_TOKEN"):
            logger.info(
                "Authenticating with HuggingFace using HF_TOKEN environment variable"
            )
            login(token=os.environ.get("HF_TOKEN"), write_permission=False)

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
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

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

            # Load the tokenizer with error handling
            try:
                logger.info(f"Loading tokenizer from {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.info("Set pad_token to eos_token")
            except Exception as tokenizer_error:
                logger.error(f"Failed to load tokenizer: {str(tokenizer_error)}")

                # If there's a tokenizer_config.json file in the model directory but no tokenizer files,
                # try loading from a related model
                if os.path.isdir(model_path):
                    tokenizer_config_path = Path(model_path) / "tokenizer_config.json"
                    if tokenizer_config_path.exists():
                        fallback_tokenizer_model = "meta-llama/Llama-2-7b-chat-hf"
                        logger.info(
                            f"Attempting to load tokenizer from fallback model: {fallback_tokenizer_model}"
                        )
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(
                                fallback_tokenizer_model, trust_remote_code=True
                            )
                            logger.info(f"Successfully loaded fallback tokenizer")
                        except Exception as fallback_error:
                            logger.error(
                                f"Failed to load fallback tokenizer: {str(fallback_error)}"
                            )
                            raise
                    else:
                        raise
                else:
                    # Check if this is an authentication issue
                    if "401 Client Error" in str(
                        tokenizer_error
                    ) or "403 Client Error" in str(tokenizer_error):
                        raise ValueError(
                            f"Authentication error: You need a valid HuggingFace token to access {model_path}. "
                            f"Set the HF_TOKEN environment variable."
                        )
                    raise

            # Load the model with our fixed config
            logger.info(f"Loading model with device_map={device_map}")
            if device_map == "mps":
                # For Apple Silicon, load to device directly
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=self.torch_dtype,
                    device_map={"": "mps"},  # Map all modules to MPS device
                    trust_remote_code=True,
                )
            else:
                # For other devices, use the device_map parameter
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=self.torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
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

    # Check if the path points to a local directory
    is_local_model = os.path.isdir(model_path)

    if is_local_model:
        logger.info(f"Using local model directory: {model_path}")
    else:
        logger.info(
            f"Model not found locally. Using model ID from Hugging Face: {model_path}"
        )

        # If it's a HuggingFace model ID, we may need to check for token
        if "/" in model_path and not os.environ.get("HF_TOKEN"):
            # Check if it could be a gated model (meta-llama models are gated)
            if "meta-llama" in model_path:
                logger.warning(
                    f"Using potentially gated model '{model_path}' without HF_TOKEN. "
                    "This may fail if the model requires authentication."
                )

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

    # Get HuggingFace token from environment if available
    hf_token = os.environ.get("HF_TOKEN")

    try:
        return LocalLLM(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            token=hf_token,
        )
    except Exception as e:
        # If failure is tokenizer related and it's a local model, try direct loading workarounds
        if "tokenizer" in str(e).lower() and is_local_model:
            logger.warning(f"Failed to load model with error: {str(e)}")
            logger.info("Trying workaround for local model with tokenizer issues...")

            # Try a simpler local model loading approach
            try:
                from transformers import LlamaTokenizer, LlamaForCausalLM

                # Try to find parent directories that might contain tokenizer files
                model_dir = Path(model_path)

                # Try to load with LlamaTokenizer directly
                try:
                    logger.info(
                        f"Attempting to load tokenizer with LlamaTokenizer directly..."
                    )
                    tokenizer = LlamaTokenizer.from_pretrained(
                        model_dir, trust_remote_code=True
                    )
                    model = LlamaForCausalLM.from_pretrained(
                        model_dir,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                    )

                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        framework="pt",
                    )
                    llm = LocalLLM(model_path=model_path, device_map=device_map)
                    llm.pipe = pipe
                    logger.info(
                        "Successfully loaded model with LlamaTokenizer/LlamaForCausalLM directly"
                    )
                    return llm

                except Exception as direct_error:
                    logger.error(f"Direct loading failed: {str(direct_error)}")

                    # Try loading tokenizer from HuggingFace even for local model
                    try:
                        logger.info("Attempting to load tokenizer from HuggingFace...")
                        base_model_id = "meta-llama/Llama-2-7b-chat-hf"  # Common base model that might work
                        tokenizer = AutoTokenizer.from_pretrained(
                            base_model_id, trust_remote_code=True
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch_dtype,
                            device_map=device_map,
                            trust_remote_code=True,
                        )

                        pipe = pipeline(
                            "text-generation",
                            model=model,
                            tokenizer=tokenizer,
                            framework="pt",
                        )
                        llm = LocalLLM(model_path=model_path, device_map=device_map)
                        llm.pipe = pipe
                        logger.info(
                            "Successfully loaded model with HF tokenizer and local model"
                        )
                        return llm
                    except Exception as hf_tokenizer_error:
                        logger.error(
                            f"HF tokenizer loading failed: {str(hf_tokenizer_error)}"
                        )
                        raise e  # Re-raise original error if all workarounds fail
            except Exception as workaround_error:
                logger.error(f"Workaround failed: {str(workaround_error)}")
                raise e  # Re-raise original error
        elif "tokenizer" in str(e).lower() and "meta-llama" in model_path:
            # This could be an authentication issue for remote meta-llama models
            if "401 Client Error" in str(e) or "403 Client Error" in str(e):
                logger.warning(
                    "Authentication error loading meta-llama model. Trying fallback open models."
                )

                # Try Mistral-7B-Instruct-v0.2 as a high-quality open source fallback first
                try:
                    fallback_model = "mistralai/Mistral-7B-Instruct-v0.2"
                    logger.info(f"Attempting to load fallback model: {fallback_model}")
                    return get_llm_instance(fallback_model)
                except Exception as mistral_error:
                    logger.error(f"Mistral fallback model failed: {str(mistral_error)}")

                    # Try Phi-3-mini as secondary fallback
                    try:
                        phi3_model = "microsoft/Phi-3-mini-4k-instruct"
                        logger.info(
                            f"Attempting to load Phi-3 fallback model: {phi3_model}"
                        )
                        return get_llm_instance(phi3_model)
                    except Exception as phi3_error:
                        logger.error(f"Phi-3 fallback model failed: {str(phi3_error)}")

                        # Try Gemma as final fallback
                        try:
                            gemma_model = "google/gemma-2b-it"
                            logger.info(
                                f"Attempting to load Gemma fallback model: {gemma_model}"
                            )
                            return get_llm_instance(gemma_model)
                        except Exception as gemma_error:
                            logger.error(f"All fallback models failed")
                            raise e  # Re-raise original error if all fallbacks fail
            else:
                raise
        else:
            # For other errors, just raise the original exception
            raise
