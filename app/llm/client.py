import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Client for interacting with the LLM service.
    Provides methods to generate text and expand creative prompts.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        """
        Initialize the LLM client.

        Args:
            base_url: Base URL of the LLM service
        """
        self.base_url = base_url
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text based on a prompt.

        Args:
            prompt: The user prompt to generate from
            system_prompt: Optional system prompt to guide the generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter

        Returns:
            The generated text

        Raises:
            Exception: If the request fails
        """
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if system_prompt:
            payload["system_prompt"] = system_prompt

        try:
            response = self.session.post(f"{self.base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()["text"]
        except requests.RequestException as e:
            logger.error(f"Failed to generate text: {str(e)}")
            raise Exception(f"LLM service error: {str(e)}")

    def expand_prompt(self, prompt: str) -> str:
        """
        Expand a creative prompt with rich details.

        Args:
            prompt: The user's original prompt

        Returns:
            An expanded, detailed creative prompt

        Raises:
            Exception: If the request fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/expand", json={"prompt": prompt}
            )
            response.raise_for_status()
            return response.json()["text"]
        except requests.RequestException as e:
            logger.error(f"Failed to expand prompt: {str(e)}")
            raise Exception(f"LLM service error: {str(e)}")

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the LLM service is healthy.

        Returns:
            Health status information

        Raises:
            Exception: If the health check fails
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Health check failed: {str(e)}")
            raise Exception(f"LLM service error: {str(e)}")
