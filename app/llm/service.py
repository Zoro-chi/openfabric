import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.llm.model import LocalLLM, get_llm_instance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Local LLM Service", description="API for local LLM interaction")


# Model request and response classes
class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ExpandRequest(BaseModel):
    prompt: str


class LLMResponse(BaseModel):
    text: str


# Global LLM instance
llm = None


@app.on_event("startup")
async def startup_event():
    """Initialize the LLM on startup"""
    global llm
    model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
    try:
        llm = get_llm_instance(model_id)
        logger.info(f"LLM initialized successfully with model: {model_id}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise


@app.post("/generate", response_model=LLMResponse)
async def generate_text(request: PromptRequest):
    """Generate text based on a prompt"""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    try:
        response = llm.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        return LLMResponse(text=response)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/expand", response_model=LLMResponse)
async def expand_prompt(request: ExpandRequest):
    """Expand a creative prompt with rich details"""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    try:
        expanded = llm.expand_creative_prompt(request.prompt)
        return LLMResponse(text=expanded)
    except Exception as e:
        logger.error(f"Error expanding prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if llm:
        return {"status": "healthy", "model": llm.model_path}
    return {"status": "initializing"}


# Start the service if run directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
