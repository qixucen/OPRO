# opro/config.py
from pydantic import BaseModel, Field
import aiohttp
from typing import Optional
import backoff  # 用于重试机制

class APIError(Exception):
    """Base exception for API errors"""
    pass

class OPROConfig(BaseModel):
    """Configuration class for OPRO."""
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the API endpoint"
    )
    api_key: str = Field(
        description="API key for authentication"
    )
    model: str = Field(
        default="gpt-3.5-turbo",
        description="Model to use for optimization"
    )
    max_tokens: int = Field(
        default=150,
        description="Maximum number of tokens in completion"
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature"
    )
    timeout: int = Field(
        default=30,
        description="API request timeout in seconds"
    )
    
    class Config:
        arbitrary_types_allowed = True