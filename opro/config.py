# opro/config.py
from pydantic import BaseModel, Field

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
    optimization_model: str = Field(
        default="gpt-3.5-turbo",
        description="Model to use for optimization"
    )
    execution_model: str = Field(
        default="gpt-3.5-turbo",
        description="Model to use for execution"
    )
    max_tokens: int = Field(
        default=None,
        description="Maximum number of tokens in completion"
    )
    temperature: float = Field(
        default=1,
        description="Sampling temperature"
    )
    timeout: int = Field(
        default=5,
        description="API request timeout in seconds"
    )
    
    class Config:
        arbitrary_types_allowed = True