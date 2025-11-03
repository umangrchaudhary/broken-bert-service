"""
Pydantic schemas for the sentiment analysis API.

This module defines the request and response models used by the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Request model for sentiment prediction.
    
    Attributes:
        text: The review text to classify
    """
    text: str = Field(
        ..., 
        min_length=1,
        max_length=5000,
        description="The review text to classify for sentiment",
        example="This movie was absolutely fantastic! Great acting and wonderful storyline."
    )


class PredictionResponse(BaseModel):
    """
    Response model for sentiment prediction.
    
    Attributes:
        label: The predicted sentiment label ("positive" or "negative")
        confidence: The confidence score for the prediction (0.0 to 1.0)
    """
    label: str = Field(
        ...,
        description="The predicted sentiment label",
        example="positive"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction",
        example=0.93
    )


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch sentiment prediction.
    
    Attributes:
        texts: List of review texts to classify
    """
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of review texts to classify",
        example=[
            "This movie was fantastic!",
            "Terrible film, waste of time."
        ]
    )


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch sentiment prediction.
    
    Attributes:
        predictions: List of prediction results
    """
    predictions: list[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    
    Attributes:
        status: Health status
        model_ready: Whether the model is loaded
        message: Additional information
    """
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(
        ...,
        description="Health status",
        example="healthy"
    )
    model_ready: bool = Field(
        ...,
        description="Whether the model is loaded",
        example=True
    )
    message: str = Field(
        ...,
        description="Additional information",
        example="Sentiment analysis API is running"
    )


class RecommendationRequest(BaseModel):
    """
    Request model for product recommendations.
    
    Attributes:
        text: The search query text for finding similar products
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query text for product recommendations",
        example="Looking for a fast laptop"
    )


class RecommendationResponse(BaseModel):
    """
    Response model for product recommendations.
    
    Attributes:
        recommended_products: List of recommended product names
    """
    recommended_products: list[str] = Field(
        ...,
        description="List of recommended product names",
        example=["MacBook Air M2", "Dell XPS 13", "Lenovo Yoga Slim"]
    )


class DetailedRecommendationResponse(BaseModel):
    """
    Detailed response model for product recommendations with scores.
    
    Attributes:
        recommendations: List of detailed recommendation objects
    """
    
    class RecommendationItem(BaseModel):
        """Individual recommendation item."""
        product_id: int = Field(..., description="Product ID")
        product_title: str = Field(..., description="Product title")
        product_description: str = Field(..., description="Product description")
        similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    
    recommendations: list[RecommendationItem] = Field(
        ...,
        description="List of detailed recommendations with scores"
    )


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    
    Attributes:
        error: Error type
        message: Error message
        details: Optional additional details
    """
    error: str = Field(
        ...,
        description="Error type",
        example="ValidationError"
    )
    message: str = Field(
        ...,
        description="Error message",
        example="Input text cannot be empty"
    )
    details: Optional[str] = Field(
        None,
        description="Additional error details"
    )