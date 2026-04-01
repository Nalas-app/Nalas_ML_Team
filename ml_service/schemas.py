"""
Pydantic Schemas for ML Costing API
====================================
Matches the API contract defined in Phase 1 (1.4_api_contract_specification.md)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from datetime import date
from uuid import UUID


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class CostPredictionRequest(BaseModel):
    """Request schema for POST /ml/predict-cost"""
    menuItemId: str = Field(..., description="Menu item identifier (item_id from menu_items)")
    quantity: int = Field(..., gt=0, description="Number of servings (must be > 0)")
    eventDate: Optional[str] = Field(None, description="Event date in YYYY-MM-DD format")
    guestCount: Optional[int] = Field(None, gt=0, description="Expected guest count")

    @field_validator('eventDate')
    @classmethod
    def validate_event_date(cls, v):
        if v is not None:
            try:
                parsed = date.fromisoformat(v)
            except ValueError:
                raise ValueError('eventDate must be in YYYY-MM-DD format')
        return v


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class CostPredictionResponse(BaseModel):
    """Response schema for successful prediction"""
    ingredientCost: float = Field(..., ge=0, description="Total ingredient cost")
    laborCost: float = Field(..., ge=0, description="Labor cost (15% of ingredient)")
    overheadCost: float = Field(..., ge=0, description="Overhead cost (10% of ingredient)")
    totalCost: float = Field(..., gt=0, description="Final cost including profit margin")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    modelVersion: str = Field(..., description="Model version used")
    method: str = Field(..., description="Prediction method: rule_based or ml_model")


class ErrorDetail(BaseModel):
    """Error detail object"""
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    modelLoaded: bool
    activeModelVersion: str
    databaseConnected: bool
    menuItemsLoaded: int
    ingredientsLoaded: int


class ModelInfoResponse(BaseModel):
    """Model info response"""
    activeVersion: str
    method: str
    algorithm: str
    deployedAt: str
    menuItemsCovered: int
    ingredientsInMaster: int


class PredictionMetrics(BaseModel):
    """Prediction metrics"""
    total_predictions: int
    predictions_today: int
    avg_latency_ms: float
    error_count: int
    fallback_count: int
