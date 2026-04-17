"""
Nalas ML Costing Engine — FastAPI Application
==============================================
Phase 6: Production Preparation
Hybrid ML + Rule-Based prediction API.

Endpoints:
    POST /ml/predict-cost   — Cost prediction
    GET  /ml/health          — Health check
    GET  /ml/model-info      — Model information
    GET  /ml/metrics         — Prediction metrics
    GET  /ml/menu-items      — List available menu items
    GET  /                   — Root greeting

Run: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import sys
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

# Local imports
from ml_service.schemas import (
    CostPredictionRequest,
    CostPredictionResponse,
    ErrorResponse,
    ErrorDetail,
    HealthResponse,
    ModelInfoResponse,
)
from ml_service.cost_engine import CostingError
from ml_service.production_predictor import ProductionPredictor
from ml_service.prediction_logger import PredictionLogger

# ============================================================
# LOGGING SETUP
# ============================================================
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "app.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("ml_costing.api")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "structured")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "business_rules.json")

# Ensure directories exist
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# GLOBAL OBJECTS (initialized at startup)
# ============================================================
predictor = None
pred_logger = PredictionLogger()
startup_time = None


# ============================================================
# LIFESPAN — Initialize on startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and model on startup, cleanup on shutdown."""
    global predictor, startup_time

    logger.info("=" * 50)
    logger.info("NALAS ML COSTING ENGINE — Starting up")
    logger.info("=" * 50)

    try:
        predictor = ProductionPredictor(
            model_dir=MODEL_DIR,
            data_dir=DATA_DIR,
            config_path=CONFIG_PATH
        )
        predictor.initialize()
        logger.info(f"Predictor initialized (ML active: {predictor.is_ml_available})")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}", exc_info=True)

    startup_time = datetime.now()
    logger.info(f"Server ready at {startup_time.isoformat()}")

    yield

    # Cleanup
    logger.info("Shutting down...")


# ============================================================
# APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="Nalas ML Costing Engine",
    description="Hybrid ML + Rule-based cost prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.exception_handler(CostingError)
async def costing_error_handler(request: Request, exc: CostingError):
    """Handle CostingError with proper error codes."""
    status_map = {
        "ML_001": 400,  # Invalid input
        "ML_002": 400,  # Item not found
        "ML_003": 404,  # Recipe not found
        "ML_004": 404,  # Cannot calculate (missing data)
        "ML_005": 500,  # ML model error
        "ML_500": 500,  # Internal error
    }
    status_code = status_map.get(exc.code, 500)

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


# ============================================================
# SECURITY
# ============================================================
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API keys from headers."""
    expected_key = os.getenv("ML_API_KEY", "nalas-ml-secret-2026")
    if api_key == expected_key:
        return api_key
    
    logger.warning(f"Unauthorized access attempt with API key: {api_key}")
    raise HTTPException(
        status_code=403,
        detail="Could not validate API Key"
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/ml/diagnose")
def diagnose():
    """Diagnostic endpoint to debug environment issues."""
    try:
        model_exists = os.path.exists(MODEL_DIR)
        model_files = os.listdir(MODEL_DIR) if model_exists else []
        
        v101_dir = os.path.join(MODEL_DIR, "v1.0.1")
        v101_exists = os.path.exists(v101_dir)
        v101_files = os.listdir(v101_dir) if v101_exists else []

        return {
            "status": "diagnostic_mode",
            "paths": {
                "base_dir": BASE_DIR,
                "model_dir": MODEL_DIR,
                "data_dir": DATA_DIR
            },
            "filesystem": {
                "model_dir_exists": model_exists,
                "root_model_files": model_files,
                "v101_exists": v101_exists,
                "v101_files": v101_files
            },
            "predictor_state": {
                "initialized": predictor._initialized if predictor else False,
                "ml_available": predictor.is_ml_available if predictor else False,
                "init_error": predictor._init_error if predictor else "Predictor not created"
            }
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/")
def root():
    """Root endpoint."""
    is_up = predictor is not None and predictor._initialized
    return {
        "message": "Nalas ML Costing Engine is running",
        "version": "1.0.0",
        "ml_available": predictor.is_ml_available if is_up else False,
        "docs": "/docs",
        "diagnostic": "/ml/diagnose"
    }


@app.post("/ml/predict-cost", response_model=CostPredictionResponse, dependencies=[Depends(get_api_key)])
async def predict_cost(request: CostPredictionRequest):
    """
    Predict cost breakdown for a menu item.

    Uses ML model if available and confident, otherwise falls back
    to deterministic rule-based calculation. Returns full cost breakdown.
    """
    start_time = time.time()

    if predictor is None or not predictor._initialized:
        error = CostingError("ML_500", "Predictor not initialized — startup failed")
        pred_logger.log_prediction(
            request=request.model_dump(),
            response={},
            latency_ms=0,
            error={"code": error.code, "message": error.message},
        )
        raise error

    try:
        result = predictor.predict(
            menu_item_id=request.menuItemId,
            quantity=request.quantity,
            event_date=request.eventDate,
            guest_count=request.guestCount,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Log the prediction
        pred_logger.log_prediction(
            request=request.model_dump(),
            response=result,
            latency_ms=latency_ms,
        )

        # Return only the API contract fields (not _meta)
        return CostPredictionResponse(
            ingredientCost=result["ingredientCost"],
            laborCost=result["laborCost"],
            overheadCost=result["overheadCost"],
            totalCost=result["totalCost"],
            confidence=result["confidence"],
            modelVersion=result["modelVersion"],
            method=result["method"],
        )

    except CostingError as e:
        latency_ms = (time.time() - start_time) * 1000
        pred_logger.log_prediction(
            request=request.model_dump(),
            response={},
            latency_ms=latency_ms,
            error={"code": e.code, "message": e.message},
        )
        raise

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"Unexpected error: {e}", exc_info=True)
        pred_logger.log_prediction(
            request=request.model_dump(),
            response={},
            latency_ms=latency_ms,
            error={"code": "ML_500", "message": str(e)},
        )
        raise CostingError("ML_500", f"Internal error: {str(e)}")


@app.get("/ml/health", response_model=HealthResponse, dependencies=[Depends(get_api_key)])
async def health_check():
    """Service health check."""
    is_healthy = predictor is not None and predictor._initialized
    ds = predictor._data_store if is_healthy else None

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        modelLoaded=is_healthy and predictor.is_ml_available,
        activeModelVersion=predictor.active_model_version if is_healthy and predictor.is_ml_available else "N/A",
        databaseConnected=is_healthy and ds and ds.is_loaded,
        menuItemsLoaded=len(ds.menu_items) if is_healthy and ds else 0,
        ingredientsLoaded=len(ds.ingredients) if is_healthy and ds else 0,
    )


@app.get("/ml/model-info", response_model=ModelInfoResponse, dependencies=[Depends(get_api_key)])
async def model_info():
    """Get information about the active model."""
    is_up = predictor is not None and predictor._initialized
    ml_up = is_up and predictor.is_ml_available
    ds = predictor._data_store if is_up else None

    return ModelInfoResponse(
        activeVersion=predictor.active_model_version if ml_up else "N/A",
        method="ml_model + rule_based_fallback" if ml_up else "rule_based",
        algorithm="GradientBoosting" if ml_up else "Deterministic rules",
        deployedAt=startup_time.isoformat() if startup_time else "N/A",
        menuItemsCovered=len(ds.menu_items) if is_up and ds else 0,
        ingredientsInMaster=len(ds.ingredients) if is_up and ds else 0,
    )


@app.get("/ml/metrics", dependencies=[Depends(get_api_key)])
async def get_metrics():
    """Get prediction metrics and fallback stats."""
    metrics = pred_logger.get_metrics()
    predictor_stats = predictor.get_stats() if predictor and predictor._initialized else {}

    return {
        "predictions": {
            "total": metrics["total_predictions"],
            "today": metrics["predictions_today"],
        },
        "latency": {
            "avg_ms": metrics["avg_latency_ms"],
            "p95_ms": metrics["p95_latency_ms"],
            "p99_ms": metrics["p99_latency_ms"],
        },
        "errors": {
            "total": metrics["error_count"],
            "rate_pct": metrics["error_rate_pct"],
        },
        "model_performance": predictor_stats,
    }


@app.get("/ml/menu-items", dependencies=[Depends(get_api_key)])
async def list_menu_items():
    """List all available menu items and their prediction readiness."""
    if predictor is None or not predictor._initialized or predictor._data_store is None:
        raise HTTPException(500, "Data not loaded")

    ds = predictor._data_store
    items = []
    for _, row in ds.menu_items.iterrows():
        item_id = row['item_id']
        recipe = ds.get_recipe(item_id)
        has_qty = recipe['quantity_per_base_unit'].notna().sum() if len(recipe) > 0 else 0
        total_ing = len(recipe)

        items.append({
            "itemId": item_id,
            "name": str(row['item_name']),
            "category": str(row['category']),
            "source": str(row.get('source', 'Unknown')),
            "totalIngredients": int(total_ing),
            "ingredientsWithQuantity": int(has_qty),
            "predictionReady": bool(has_qty > 0),
        })

    return {
        "total": len(items),
        "predictionReady": sum(1 for i in items if i["predictionReady"]),
        "items": items,
    }