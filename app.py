from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from food_recognition import FoodRecognitionService
from model import FoodResponse
import uvicorn
import logging
import os
from typing import Optional
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Food Focus AI",
    description="API for food recognition and calorie estimation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the food recognition service
try:
    logger.info("Initializing FoodRecognitionService...")
    food_service = FoodRecognitionService()
    logger.info("FoodRecognitionService initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize FoodRecognitionService: {str(e)}")
    raise

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "Food Focus AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "recognize": "/api/recognize",
            "health": "/api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": food_service is not None
    }

@app.post("/api/recognize", response_model=FoodResponse)
async def recognize_food(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...)
):
    """
    Endpoint to recognize food from uploaded image and return nutritional information
    """
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.content_type}. Only image files are allowed."
            )
        
        # Validate file size (max 10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await image.read(chunk_size):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
                )
        
        # Reset file pointer
        await image.seek(0)
        
        # Read image file
        image_bytes = await image.read()
        
        # Process image and get predictions
        result = food_service.process_image(image_bytes)
        
        if result.error:
            logger.error(f"Error processing image: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        
        # Log successful prediction
        logger.info(f"Successfully processed image. Predicted food: {result.food_name}")
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "food_name": "Unknown",
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "match_confidence": 0
            }
        )

if __name__ == '__main__':
    logger.info("Starting Food Focus AI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Enable auto-reload for development
    )