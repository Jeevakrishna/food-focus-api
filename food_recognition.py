from PIL import Image
import io
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
import pandas as pd
from model import FoodResponse
import logging
import os
from huggingface_hub import hf_hub_download

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodRecognitionService:
    def __init__(self):
        try:
            # Create data directory if it doesn't exist
            os.makedirs("Foods_data", exist_ok=True)
            
            # Download dataset if it doesn't exist
            dataset_path = "Foods_data/Foods_data.csv"
            if not os.path.exists(dataset_path):
                logger.info("Downloading dataset...")
                hf_hub_download(
                    repo_id="JeevakrishnaVetrivel/Foods_data",
                    filename="Foods_data.csv",
                    local_dir="Foods_data"
                )
            
            # Load the dataset
            logger.info("Loading food dataset...")
            self.dataset = pd.read_csv(dataset_path)
            
            # Initialize the model and feature extractor
            logger.info("Loading model and feature extractor...")
            model_name = "google/vit-base-patch16-224-in21k"  # Use base model for now
            self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=len(self.dataset),
                ignore_mismatched_sizes=True
            )
            
            # Move model to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}!")
            
        except Exception as e:
            logger.error(f"Failed to initialize FoodRecognitionService: {str(e)}")
            raise
        
    def process_image(self, image_bytes):
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Ensure image is in RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess image
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits.softmax(dim=-1)
                
            # Get the predicted class and confidence
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()
            
            # Get food name and nutrition info from dataset
            food_info = self.get_food_info(pred_class)
            
            return FoodResponse(
                food_name=food_info["name"],
                calories=food_info["calories"],
                protein=food_info["protein"],
                carbs=food_info["carbs"],
                fat=food_info["fat"],
                match_confidence=confidence * 100  # Convert to percentage
            )
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return FoodResponse(
                food_name="Unknown",
                calories=0,
                protein=0,
                carbs=0,
                fat=0,
                match_confidence=0,
                error=str(e)
            )
            
    def get_food_info(self, class_id):
        try:
            # Get the food information from the dataset
            food_row = self.dataset.iloc[class_id]
            
            # Extract nutrition information
            nutrition_dict = eval(food_row['nutritions'])
            
            # Parse nutrition values
            calories = float(nutrition_dict['Calories'].split()[0])
            protein = float(nutrition_dict.get('Protein', '0 g').split()[0])
            carbs = float(nutrition_dict.get('Carbohydrates', '0 g').split()[0])
            fat = float(nutrition_dict.get('Fat', '0 g').split()[0])
            
            return {
                "name": food_row['name'],
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fat": fat
            }
        except Exception as e:
            logger.error(f"Error getting food info: {str(e)}")
            raise