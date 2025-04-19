from datasets import load_dataset, Dataset
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer
import torch
import torch.nn as nn
from torchvision.transforms import Compose, RandomResizedCrop, Normalize
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
import numpy as np
from huggingface_hub import login
import logging
import sys
import traceback

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

try:
    # Login to Hugging Face
    logger.info("Attempting to login to Hugging Face...")
    login('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')
    logger.info("Successfully logged in to Hugging Face")
except Exception as e:
    logger.error(f"Failed to login to Hugging Face: {str(e)}")
    logger.error(traceback.format_exc())
    raise

try:
    logger.info("Loading dataset...")
    dataset = load_dataset("JeevakrishnaVetrivel/Foods_data")
    logger.info(f"Dataset loaded successfully! Found {len(dataset['train'])} training examples")
except Exception as e:
    logger.error(f"Failed to load dataset: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {str(e)}")
        return None

try:
    logger.info("Initializing feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        use_fast=True
    )
    logger.info("Feature extractor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize feature extractor: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Create global label mappings
all_names = list(set(dataset["train"]["name"]))
label2id = {label: i for i, label in enumerate(all_names)}
id2label = {i: label for label, i in label2id.items()}
logger.info(f"Created label mappings for {len(label2id)} classes")

def preprocess_images(examples):
    logger.info(f"Processing batch of {len(examples['imgurl'])} examples...")
    
    # Download images from URLs
    images = []
    valid_indices = []
    labels = []
    nutritional_values = []
    
    for idx, (url, name, nutrition_str) in enumerate(zip(examples["imgurl"], examples["name"], examples["nutritions"])):
        img = download_image(url)
        if img is not None:
            images.append(img)
            labels.append(label2id[name])
            
            # Extract nutritional values
            try:
                nutrition_dict = eval(nutrition_str)
                nutritional_values.append([
                    float(nutrition_dict.get('Calories', '0 kcal').split()[0]),
                    float(nutrition_dict.get('Protein', '0 g').split()[0]),
                    float(nutrition_dict.get('Fat', '0 g').split()[0]),
                    float(nutrition_dict.get('Carbohydrates', '0 g').split()[0]),
                    float(nutrition_dict.get('Fiber', '0 g').split()[0])
                ])
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Failed to process nutrition data for {name}: {str(e)}")
                continue
    
    if not images:
        logger.warning("No valid images found in batch")
        return {}
    
    try:
        # Process images using feature extractor
        inputs = feature_extractor(images=images, return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values,
            "labels": labels,
            "nutritional_values": nutritional_values
        }
    except Exception as e:
        logger.error(f"Failed to process images: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

try:
    logger.info("Preprocessing dataset...")
    # Split dataset into train and validation
    train_dataset = dataset["train"]
    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)
    logger.info(f"Split dataset into {len(train_val_split['train'])} training and {len(train_val_split['test'])} validation examples")

    # Process train and validation sets
    processed_train = train_val_split["train"].map(
        preprocess_images,
        batched=True,
        batch_size=16,
        remove_columns=train_dataset.column_names
    )
    logger.info(f"Processed {len(processed_train)} training examples")

    processed_val = train_val_split["test"].map(
        preprocess_images,
        batched=True,
        batch_size=16,
        remove_columns=train_dataset.column_names
    )
    logger.info(f"Processed {len(processed_val)} validation examples")

    logger.info(f"Number of unique food classes: {len(label2id)}")
except Exception as e:
    logger.error(f"Failed to preprocess dataset: {str(e)}")
    logger.error(traceback.format_exc())
    raise

class FoodClassificationModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        logger.info(f"Initializing model with {num_labels} classes")
        self.vit = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        # Additional layer for nutritional value prediction
        self.nutrition_head = nn.Sequential(
            nn.Linear(num_labels, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 nutritional values: calories, protein, fat, carbs, fiber
        )
        logger.info("Model initialized successfully")
        
    def forward(self, pixel_values, labels=None, nutritional_values=None):
        try:
            outputs = self.vit(pixel_values, labels=labels)
            logits = outputs.logits
            
            # Predict nutritional values from logits
            nutrition_pred = self.nutrition_head(logits)
            
            if labels is not None and nutritional_values is not None:
                # Classification loss
                loss_cls = outputs.loss
                
                # Nutritional value prediction loss (MSE)
                loss_nutrition = nn.MSELoss()(nutrition_pred, torch.tensor(nutritional_values, dtype=torch.float32).to(pixel_values.device))
                
                # Combined loss
                total_loss = loss_cls + 0.5 * loss_nutrition
                
                return {"loss": total_loss, "logits": logits, "nutrition_pred": nutrition_pred}
            
            return {"logits": logits, "nutrition_pred": nutrition_pred}
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise

try:
    logger.info("Initializing model...")
    model = FoodClassificationModel(num_labels=len(label2id))

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        push_to_hub=True,
        learning_rate=2e-5,
        hub_model_id="food-focus-ai",
        hub_strategy="end",
        remove_unused_columns=False
    )

    class NutritionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            try:
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    labels=inputs["labels"],
                    nutritional_values=inputs["nutritional_values"]
                )
                
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                return (loss, outputs) if return_outputs else loss
            except Exception as e:
                logger.error(f"Error in compute_loss: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    logger.info("Initializing trainer...")
    trainer = NutritionTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_val,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Pushing model to hub...")
    trainer.push_to_hub("food-focus-ai")

    logger.info("Training completed successfully!")
except Exception as e:
    logger.error(f"Training failed: {str(e)}")
    logger.error(traceback.format_exc())
    raise
