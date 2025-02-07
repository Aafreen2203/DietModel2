from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import logging
import os
import re  # Import re for regex operations

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NutriAI Recipe API")

# Load the dataset
try:
    # Check if the file exists in the current directory
    if not os.path.exists('NutriAI1.csv'):
        raise FileNotFoundError("NutriAI1.csv not found in the current directory.")
    
    df = pd.read_csv('NutriAI1.csv', encoding='latin-1')
    df['processed_recipes'] = df['Recipes'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    logger.info("Dataset loaded successfully")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise HTTPException(status_code=500, detail="Error loading dataset. Please ensure the dataset file is in the correct directory.")

# Load models and vectorizers
try:
    cal_model = joblib.load('calories_model (1).joblib')
    carb_model = joblib.load('carbs_model (1).joblib')
    recipe_vectorizer = joblib.load('recipe_vectorizer.joblib')
    ingredients_nn = joblib.load('ingredients_nn.joblib')
    steps_nn = joblib.load('steps_nn.joblib')
    logger.info("Models and vectorizers loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise HTTPException(status_code=500, detail="Error loading models. Please ensure all model files are in the correct directory.")

class RecipeInput(BaseModel):
    recipe_name: str

class RecipeResponse(BaseModel):
    recipe_name: str
    predicted_calories: float
    predicted_carbs: float
    suggested_ingredients: List[str]
    suggested_steps: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to NutriAI Recipe API"}

@app.post("/predict_recipe", response_model=RecipeResponse)
async def predict_recipe(recipe: RecipeInput):
    try:
        # Ensure the DataFrame is loaded
        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="Dataset not loaded. Please ensure the recipe dataset is available."
            )
            
        # Preprocess and transform recipe name
        processed_recipe_name = re.sub(r'[^\w\s]', '', recipe.recipe_name.lower())
        recipe_vector = recipe_vectorizer.transform([processed_recipe_name])
        
        # Predict calories and carbs
        predicted_calories = cal_model.predict(recipe_vector)[0]
        predicted_carbs = carb_model.predict(recipe_vector)[0]
        
        # Find similar recipes
        ingredients_distances, ingredients_indices = ingredients_nn.kneighbors(recipe_vector)
        steps_distances, steps_indices = steps_nn.kneighbors(recipe_vector)
        
        # Get ingredients and steps from similar recipes
        similar_ingredients = df['Ingredients'].iloc[ingredients_indices[0]].tolist()
        similar_steps = df['Steps'].iloc[steps_indices[0]].tolist()
        
        return RecipeResponse(
            recipe_name=recipe.recipe_name,
            predicted_calories=round(float(predicted_calories), 2),
            predicted_carbs=round(float(predicted_carbs), 2),
            suggested_ingredients=similar_ingredients,
            suggested_steps=similar_steps
        )
    except Exception as e:
        logger.error(f"Error in /predict_recipe endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
