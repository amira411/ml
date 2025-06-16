#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import uvicorn
from fastapi import APIRouter


class_names = ['fractured', 'not fractured']


def preprocess_image(image, img_height=224, img_width=224):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((img_width, img_height))
    img_array = np.array(image)

    if len(img_array.shape) == 2:  # If grayscale (2D array)
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 1:  # If single channel image
        img_array = np.concatenate([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:  # If RGBA (with alpha channel)
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def is_possible_xray(image: Image.Image):
    try:
        img_array = np.array(image)
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
            if np.allclose(r, g, atol=15) and np.allclose(g, b, atol=15):
                h, w = img_array.shape[:2]
                aspect_ratio = w / h
                if 0.6 < aspect_ratio < 1.6:
                    return True
        return False
    except:
        return False


app = FastAPI()
model = tf.keras.models.load_model("fractured_bones_detection/fracture_classification_model.h5")
fracture_router = APIRouter()


@fracture_router.post("/detect_fracture", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))

        
        if not is_possible_xray(image):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "اThe image is not suitable. Please make sure it is an X-ray image."
                }
            )

        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

        probability = float(predictions[0][0])
        predicted_class = int(probability > 0.5)
        probability_percent = round(probability * 100, 2)

        # تفسير النتيجة
        probability = 1 - probability
        if probability >= 0.85:
            comment = "High chance of fracture. Please consult a doctor immediately."
        elif probability >= 0.5:
            comment = "Possible fracture detected. It's best to get medical advice."
        elif probability >= 0.15:
            comment = "Low chance of fracture. But a check-up is still a good idea."
        else:
            comment = "Very low chance of fracture. No immediate concern detected."

        result = {
            "prediction": class_names[predicted_class],
            "fracture_probability_percent": 100 - probability_percent,
            "Analysis": comment
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


