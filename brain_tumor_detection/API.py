#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import cv2
import json
import io
from PIL import Image
import uvicorn
from fastapi import APIRouter


# In[2]:


def preprocess_image(image_bytes, target_size=(224, 224)):
  
    image = Image.open(io.BytesIO(image_bytes))
     
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image)
   
    resized_image = cv2.resize(image, target_size)

    
    rescaled_image = resized_image.astype('float32') / 255.0  
    rescaled_image = np.expand_dims(rescaled_image, axis=0)  

    return rescaled_image


# In[3]:


app = FastAPI()

brain_tumor_router = APIRouter()
model = tf.keras.models.load_model("brain_tumor_detection/brain_tumor_model.h5")


CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

@brain_tumor_router.post("/brain_tumor_classifier", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction for the uploaded MRI image.
    """
    try:
       
        contents = await file.read()

        
        image = preprocess_image(contents)

        
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)  
        
        
        prediction_class = CLASS_NAMES[predicted_class_index]

        return {
    
            "prediction": prediction_class
    

        }
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

