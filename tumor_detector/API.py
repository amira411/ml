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


app = FastAPI()
brain_tumor_detector_router = APIRouter()
model = tf.keras.models.load_model("tumor_detector/brain_tumor_model.h5")


# In[3]:


def preprocess_image(image_bytes):
    # Read image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB
    image = image.convert('RGB')
    
    # Convert to numpy array
    image = np.array(image)
    
    # Resize
    image = cv2.resize(image, (128,128))
    
    # Rescale
    image = image * (1./255)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image



# In[4]:


@brain_tumor_detector_router.post("/detect_tumor", response_class=JSONResponse)

async def predict(file: UploadFile = File(...)):
    """
    Make a prediction for brain tumor presence in the uploaded MRI image.
    """
    try:
        # Read the file
        contents = await file.read()
        
        # Preprocess the image 
        image = preprocess_image(contents)
        
        # Make prediction
        prediction = model.predict(image)
        probability = float(prediction[0][0])
        
        prediction_class = "TUMOR" if np.argmax([1 - probability, probability]) == 1 else "NO TUMOR DETECTED"
        
        return prediction_class
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# In[ ]:




