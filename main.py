#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
import uvicorn





from fractured_bones_detection.API import fracture_router
from tumor_detector.API import brain_tumor_detector_router




# In[2]:


app=FastAPI()

# Add root endpoint
@app.get("/")
async def root():
    return {"Upload A Clear MRI OR X-RAY Image To Diagnose "}
app.include_router(brain_tumor_detector_router)

app.include_router(fracture_router)



# In[ ]:




