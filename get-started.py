#!/usr/bin/env python
# coding: utf-8

# ## MLFlow Tracking Server

# Used to track each and every experiments - will also be able to compare in form of visualizations
# in terminal : 'mlflow ui' then press enter -> this will start the mlflow tracking uri

# In[1]:


import mlflow


# In[2]:


mlflow.set_tracking_uri("http://127.0.0.1:5000") ##i.e. here u are setting that mlflow will track everything in this specific url.
## (this url is our localhost url) , port number is '5000'.
## If my remote repository is somewhere else can i give that particular ulr ?? -> Yes u can.
## Whatever experiments are happening in this notebook mlflow will make sure that those experiments get tracked in that particular url.
## On this url my mlflow tracking server is running.


# In[ ]:


mlflow.set_experiment("Check localhost connection") ##Experiment name is passed here (Note MLFlow must be started b4 doing this -> mlfow ui in terminal)

with mlflow.start_run():
    mlflow.log_metric("test",1)
    mlflow.log_metric("Krish",2)


# In[6]:


with mlflow.start_run():
    mlflow.log_metric("test1",1)
    mlflow.log_metric("Krish1",2)


# 'Check localhost connection' me ab 2 experiements aa jayenge (mlflow.start_run() executed 2 times) -> in 'Check localhost connection' we can 
# select the two experiements and press 'compare' to compare both the experiments
# Whichenver experiment performs well we can select that. 
# 
# MLruns folder created with all the details

# In[ ]:


with mlflow.start_run():
    mlflow.log_metric("test2",1)
    mlflow.log_metric("Krish2",2)  ## Logging metrics


# if u stop the terminal mlflow tracking will also stop
# if u delete the mlruns folder created then also information will be lost
# This tells that entire tracking is actually happening 
