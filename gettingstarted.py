#!/usr/bin/env python
# coding: utf-8

# Idea -> Take some dataset , train our model (during training we will track multiple information)
# (Info abt multiple parameters while training -> try visualize these things(parameters))

# In[2]:


import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature  ##Very important in inferencing step once we train our model , we also use this to make ensure that
## our schema is equivalent w.r.t. to both input and output , and also used to set schema.


# In[3]:


## set the tracking uri
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


# In[4]:


## load the dataset
X,y = datasets.load_iris(return_X_y=True)


# In[5]:


X


# In[6]:


y


# In[7]:


# split the data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20) ##20% data is test data

# Define the model hyperparameters
params = {"penalty":"l2","solver":"lbfgs","max_iter":1000,"multi_class":"auto","random_state":8888}

# Train the model
lr = LogisticRegression(**params) ## ** means keyword argument for params(as its key value pair)
lr.fit(X_train,y_train)


# In[8]:


X_test


# In[9]:


## Prediction on the test set
y_pred = lr.predict(X_test)
y_pred


# In[10]:


accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


# In[11]:


### MLFLOW tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000") ## Default port : 5000(u can keep any)
## MLFlow is also packaging the versions of the module u are using

## Experiment will be created aur uske baad jab jab mlflow.start_run() hoga uss experiment me instances create honge unique id se

## create a new MLFLOW experiment
mlflow.set_experiment("MLFLOW Quickstart")

## Start an MLFLOW run
with mlflow.start_run():
    ## log the hyperparameters
    mlflow.log_params(params)

    ## Log the accuracy metrics -> mlflow.log_metric() for only one metric we can use , mlflow.log_metrics() is for multiple metrics
    ## Note: Metrics should be in form of key,value pairs
    mlflow.log_metric("accuracy",accuracy)

    ## Set a tag we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info","Basic LR model for iris data")

    ## Infer the model signature
    ## The signature represents model input and output as data frames with named columns and data type specified as one of the types
    ## defined in mlflow.types.DataType. i.e. schema of the input and output is fixed over here and it represents that.
    signature = infer_signature(X_train,lr.predict(X_train)) ## X_train will be the input and prediction on X_train will be the output.

    ## log the model (we trained our model using sklearn hence we used mlflow.sklearn, if we use langchain then mlflow.langchain)
    model_info = mlflow.sklearn.log_model(
        sk_model = lr, ## lr is my model name
        artifact_path = "iris_model", ## path name(where model will be saved) , artifact_path is depreceated use -> name = "iris_model"
        signature = signature,
        input_example=X_train,
        registered_model_name = "tracking-quickstart"
    )


# In[12]:


## Creating model with other parameters
# Define the model hyperparameters
params = {"solver":"newton-cg","max_iter":1000,"multi_class":"auto","random_state":1000}

# Train the model
lr = LogisticRegression(**params) ## ** means keyword argument for params(as its key value pair)
lr.fit(X_train,y_train)


# In[13]:


## Prediction on the test set
y_pred = lr.predict(X_test)
y_pred


# In[14]:


accuracy = accuracy_score(y_test,y_pred)
print(accuracy)


# In[15]:


## Start an MLFLOW run
with mlflow.start_run():
    ## log the hyperparameters
    mlflow.log_params(params)

    ## Log the accuracy metrics -> mlflow.log_metric() for only one metric we can use , mlflow.log_metrics() is for multiple metrics
    ## Note: Metrics should be in form of key,value pairs
    mlflow.log_metric("accuracy",accuracy)

    ## Set a tag we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info","Basic LR model for iris data")

    ## Infer the model signature
    ## The signature represents model input and output as data frames with named columns and data type specified as one of the types
    ## defined in mlflow.types.DataType. i.e. schema of the input and output is fixed over here and it represents that.
    signature = infer_signature(X_train,lr.predict(X_train)) ## X_train will be the input and prediction on X_train will be the output.

    ## log the model (we trained our model using sklearn hence we used mlflow.sklearn, if we use langchain then mlflow.langchain)
    model_info = mlflow.sklearn.log_model(
        sk_model = lr, ## lr is my model name
        name = "iris_model", ## path name(where model will be saved) , artifact_path is depreceated use -> name = "iris_model"
        signature = signature,
        input_example=X_train,
        registered_model_name = "tracking-quickstart"
    )


# In[16]:


model_info.model_uri


# ## Inferencing And Validating Model

# In[17]:


from mlflow.models import validate_serving_input

model_uri = model_info.model_uri

## The model is logged with an input example. MLFlow converts
# it into the serving payload format for the deployed model endpoint
# and saves it to 'serving_input_payload.json'      ## Serving payload is nothing but list of inputs(test inputs)-> just like new test data
## we are trying to do prediction for this
## inputs is list of lists
serving_payload = """{ 
   "inputs":[
     [
     5.7,
     3.8,
     1.7,
     0.3
     ],
     [
     4.8,
     3.4,
     1.6,
     0.2
     ],
     [
     5.6,
     2.9,
     3.6,
     1.3
     ]
   ]
}"""
#Validate the serving payload works on the model
validate_serving_input(model_uri,serving_payload)


# ## Load the model back for prediction as a generic python function model

# In[19]:


## Another and better way of testing
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri) ##This code will load the model as a generic python function
predictions = loaded_model.predict(X_test)

iris_features_name = datasets.load_iris().feature_names

result = pd.DataFrame(X_test,columns=iris_features_name)
result["actual_class"] = y_test
result["predicted_class"] = predictions


# In[20]:


result


# In[21]:


result[:5]  ## Top 5 records


# ## Model Registry
# The MLflow Model Registry component is a centralized model store, set of APIs , and UI, to collaboratively manage the full lifecycle of
# an MLflow Model. It provides model lineage (which MLflow experiment and run produced the model), model versioning , model aliasing , model tagging , and annotations.

# Things are getting tracked in model registry which is available in the MLflow
# While we log the model:
# giving the parameter registered_model_name is not a good practice -> reason is that we should validate whether this is the best model or not.
# If this is the best model then only we should go and register the model name.
# 
# How can we register model in the later stages , after we validate things ??

# In[ ]:


#create a new MLFLOW experiment
mlflow.set_experiment("MLFLOW Quickstart")

## Start an MLFLOW run
with mlflow.start_run():
    ## log the hyperparameters
    mlflow.log_params(params)

    ## Log the accuracy metrics
    mlflow.log_metric("accuracy",1.0) ## instead of accuracy we are giving a hard coded value -> lets say accuracy we got is 100 percent

    ## Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info2","Basic LR model for iris data")

    ## Infer the model signature
    signature = infer_signature(X_train,lr.predict(X_train))

    ## log the model
    model_info = mlflow.sklearn.log_model(
        sk_model = lr,
        name = "iris_model",
        signature = signature,
        input_example = X_train
    )  ## Here we are not registering the model right now , bcoz initially i will validate it w.r.t various parameters we will go ahead and compare
    ## to find which is the best model , then only we will go ahead and register it.
    ## we are logging the model along with the artifact , but not registering it.
    ## Earlier the models were getting saved with name tracking-quickstart with the respective version no. as we were using
    ## registered_model_name , now here since we didn't used any such thing(didn't registered) and just used name = "iris_model" , it will be saved with name 
    ## iris_model only , later if we register this model with name : tracking-quickstart then it will be v5/latest version.
    ## all these are getting created in the experiment which u have created.


# Now when u will go to the mlflow ui -> then under our experiments -> we go to the artifacts section -> there we will be able to see the 
# button saying "Register Model" -> its saying model is not already registered we can register it now. Its asking whether we should register this particular model or not.
# But how should i decide whether i should register or not ??
# Go to experiments and compare all the models. -> then register the best model.
# Select the model with the best accuracy and then register it.
# Select the instance of experiment where that particular model is and then go to artifacts and click register model.
# When we register the model : It will be registered with experiment name along with the instance(version name) :
# tracking-quickstart,v3 -> automatically version 3 will be picked as it had the maximum accuracy.
# Experiements ke andar model/instances aate hai.
# Uss particular experiment me konse se model ka accuracy best tha(named with version number) should be registered.
# We go to experiements -> models -> their we can also give tag to our best model(that option is displayed)
# Models section is other section just like experiments -> this is where we can see our registered model.

# Once we register how do we particularly access the model(which we have registered) or how do we predict from this model ??

# In[23]:


## Inferencing from model from model registry(model registry basically manages all the models)

import mlflow.sklearn
model_name = "tracking-quickstart"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}" ## path to access from the model registry

model = mlflow.sklearn.load_model(model_uri)
model


# In[24]:


model_uri


# In[26]:


y_pred_new = model.predict(X_test)
y_pred_new

