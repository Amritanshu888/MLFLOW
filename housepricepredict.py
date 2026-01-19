#!/usr/bin/env python
# coding: utf-8

# ## House Price Prediction With MLFLOW
# ## In this we will:
# - Run a Hyperparameter tuning while training a model
# - Log every Hyperparameter and metrics in the MLFLOW UI
# - Compare the results of the various runs in the MLflow UI
# - Choose the best run and register it as a model

# In[2]:


import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing)


# In[3]:


housing


# In[7]:


## Preparing the dataset
data = pd.DataFrame(housing.data,columns=housing.feature_names)  ##input features
data['Price'] = housing.target ## Creating a output feature 'Price' , which has target values -> new column 'Price' created
data.head(10)


# ## Train test split , Model Hyperparameter Tuning , MLFLOW Experiments

# In[9]:


from urllib.parse import urlparse  ## This library will be used in mlflow experiments itself
## Independent and Dependent features
X = data.drop(columns=["Price"]) ## dropping the Price column to get independent features
y = data["Price"] ## This is our dependent feature(target variable)


# In[10]:


## Hyperparameter Tuning using Grid SearchCV
def hyperparameter_tuning(X_train,y_train,param_grid):
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2,scoring="neg_mean_squared_error")
    ## "neg_mean_squared_error" is for regression problem statement
    grid_search.fit(X_train,y_train)
    return grid_search


# In[ ]:


## Split our data into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

from mlflow.models import infer_signature ## Why infer_signature -> So that we set our schema w.r.t our input and output
signature = infer_signature(X_train,y_train) ## input schema is X_train(independent features) and output schema is y_train(output feature)

## Define the hyperparameter grid
param_grid = {
    'n_estimators': [100,200],
    'max_depth': [5,10,None],
    'min_samples_split': [2,5],
    'min_samples_leaf': [1,2]
}

## start the MLFLOW Experiements
with mlflow.start_run():
    ## Perform hyperparameter tuning
    grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

    ## Get the best model
    best_model = grid_search.best_estimator_

    ## Evaluate the best model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)

    ## Log best parameters and metrics
    mlflow.log_param("best_n_estimators",grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth",grid_search.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split",grid_search.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf",grid_search.best_params_['min_samples_leaf'])
    mlflow.log_metric("mse",mse)
    ## here we can log other parameters also like confusion matrix

    ## Tracking url
    ## Note if u want to make mlruns folder in the current folder where u are executing ur code u have to do
    ## cd 1-MLproject  then in that u have to do mlfow ui in terminal , after this mlruns folder will be created in the current folder
    ## where u are executing ur code.
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme  ## Parsing the uri and trying to get scheme
    ## if we don't set the tracking uri and execute the above line code we will get a file -> not assigned any uri
    ## if we set the tracking uri then execute we will get the scheme of this uri. this is what urlparse is doing
    ## yha pe experiemnt set nhi kiya isliye new experiment create nhi hua

    if tracking_url_type_store != 'file':
        mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Randomforest Model")
    else:
        mlflow.sklearn.log_model(best_model,"model",signature=signature)   ## jab 'file' type hai i.e. it means their is no tracking server i.e. we are not registering, their we pass signature.

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse}")    

