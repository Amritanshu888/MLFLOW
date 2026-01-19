#!/usr/bin/env python
# coding: utf-8

# ### Quickstart: Compare runs , choose a model , and deploy it to a REST API
# 
# In this quickstart , you will:
# 
# - Run a hyperparameter sweep on a training script
# 
# - Compare the results of the runs in the MLflow UI
# 
# - Choose the best run and register it as a model
# 
# - Deploy the model to a REST API
# 
# - Build a container image suitable for deployment to a cloud platform

# When we register a model into the model registry we then convert it into an API(ready for deployment) which we deploy.
# Basically we are deploying a model as a REST API.
# hyperopt -> Library which will allow us to do hyperparameter tuning in ANN

# In[2]:


import keras
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK,Trials,fmin,hp,tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature


# In[ ]:


## load the dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)
data  ## quality is the output feature


# In[ ]:


## Split the data into training , validation and test sets
train,test = train_test_split(data,test_size=0.25,random_state=42)
train    ## quality is the dependent feature and rest other becomes our independent features


# In[5]:


train_x = train.drop(['quality'], axis = 1).values ## Dropping the quality feature as this is the output , 
## remaining all are the input features. Basically we are splitting our independent and dependent features.
## .values for converting it to a array.
train_y = train[['quality']].values.ravel()  ## .values -> 2D array , while .ravel() -> 1D array. No reshaping required

## test dataset(will be treated as new data to do the prediction)
test_x = test.drop(['quality'],axis=1).values  ##independent
test_y = test[['quality']].values.ravel()      ##dependent

## validation data(further splitting the train data)
## splitting this train data into train and validation
train_x,valid_x,train_y,valid_y = train_test_split(train_x,train_y,test_size=0.20,random_state=42)

signature = infer_signature(train_x,train_y) ## Input and output schema passed -> train_x is input schema and train_y is output schema


# In[ ]:


### ANN Model

def train_model(params,epochs,train_x,train_y,valid_x,valid_y,test_x,test_y):
    ## Define model architecture
    mean = np.mean(train_x,axis=0) ## axis = 0 means column wise , its giving us mean of every column -> this is required as we have to perform normalisation while training our artificial neural network
    var = np.var(train_x,axis=0)  ## variance , both mean and variance parameter i have taken and this will be used in my layer normalisation(on input layer)

    model = keras.Sequential(
        [
            keras.Input([train_x.shape[1]]),  ## 11 features passed here
            keras.layers.Normalization(mean=mean,variance=var),
            keras.layers.Dense(64,activation='relu'),
            keras.layers.Dense(1) ## output node
        ]
    )

    ## compile the model
    ## In learning rate their are list of learning rate's , and list of momentum's in momentum
    ## the list will have possible values which we will use to check , we are taking these params as it will help us to log
    ## the best params(log some experiments as we are testing/playing with multiple parameters)
    model.compile(optimizer=keras.optimizers.SGD(
        learning_rate=params["lr"],momentum=params["momentum"]  ## here for every value we will try to track each and every experiment
        ## This is why we are using MLFlow and hyperopt(this is going to check with each and every parameter which is given over their)
    ),
    loss = "mean_squared_error",
    metrics = [keras.metrics.RootMeanSquaredError()]
    )

    ## Train the ANN model with lr and momentum params with MLFLOW tracking and track each of them.
    ## here with mlflow its tracking the evaluation result with each and every parameter
    with mlflow.start_run(nested=True): ## Here we have to try out with multiple parameters nested=True -> nested structure
        model.fit(train_x,train_y,validation_data=[valid_x,valid_y],
        epochs=epochs,
        batch_size=64)

        ## Evaluate the model to find the best model
        eval_result = model.evaluate(valid_x,valid_y,batch_size=64)

        eval_rmse = eval_result[1]

        ## Log the parameters and results
        mlflow.log_params(params) ## log_params as multiple parameters
        mlflow.log_metric("eval_rmse",eval_rmse)

        ## Log the model
        mlflow.tensorflow.log_model(model,"model",signature=signature) ## signature -> defining schema , name = "model"

        return {"loss":eval_rmse,"status":STATUS_OK,"model":model}



# In[7]:


## For Hyperopt we will create a objective function
def objective(params):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        epochs=3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y
    )
    return result


# In[9]:


space = {
    "lr":hp.loguniform("lr",np.log(1e-5),np.log(1e-1)), ## Learning rate ranges between 10^-5 to 10^-1
    "momentum":hp.uniform("momentum",0.0,1.0)
}
## This is the space in which we are going to try


# In[11]:


mlflow.set_experiment("wine-quality")
with mlflow.start_run():
    ## Conduct the hyperparameter search using Hyperopt, Trials library -> will basically perform hyperparameter tuning
    trials = Trials() ## This is the parameter which is set and its set only for hyperparameter search
    best = fmin(
        fn = objective, ## it will call the objective function created above, objective function will call -> train_model function
        space = space,  ## space contains the parameters which will be passed
        algo = tpe.suggest, ## Internally using different types of algorithm based on the suggestion
        max_evals = 4,
        trials = trials
    )

    # Fetch the details of the best run
    best_run = sorted(trials.results,key=lambda x: x["loss"])[0]  ## Will take the one with the minimum loss

    # Log the best parameters, loss, and model
    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse",best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"],"model",signature=signature) ## here name is "model"

    ## Print out the best parameters and corresponding loss
    print(f"Best parameters: {best}")
    print(f"Best eval rmse: {best_run['loss']}")


# Good practice is creating mlruns inside the folder where we are creating our project
# cd into the respective folder and then in terminal do mlflow ui
# 
# Why so many experiemnts ??
# Because of different learning rates values and different momentum values
# 
# After u execute in wine quality u will be able to see so many experiments created (+) icon will be their click that , all the experiments/instances will come.
# 4 different sub experiments were their , why ??
# Because of different learning rates and momentum
# 
# To compare select everything and click compare 
# parameters giving lowest eval_rmse are the best parameters
# The model with the best parameters have to be registered
# Go to that model click register model -> register the model with the name of ur choice
# Then this model will be available in the models section , there we can add tags and also add aliases
# Once we have the model we can load it with using pyfunc and test it on our new test data/do execution

# In[ ]:


## Inferencing 

from mlflow.models import validate_serving_input
model_uri = 'runs:/cca2584b26884c8bba50931ec981a418/model' ## Get the model uri from mlflow server(choose the best model)-> 'runs:/(uri)/model'

from mlflow.models import convert_input_example_to_serving_input
serving_payload = convert_input_example_to_serving_input(test_x)
validate_serving_input(model_uri,serving_payload)


# The output which we are getting here is the quality of the wine

# In[13]:


## Another way of doing the same as above
## Load the model as PyfuncModel.
model_uri = 'runs:/cca2584b26884c8bba50931ec981a418/model'
loaded_model = mlflow.pyfunc.load_model(model_uri)

## Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(test_x))


# In[14]:


## Register in the model registry -> registering manually with code
mlflow.register_model(model_uri,"wine-quality") ## name of the registered model will be "wine-quality"


# Above will create version 1 of wine-quality , -> this will be accessible in the models section of mlflow as it is
# a registered model
