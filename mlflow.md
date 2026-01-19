MLFlow
1. Core Components of MLFlow
2. Why use MLFlow
3. Who uses MLFLow
4. Usecase of MLFlow

Life Cycle of a data science project:
Data Preparation(involves data engineer->creating ETL pipelines, with help of ETL pipeline they aggregate data from various sources like APIs , databases and store it in some kind of source and the source can be a database like MongoDB or any other database) -> Exploratory Data Analysis -> Feature Engineering -> Model Training(also includes hyperparameter tuning, how different models give different accuracy) -> Model Validation(validate model based on some parameter) -> Deployment(deploy on various cloud platforms like GCP,Azure,AWS) -> Monitoring(implement concepts like data drift, which will actually help u to understand when ur model needs to be retrained)

Data Engineer -> Mostly involved in data preparation step
Data Scientist -> Involved in EDA,FE,Model Training and Model Validation
ML Engineer -> Involved in Model Validation, Deployment , Monitoring 

What does Data Scientist Do with MLFlow ??
Data Scientist leverages MLFlow for :
1. Experiment Tracking and Hypothesis Testing(EDA,FE,Model Training , Model Validation)-> EDA(Statistical Analysis needs to be performed->i.e hypothesis testing needs to be done(multiple statistial analysis needs to be done based on the data , problem statement that we have)->for each and every statistical analysis we will try to perform experiment tracking , to track each and every details in such a way that we understand tracking is happening in efficient way with versioning considering in place)  --> This is where MLFlow as an MLOPs platform will be used.

2. Code Structuring : We create a pipeline(w.r.t EDA,FE,Model Training and Model Validation) -> With MLFlow this pipeline creation and code structuring  will be much better when compared to the previous way.

3. Model Packaging and Dependency Management: When we create and end to end project we create some files like setup.py files , with setup.py files we package our entire application along with the dependencies that we specifically have.
W.r.t model packaging mlflow provides more features which help us do model packaging and dependency management in a more efficent way.

4. Evaluating Hyperparameter Tuning: (Strongest Feature of MLFLOW) Let's say i will train a specific model, -> Training a model i have lot of parameters which i can use -> To use multiple parameters efficiently i will be using hyperparameter tuning . What mlflow does is that , it helps u to track the hyperparameter tuning w.r.t every parameter(every combination)-> This tracking we can see in form of visualization graph.

5. Can Compare Results of Models Retraining over time(since versioning,experimenting basically is happening): By this we will be able to select most optimal model for deployment.

MLOPs Professional/ML engineer(they are basically part of model validation,deployment and monitoring), how they use MLFlow ??
1. They manage the lifecycle of trained models , both pre and post deployement : From versioning to making sure that we move them from staging to production environment and we do model registry(very imp. concept) thing.

2. Deploy the models securely to the production environment.

3. Manage deployment dependency (what all pacakges required , what all dependencies are required all get completely managed).


MLOPs have also integrated prompt engineering techniques.
Prompt engineering user's mlflow usage(here we can do experiment tracking w.r.t to differnet different prompts) :
1. Evaluate an experiment even with large language models.
2. Create custom prompts and experiment with them.
3. Deciding on the best base model suitable for project requirements.


Usecases of MLFlow:
1. Experiment Tracking: We will be able to track each and every parameters and metrics(performance metrics like log loss,r2 metrics) , when we train a model. MLFlow provides a UI to see all these types of metrics.
2. Model Selection and Deployment: In this MLOPs engineers employ the model UI again , to assess the top performing models. Since every experiment is getting tracked so mlops uses this UI to track and select the best model and then do the deployment.
3. Model Performance Monitoring.
4. MLFLOW allows to work in a collaborative project(working in a team->MLFlow is the best platform->bcoz everything is tracked and it will be displayed in some UI-> Every developer will be able to access this UI where all the tracking experiments are specifically done).

eg. Model training ke time pe MLFlow ka use karke we can track the accuracy,error,training accuracy,test accuracy and other metrics.
eg. w.r.t model validation we can track which is our best model
After this model validation let's say all the results are displayed in a UI -> MLFlow UI. Then MLOPs engineer will try to see this particular UI-> Find out the best model over here -> Then try to deploy it.
