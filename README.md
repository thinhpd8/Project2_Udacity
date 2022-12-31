# Project2_Udacity
Project 2: Disaster Response Pipelines
Submission of Project 2 in Data Scientist Nanodegree Program - Udacity

### Installations
The code was developed using Python version 3.x. We recommend using Anaconda for installing Python 3 as well as other required libraries, although you can install them by other means.

Other required libraries that are not included in the Anaconda package:

plotly
sqlalchemy
Trained model can be downloaded here, put it under trained_models/
### Project Motivation
There are generally millions of communications after a disaster, either directly or via news and social media. Organizations that deal with disasters might not have the resources necessary to address these notifications manually. 
Additionally, many organizations will handle various aspects of the issue. Therefore, a system for automatically classifying catastrophe responses is required to ensure that the issues are dealt with quickly and effectively by the appropriate parties. 
In this research, a multi-output machine learning model is trained using the disaster response data compiled by Figure Eight. A web application with an easy-to-use interface and visualizations is then created using the model.


###File Descriptions
app

| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # data cleaning pipeline
|- InsertDatabaseName.db # database to save clean data to

models

|- train_classifier.py # machine learning pipeline
|- classifier.pkl # saved model

README.md

###Components
There are three components I completed for this project.

1. ETL Pipeline
A Python script, process_data.py, writes a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
A jupyter notebook ETL Pipeline Preparation was used to do EDA to prepare the process_data.py python script.

2. ML Pipeline
A Python script, train_classifier.py, writes a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
A jupyter notebook ML Pipeline Preparation was used to do EDA to prepare the train_classifier.py python script.


### How to use:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/
