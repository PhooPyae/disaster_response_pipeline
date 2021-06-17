# Disaster Response Pipeline Project

![wordcloud](https://user-images.githubusercontent.com/20230956/122337103-e247cb80-cf63-11eb-8fb7-6e09553543f7.png)

A disaster is a serious or deadly problem occurring over a short or long period of time. That typically causes widespread material, human, economic, or environmental loss.

## Table of Contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info

This project is to help emergency worker to understand the messages that were sent during disaster. The emergency worker can input the messages and get classification results in different categories.

The disaster data are supported by <b> Figure Eight </b>
The project works on two separate datasets: messages dataset and categories dataset.

[Project Structure]
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # preprocess the data to get ready for training
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py #train the model using RandomForestClassifier
|- model.pkl  # saved model 

- README.md
```

## Technologies
The following libraries are used:
* sqlalchemy
* pandas
* numpy
* pickle
* nltk
* matplotlib
* sys
* sklearn
* flask
* and so on.

## Setup

You need to install all the libraries necessary for running the project files. You can install it using <b>conda</b> or  <b>pip</b>.
```
$ conda install pip
$ pip install sqlalchemy pandas pickle ...
```
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves
        
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`
        
[Remark] Unzip the model file before you test it.

2. Run the following command in the app's directory to run your web app.
    
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
