# Disaster_Response_Project

This project is part of Udacity's Data Science Nanodegree program. In this project, we will build a natural language processing (NLP) model to categorize messages in real time when a disaster strikes.
# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run your web app: `python run.py`

3. Go to web app: http://0.0.0.0:3000/
   
# Photos attached

![Sample Input](https://github.com/jhon27195/Disaster_Response/blob/19ce28701a577957a326b64bc9abd87001a1c5bf/WebApp.PNG)
