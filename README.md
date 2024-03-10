# Disaster_Response_Project
![Sample Input](https://github.com/jhon27195/Disaster_Response/blob/9992ee3fd947f6261169adc86aa51f0bbb619e31/Disaster.jpg)
This project is part of Udacity's Data Science Nanodegree program. In this project, we will build a natural language processing (NLP) model to categorize messages in real time when a disaster strikes.

In this project, a web application will be developed where the user can enter a message and obtain results of different classifications.

This web application will be of great importance so that emergency operators can take advantage when a disaster occurs to classify text messages into various categories and that the emergency can be channeled in a timely manner with the corresponding entity.
# File Description
- App: The app folder contains "run.py" for the web application.
- data: It contains the data set within the pipeline process that cleans the data and stores it in the database.
- models: The folder that contains the developed model
In addition, the ETL and ML Pipeline Notebooks have been added.

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


2. Run your web app: `python run.py`

3. Go to web app: http://0.0.0.0:3000/
   
# Photos attached
Web application where you can enter a message and get ranking results that have been worked on.
![Sample Input](https://github.com/jhon27195/Disaster_Response/blob/19ce28701a577957a326b64bc9abd87001a1c5bf/WebApp.PNG)
