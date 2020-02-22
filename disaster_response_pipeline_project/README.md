# Disaster Response Pipeline Project

## Project Description

In this Project, a classifier to categorize disaster messages is trained from a data set containing real messages that were sent during disaster events.
To this end a machine learning pipeline to categorize these events is implemented, so that you can classify the messages and send them to an appropriate disaster relief agency.

An ETL pipeline is created to load data, clean data, and save data in a sqlite database, which can be parsed to the ML pipeline for training the classifier.

The ML pipeline load data from the sqlite database and split the data into training and testing data. 
A MultiOutputClassifier is trained using GridSearch optimization to find the best parameters. The classifier is evaluated using the test data. The final model is saved as a pickle file

##Imbalanced support
In the training data the number of actual occurrences of many catagories are very small. 
Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and 
could indicate the need for stratified sampling or rebalancing. 
For details please refer to the classification report result in the end of the readme document.


## Project Files:

/data 
    - DisasterResponse.db: the sqlite3 databse file that is output by ETL pipeline 
    - process_data.py: the ETL pipeline script

/models 
    - classifier.pkl: the classifier pickle file
    - train_classifier.py: ML pipeline script that prepare training/testing data, builds, trains, test and saves the classifier.

/app 
    - run.py: the script that runs the flask app 
    - /templates: the default page files for the app 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

##classification report


                   precision    recall  f1-score   support

               request       0.92      0.07      0.14       903
                 offer       0.00      0.00      0.00        24
           aid_related       0.82      0.33      0.47      2208
          medical_help       0.83      0.01      0.02       421
      medical_products       1.00      0.01      0.01       269
     search_and_rescue       1.00      0.01      0.03       146
              security       0.00      0.00      0.00       101
              military       0.00      0.00      0.00       180
           child_alone       0.00      0.00      0.00         0
                 water       1.00      0.01      0.02       341
                  food       1.00      0.01      0.03       600
               shelter       0.80      0.01      0.02       495
              clothing       0.00      0.00      0.00        76
                 money       0.00      0.00      0.00       120
        missing_people       0.00      0.00      0.00        73
              refugees       0.00      0.00      0.00       168
                 death       1.00      0.02      0.03       249
             other_aid       1.00      0.00      0.00       687
infrastructure_related       1.00      0.00      0.01       329
             transport       1.00      0.02      0.03       227
             buildings       1.00      0.01      0.03       279
           electricity       1.00      0.01      0.02       115
                 tools       0.00      0.00      0.00        39
             hospitals       0.00      0.00      0.00        40
                 shops       0.00      0.00      0.00        24
           aid_centers       0.00      0.00      0.00        52
  other_infrastructure       0.00      0.00      0.00       234
       weather_related       0.91      0.17      0.29      1479
                floods       0.86      0.03      0.05       432
                 storm       0.92      0.02      0.05       489
                  fire       0.00      0.00      0.00        61
            earthquake       0.87      0.10      0.18       491
                  cold       0.00      0.00      0.00       110
         other_weather       1.00      0.00      0.01       299
         direct_report       0.94      0.05      0.09      1037

           avg / total       0.82      0.10      0.15     12798