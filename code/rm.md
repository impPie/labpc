# Predicting from CUI and without activating GUI

Sleep stages for wave files in the "data/aipost" directory can be predicted without activating the GUI. Instead of running app.py, run

'python offline.py'

The predicted result is written out as files in the "data/prediction" directory.

# Evaluating the results
To evaluate the result of prediction, use

'python eval_offline.py PREDICTION_FILE JUDGE_PATH'

where PREDICTION_FILE is the name of the prediction file in "data/prediction", and JUDGE_PATH is the path to the Judge file that contain ground-truth labels for each epoch.