# Predicting from CUI and without activating GUI

Sleep stages for wave files in the "data/aipost" directory can be predicted without activating the GUI. Instead of running app.py, run

'python offline.py'

The predicted result is written out as files in the "data/prediction" directory.

# Evaluating the results
To evaluate the result of prediction, use

'python eval_offline.py PREDICTION_FILE JUDGE_PATH'

where PREDICTION_FILE is the name of the prediction file in "data/prediction", and JUDGE_PATH is the path to the Judge file that contain ground-truth labels for each epoch.

# Parameter setup
The "data/params" directory should contain "params.json" for setting up parameters for feature extraction, training, and prediction.

Be editing "params.json", the behavior of the GUI and also of training can be altered.

# Training and testing a classifier
A new classifier can be trained and validated by the following procedure:

``` 
python readOfflineEEGandStageLabels2pickle.py WAVEDIR
python extractFeatures.py
python trainClassifier.py
python computeTrainingError.py
python computeTestError.py
```
## readOfflineEEGandStageLabels2pickle.py 
reads text files containing EEG raw data signals and ground truth stage labels from the WAVEDIR directory. It writes files starting with "eegAndStage" into the "data/pickled" directory. These files are in Python's pickle format to enable faster access.

```
sleepstages/
  code/
  data/
    params
    pickled
    waves
    WAVEDIR/
      Raw/
      Judge/
```

## extractFeatures.py 
reads "eegAndStage" files and write files starting with "features". These files contain feature vectors used for training classifiers. They will be generated in "data/pickled".

## trainClassifier.py 
reads "features" and writes files starting with "weights", "params", and "files_used_for_training". These files contain randomly generated six-character IDs (i.e., classifier IDs) in their file names.

## The "weights" file 
contains weight parameters obtained from training. The "params" file is a copy of "params.json" in "data/params" that is intended to save the parameters used for training the classifier. "files_used_for_training" indicates which recordings were used for training that classifier. These files are excluded when testing the classifier.