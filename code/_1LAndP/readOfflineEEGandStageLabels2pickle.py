import sys,os
# import importlib
# sys.path.insert(0, '..')
sys.path.insert(0, sys.path[0]+"/../")

# print("+++++++================================================+++++++++++++++++++++++++")
# print(sys.path)
# print("+++++++================================================+++++++++++++++++++++++++")


# print(os.getcwd())
from utils.dataReader import DataReader
# from dataReader import DataReader



dataReader = DataReader()
dataReader.readAll(sys)
#readOfflineEEGandStageLabels2pickle.py reads text files containing EEG raw data signals and ground truth stage labels from the WAVEDIR directory. It writes files starting with "eegAndStage" into the "data/pickled" directory. These files are in Python's pickle format to enable faster access.