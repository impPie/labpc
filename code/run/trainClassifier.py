from __future__ import print_function
# import sys
# sys.path.insert(1,'..')
import sys,time
import datetime
sys.path.insert(1, sys.path[0]+"/../")

# print("+++++++================================================+++++++++++++++++++++++++")
# print(sys.path)
# print("+++++++================================================+++++++++++++++++++++++++")

import json
from os import listdir
from utils.parameterSetup import ParameterSetup
from _3train.classifierTrainer import trainClassifier

start_time = time.time()
args = sys.argv
optionType = args[1] if len(args) > 2 else ''
optionVals = args[2:] if len(args) > 2 else [0]
print('optionType =', optionType, ', optionVals = ', optionVals)
# read params.json
pathFilePath = open('../path.json')
p = json.load(pathFilePath)
paramDir = p['pathPrefix'] + '/' + p['paramsDir']
outputDir = ''

for paramFileName in listdir(paramDir):
    if paramFileName.startswith('params.') and not paramFileName.endswith('~'):
        print('paramFileName =', paramFileName)
        params = ParameterSetup(paramDir, paramFileName, outputDir)
        trainClassifier(params, outputDir, optionType, optionVals)
        
print(datetime.datetime.now())
print("--- %s minutes ---" % (int(time.time() - start_time)/60))
