import sys
sys.path.insert(1, sys.path[0]+"/../")

# print(sys.path)
# sys.path.insert(1, sys.path[0]+"/../")

# sys.path.insert(1,'..')
import time

from os import listdir
from utils.parameterSetup import ParameterSetup
from utils.fileManagement import getFileIDs
# from _2featureEx.algorithmFactory import AlgorithmFactory
from _2featureEx.featureExtractorRawDataWithSTFT import FeatureExtractorRawDataWithSTFT 
from _1LAndP.outlierMouseFilter import OutlierMouseFilter

#---------------
# set up parameters
args = sys.argv
if len(args) > 1:
    # if args[1]!='-o':
    #     option = " "+ args[1]
    # else:
    #     option = args[1]
    option = args[1]

else:
    option = ''
start_time = time.time()

# get params shared by programs
params = ParameterSetup()
useEMG = params.useEMG
# eegDir = "data/pickled del"
eegDir = params.eegDir
print("eegDir: "+eegDir)
featureDir = params.featureDir
pastStageLookUpNum = params.pastStageLookUpNum

extractorType = params.extractorType
# factory = AlgorithmFactory(extractorType)
# extractor = factory.generateExtractor()
extractor = FeatureExtractorRawDataWithSTFT()

oFilter = OutlierMouseFilter()

#---------------
# print out parameter setting for feature extraction

if useEMG:
    print('using EMG.')
    label4EMG = params.label4withEMG
else:
    label4EMG = params.label4withoutEMG
    print('not using EMG')

print('pastStageLookUpNum = ' + str(pastStageLookUpNum))

#---------------
# read pickled files that stores EEG and stage labels

prefix = 'eegAndStage'

fileIDs = getFileIDs(eegDir, prefix)
### fileIDs = ['HET-NR-D0717', 'DBL-NO-D1473', 'HET-NO-D0905']

for fileID in fileIDs:
    print('fileID = ' + str(fileID))
    featureFileName = params.featureFilePrefix + '.' + params.extractorType + '.' + label4EMG + '.' + fileID + '.pkl'
    if oFilter.isOutlier(fileID):
        print('file ' + fileID + ' is an outlier, so skipping.')
    else:
        flag4extraction = 1
        if option != '-o':
            for fileName in listdir(featureDir):
                if fileName == featureFileName:
                    flag4extraction = 0
                    print(featureFileName + ' already exists, so skipping')
                    break
        if flag4extraction:
            extractor.featureExtraction(params, fileID)
print("--- %s minutes ---" % (int(time.time() - start_time)/60))