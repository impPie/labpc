from __future__ import print_function
# import sys
# sys.path.insert(1,'..')
# from os import listdir
import pickle
import numpy as np
import string
import random
from utils.fileManagement import getEEGAndFeatureFiles, findClassifier, writeTrainFileIDsUsedForTraining, getEEGAndFeatureFilesByClassifierID, getEEGAndFeatureFilesByExcludingFromTrainingByPrefix, getEEGAndFeatureFilesByExcludingTestMouseIDs,crossFoldsEEGAndFeatureFiles

#-----------------------
def extract(featureFilePath, stageFilePath):
    featureFileHandler = open(featureFilePath, 'rb')
    features = pickle.load(featureFileHandler)
    featureFileHandler.close()
    stageFileHandler = open(stageFilePath, 'rb')
    (eeg, emg, stageSeq, timeStamps) = pickle.load(stageFileHandler)
    stageFileHandler.close()
    fLen, sLen = features.shape[0], len(stageSeq)
    # fLen, sLen = features.shape[0], len(stageSeq)
    # Below is for the case that not all of the time windows have been labeled.
    # In such a case, stageSeq is shorter than featureArray
    return (features[:sLen] if fLen != sLen else features), np.array(stageSeq)

# def resampleConsecutive(x, y, resampleNumPerMouse):
    # print('*** in resampleConsecutive: x.shape = ', x.shape)
    # print('    in resampleConsecutive: y.shape = ', y.shape)
#    orig_sampleNum = x.shape[0]
#    print('    orig_sampleNum = ', orig_sampleNum, ', resampleNumPerMouse = ', resampleNumPerMouse)
#    randStart = np.random.randint(0, orig_sampleNum - resampleNumPerMouse)
    # print('    randStart = ', randStart)
#    randEnd = randStart + resampleNumPerMouse
#    return x[randStart:randEnd], y[randStart:randEnd]

# def getResampleNumPerMouse(mouseNum, maxSampleNum):
#    return np.int(np.floor(1.0 * maxSampleNum / mouseNum))

#--------------------------------------------
# connect samples and train a model
# def connectSamplesAndTrain(params, paramID, classifier, featureAndStageFileFullPathsL):
def connectSamplesAndTrain(params, fileTripletL, stage_restriction, paramID=0):
    print('in classifierTrainer.connectSamplesAndTrain(), params.networkType =', params.networkType)
    if params.networkType == 'cnn_lstm':
        print('using cnn_lstm in connectSamplesAndTrain')
        if params.classifierType == 'deep':
            subseqLen = params.torch_lstm_length
        else:
            subseqLen = 1
        x_train, y_train = [], []
        for fileCnt, (eegAndStageFile, featureFile, fileID) in enumerate(fileTripletL):
            # print('fileCnt = ' + str(fileCnt) + ': for training, added ' + featureFile)
            featureFileFullPath = params.featureDir + '/' + featureFile
            stageFileFullPath = params.eegDir + '/' + eegAndStageFile
            (x, y) = extract(featureFileFullPath, stageFileFullPath)
            # x = trimFeatures(params, x)
            # print('after trimming, x.shape =', x.shape)
            x = np.array([x]).transpose((1,0,2))
            # y = restrictStages(params, y, params.maximumStageNum)
            y = stage_restriction(y)
            # print('before subseq extraction, x.shape =', x.shape)
            for offset in range(0, len(x)-subseqLen):
                x_subseq = x[offset:offset+subseqLen, :, :]
                x_train.append(x_subseq)
                y_train.append(y[offset+subseqLen])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        # print('eegAndStageFile = ', eegAndStageFile, ', x_train.shape = ', x_train.shape)
        classLabels = np.unique(y_train)
    else:
        for fileCnt, (eegAndStageFile, featureFile, fileID) in enumerate(fileTripletL):
            # print('fileCnt = ' + str(fileCnt) + ': for training, added ' + featureFile)
            featureFileFullPath = params.featureDir + '/' + featureFile
            stageFileFullPath = params.eegDir + '/' + eegAndStageFile
            (x, y) = extract(featureFileFullPath, stageFileFullPath)
            ### if resampleNumPerMouse > 0:
            ###    x, y = resampleConsecutive(x, y, resampleNumPerMouse)
            # print('trainFileID = ' + str(trainFileID))
            if fileCnt == 0:
                x_train = x
                y_train = y
            else:
                x_train = np.append(x_train, x, axis=0)
                y_train = np.append(y_train, y)
                # print('eegAndStageFile = ' + eegAndStageFile + ', x_train.shape = ' + str(x_train.shape))
        # x_train = trimFeatures(params, x_train)
        # print('%%% after trimming, x_train.shape =', x_train.shape)
        # y_train = restrictStages(params, y_train, params.maximumStageNum)
        y_train = stage_restriction(y_train)
        classLabels = np.unique(y_train)
    ### (x_train, y_train) = supersample(x_train, y_train)
    print(' ')
    print('For training:')
    print('  x_train.shape = ' + str(x_train.shape))
    print('  y_train = ' + str(y_train))
    print('  y_train.shape = ' + str(y_train.shape))

    # print('before calling findClassifier. params.networkType =', params.networkType)
    classifierID = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    params.writeAllParams(params.classifierDir, classifierID)
    
    writeTrainFileIDsUsedForTraining(params, classifierID, fileTripletL)
    classifier = findClassifier(params, paramID, classLabels, classifierID)
    classifier.train(x_train, y_train)
    if params.classifierType != 'deep':
        classifierFileName = params.classifierDir + '/' + params.classifierPrefix + '.' + classifierID + '.pkl'
        with open(classifierFileName, 'wb') as classifierFileHandler:
            pickle.dump(classifier, classifierFileHandler)
    return classifierID

#-----------------------
def trainClassifier(params, outputDir, optionType, optionVals):
    #  '5fold cross vertification':
    if optionType == '-cf' or optionType == '':

        train_fileTripletL, test_fileTripletL = crossFoldsEEGAndFeatureFiles(
            params, testNum, offset, randomize)
        for i in range(len(test_fileTripletL)):
            if len(train_fileTripletL[i]) > 0:
                def stage_restriction(orig_stageSeq):
                    return orig_stageSeq
                connectSamplesAndTrain(params, train_fileTripletL[i], stage_restriction)
            else:
                print('%%% No file for training.')
        return
    
    if optionType == '-o':
        testNum, offset = optionVals# set test and train number 
        randomize = False
        train_fileTripletL, test_fileTripletL = getEEGAndFeatureFiles(params, testNum, offset, randomize)
    if optionType == '-r':
        testNum = optionVals[0]  # set test number, choose randomly
        offset, randomize = 0, True
        train_fileTripletL, test_fileTripletL = getEEGAndFeatureFiles(params, testNum, offset, randomize)
    elif optionType == '-p':
        classifierIDforTrainingFiles = optionVals[0]
        train_fileTripletL, test_fileTripletL = getEEGAndFeatureFilesByClassifierID(params, classifierIDforTrainingFiles)
    elif optionType == '-e':   # specify excluded files
        test_file_prefix = optionVals[0]
        train_fileTripletL, test_fileTripletL = getEEGAndFeatureFilesByExcludingFromTrainingByPrefix(params, test_file_prefix)
    
   
    if len(train_fileTripletL) > 0:
        def stage_restriction(orig_stageSeq):
            return orig_stageSeq
        connectSamplesAndTrain(params, train_fileTripletL, stage_restriction)
    else:
        print('%%% No file for training.')
