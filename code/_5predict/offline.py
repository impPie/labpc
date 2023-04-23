#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1,'..')
from os import listdir
from os.path import splitext, isfile
from _7utils.parameterSetup import ParameterSetup
from classifierClient import ClassifierClient
from _5predict.eegFileReaderServer import EEGFileReaderServer
from _7utils.fileManagement import selectClassifierID

class RemOfflineApplication:

    def __init__(self, args):
        self.args = args
        self.classifier_type = 'UTSN-L'
        pass

    def start(self):
        channelOpt = 1
        params = ParameterSetup()
        self.recordWaves = params.writeWholeWaves
        self.extractorType = params.extractorType
        self.classifierType = params.classifierType
        self.postDir = params.postDir
        self.predDir = params.predDir
        self.finalClassifierDir = params.finalClassifierDir
        observed_samplingFreq = params.samplingFreq
        observed_epochTime = params.windowSizeInSec

        # eegFilePath = args[1]
        # inputFileID = splitext(split(eegFilePath)[1])[0]
        postPath = listdir(self.postDir)
        predPath = listdir(self.predDir)
        # postFiles = sorted(listdir(self.postDir))
        # fileCnt = 0
        for inputFileFolder in postPath:
        
            for inputFileName in listdir(self.postDir+"/"+inputFileFolder):
                if not inputFileName.startswith('.'):
                    print('inputFileName = ' + inputFileName)
                    inputFileID = splitext(inputFileName)[0]
                    print('inputFileID = ' + inputFileID)
                    predFileFullPath = self.predDir + '/' +inputFileFolder+ '/'+ inputFileID + '_pred.txt'
                    print('predFileFullPath = ' + predFileFullPath)

                    if not isfile(predFileFullPath):
                        # fileCnt += 1
                        print('  processing ' + inputFileID)
                        try:
                            classifierID, model_samplingFreq, model_epochTime = selectClassifierID(self.finalClassifierDir, self.classifier_type)
                            classifierID = inputFileFolder #
                            if len(self.args) > 1:
                                if self.args[1] == '--output_the_same_fileID':
                                    self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID, inputFileID=inputFileID,
                                                                    samplingFreq=model_samplingFreq, epochTime=model_epochTime)
                                else:
                                    if self.args[1] == '--samplingFreq' and len(self.args) > 2:
                                        observed_samplingFreq = int(self.args[2])
                                    self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID,
                                        samplingFreq=model_samplingFreq, epochTime=model_epochTime)
                            else:
                                self.client = ClassifierClient(self.recordWaves, self.extractorType, self.classifierType, classifierID,
                                    inputFileID=inputFileID,samplingFreq=model_samplingFreq, epochTime=model_epochTime)# same id
                            self.client.predictionStateOn()
                            self.client.hasGUI = False
                            # sys.stdout.write('classifierClient started by ' + str(channelOpt) + ' channel.')

                        except Exception as e:
                            print(str(e))
                            raise e

                        try:
                            eegFilePath = self.postDir+"/"+inputFileFolder+ '/' + inputFileName
                            self.server = EEGFileReaderServer(self.client, eegFilePath, model_samplingFreq=model_samplingFreq, model_epochTime=model_epochTime,
                                observed_samplingFreq=observed_samplingFreq, observed_epochTime=observed_epochTime)

                        except Exception as e:
                            print(str(e))
                            raise e

                    else:
                        print('  skipping ' + inputFileID + ' because ' + predFileFullPath + ' exists.')


if __name__ == '__main__':
    args = sys.argv
    mainapp = RemOfflineApplication(args)
    mainapp.start()
    # while True:
        # print('*')
        # time.sleep(5)
    # sys.exit(app.exec_())
