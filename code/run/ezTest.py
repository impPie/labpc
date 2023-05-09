#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, '..')
import pickle,numpy
from _3train.deepClassifier import DeepClassifier
from os import listdir, makedirs
from utils.fileManagement import selectClassifierID
# from eegFileReaderServer import EEGFileReaderServer
# from classifierClient import ClassifierClient
from utils.parameterSetup import ParameterSetup
from os.path import splitext, isfile, exists
from _4predict.stagePredictor import StagePredictor
from _2featureEx.featureExtractorRawDataWithSTFT import FeatureExtractorRawDataWithSTFT 
from _5test.evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat
import time


class EzT:

    def __init__(self, args):
        self.args = args
        self.classifier_type = 'UTSN-L'
        pass

    def start(self):
        # channelOpt = 1
        params = ParameterSetup()
        self.recordWaves = params.writeWholeWaves
        self.extractorType = params.extractorType
        self.classifierType = params.classifierType
        self.postDir = params.postDir
        # self.predDir = params.predDir
        self.classLabels = params.sampleClassLabels[:params.maximumStageNum]
        self.predictionState=1
        # observed_samplingFreq = params.samplingFreq
        # observed_epochTime = params.windowSizeInSec

        finalClassifierDir = params.finalClassifierDir
        
        classifierTypeFileName = 'classifierTypes.csv'
        with open(finalClassifierDir + '/' + classifierTypeFileName) as f:
                    for line in f:
                        classifierID, classifierType, samplingFreq, epochTime = [elem.strip() for elem in line.split(',')]
                        print(classifierID, ',', classifierType, ',', samplingFreq, ',', epochTime)
                        # classifierMetadataList.append((classifierID, classifierType, int(samplingFreq), int(epochTime)))

        paramFileName = 'params.' + str(classifierID) + '.json'

        paramsForNetworkStructure = ParameterSetup(paramDir=finalClassifierDir, paramFileName=paramFileName)

        self.extractor = FeatureExtractorRawDataWithSTFT()
        
        self.y_pred_L=[]
        
        try: 
            classifier = DeepClassifier(self.classLabels, classifierID=classifierID, paramsForDirectorySetup=params, paramsForNetworkStructure=paramsForNetworkStructure)
            model_path = finalClassifierDir + '/weights.' + str(classifierID) + '.pkl'

            print('model_path = ', model_path)
            classifier.load_weights(model_path)

            self.stagePredictor = StagePredictor(paramsForNetworkStructure, self.extractor, classifier, finalClassifierDir, classifierID, params.markovOrderForPrediction)
            
            timeStampSegment=[]
            #------------------------------------------------
            #------------------------------------------------
            with open("../../data/pickled ori/eegAndStage.sixFilesNo1.pkl","rb") as dataFileHandler:
                (eeg, emg, stageSeq, timeStamps) = pickle.load(dataFileHandler)
            
            beeg=eeg.reshape(-1,512)
            # eegSegment = self.one_record[:, 0]
            for eegSegment in beeg:
                if self.predictionState:

                    stagePrediction = self.stagePredictor.predict(
                        eegSegment, timeStampSegment, params.stageLabels4evaluation, params.stageLabel2stageID)

                else:
                    stagePrediction = '?'


                if self.predictionState:
                    # ----
                    # if the prediction is P, then use the previous one
                    if stagePrediction == 'P':
                        # print('stagePrediction == P for wID = ' + str(wID))
                        if len(self.y_pred_L) > 0:
                            finalClassifierDirPrediction = self.y_pred_L[len(
                                self.y_pred_L)-1]
                            # print('stagePrediction replaced to ' + stagePrediction + ' at ' + str(segmentID))
                        else:
                            stagePrediction = 'M'

                    self.y_pred_L.append(stagePrediction)
            
            y_train = stageSeq
            y_pred = self.y_pred_L

            (stageLabels, sensitivity, specificity, accuracy) = y2sensitivity(y_train, y_pred)
            (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_train, y_pred)
            printConfusionMat(stageLabels4confusionMat, confusionMat)

            y_matching = (y_train == y_pred)
            correctNum = sum(y_matching)
            # print('y_train = ' + str(y_train[:50]))
            # print('y_pred = ' + str(y_pred[:50]))
            # print('correctNum = ' + str(correctNum))
            y_length = y_pred.shape[0]
            precision = correctNum / y_length
            for labelID in range(len(stageLabels)):
                print('  stageLabel = ' + stageLabels[labelID] + ', sensitivity = ' + "{0:.3f}".format(sensitivity[labelID]) + ', specificity = ' + "{0:.3f}".format(specificity[labelID]) + ', accuracy = ' + "{0:.3f}".format(accuracy[labelID]))
            print('  precision = ' + "{0:.5f}".format(precision) + ' (= ' + str(correctNum) + '/' + str(y_length) +')')
            print('')

        except Exception as e:
            print(str(e))
            raise e
        


if __name__ == '__main__':
    
    start_time = time.time()

    args = sys.argv
    mainapp = EzT(args)
    mainapp.start()
    # print(datetime.datetime.now())
    print("--- %s minutes ---" % (int(time.time() - start_time)/60))
    # while True:
    # print('*')
    # time.sleep(5)
    # sys.exit(app.exec_())
