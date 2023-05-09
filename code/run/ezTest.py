#!/Users/ssg/.pyenv/shims/python
# -*- coding: utf-8 -*-

from _3train.deepClassifier import DeepClassifier
from os import listdir, makedirs
from utils.fileManagement import selectClassifierID
from eegFileReaderServer import EEGFileReaderServer
from classifierClient import ClassifierClient
from utils.parameterSetup import ParameterSetup
from os.path import splitext, isfile, exists
from stagePredictor import StagePredictor
import sys
sys.path.insert(1, '..')


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
        # self.predDir = params.predDir
        self.finalClassifierDir = params.finalClassifierDir
        observed_samplingFreq = params.samplingFreq
        observed_epochTime = params.windowSizeInSec

        
        try:
               
                classifierID = ''
                classifier = DeepClassifier(self.classLabels, classifierID=classifierID, paramsForDirectorySetup=self.params, paramsForNetworkStructure=paramsForNetworkStructure)
                model_path = finalClassifierDir + '/weights.' + str(classifierID) + '.pkl'

                print('model_path = ', model_path)
                classifier.load_weights(model_path)

                self.stagePredictor = StagePredictor(paramsForNetworkStructure, self.extractor, classifier, finalClassifierDir, classifierID, self.params.markovOrderForPrediction)
                
                timeStampSegment=[]
                # eegSegment = self.one_record[:, 0]
                for eegSegment in pkddata:
                    if self.predictionState:
                        # stageEstimate is one of ['w', 'n', 'r']

                        stagePrediction = self.stagePredictor.predict(
                            eegSegment, timeStampSegment, self.params.stageLabels4evaluation, self.params.stageLabel2stageID)

                    else:
                        stagePrediction = '?'

                    # update prediction results in graphs by moving all graphs one window

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


        except Exception as e:
            print(str(e))
            raise e
        


if __name__ == '__main__':
    args = sys.argv
    mainapp = RemOfflineApplication(args)
    mainapp.start()
    # while True:
    # print('*')
    # time.sleep(5)
    # sys.exit(app.exec_())
