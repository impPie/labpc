import numpy as np
import sys,os
# import sys
sys.path.insert(1,'..')
from _7utils.parameterSetup import ParameterSetup
from _6test.evaluationCriteria import y2sensitivity, y2confusionMat, printConfusionMat

args = sys.argv
params = ParameterSetup()

prePath = params.predDir

# prePath = args[1]
testFilePath = args[2]


# prefds = (os.listdir(prePath))
for fd in os.listdir(prePath):
    y_pred = []
    for pFiles in os.listdir(prePath+"/"+fd):
        with open(prePath + '/' + fd+ '/' +pFiles, 'r') as predFile:
            for line in (predFile):
                p = line.rstrip()
                p = 'S' if p == '1' else p
                # print('pred:', p)
                y_pred.append(p)

    y_test = []
    Files = os.listdir(testFilePath+"/"+fd)
    # Files = os.listdir(testFilePath)
    for f in Files:
        print(f)
        with open(testFilePath+'/'+f) as testFile:
            for i in range(params.metaDataLineNumUpperBound4stage):    # skip lines that describes metadata
                line = testFile.readline()
                if line.startswith(params.cueWhereStageDataStarts):
                    break
                if i == params.metaDataLineNumUpperBound4stage - 1:
                    print('metadata header for stage file was not correct.')
                    quit()
            for line in testFile:
                elems = line.split(',')
                if len(elems) > 2:
                    t = elems[2]
                    # print('test:', t)
                    t = 'W' if (t == 'RW' or t == 'l') else t
                    t = 'H' if t =='h' else t
                    y_test.append(t)

    y_pred = np.array(y_pred[11:])
    y_test = np.array(y_test[11:])

    print('y_pred =', y_pred)
    print('y_test =', y_test)

    (stageLabels, sensitivity, specificity, accuracy, precision, f1score) = y2sensitivity(y_test, y_pred)
    (stageLabels4confusionMat, confusionMat) = y2confusionMat(y_test, y_pred, params.stageLabels4evaluation)
    printConfusionMat(stageLabels4confusionMat, confusionMat)
    print('sensitivity =', sensitivity)
    print('specificity =', specificity)
    print('accuracy =', accuracy)
    print('precision =', precision)
    print('###############################################################################################################################################################################################')

