import mne
import numpy as np
from datetime import datetime, timedelta
import os
# import matplotlib.pyplot as plt


class Loaddata:
# read stageSeq
    def readStageSeq(self, filePath):
        self.metaDataLineNumUpperBound4stage=60
        self.cueWhereStageDataStarts="Time"
        stage_fp = open(filePath, 'r', encoding='utf-8')
        for i in range(self.metaDataLineNumUpperBound4stage):    # skip lines that describes metadata
            line = stage_fp.readline()
            if line.startswith(self.cueWhereStageDataStarts):
                # stage_fp.readline()#jump ,,,,
                break
            if i == self.metaDataLineNumUpperBound4stage - 1:

                stage_fp.close()
                stage_fp = open(filePath, 'r', encoding='utf-8')

        stagesL = []
        
        for line in stage_fp:
            if line.startswith(','):#jump empty
                    continue
               
            line = line.rstrip()

            if ',' in line:
                elems = line.split(',')
            else:
                elems = line.split('\t')

            if len(elems) > 1:
                stageLabel = elems[2]
            else:
                stageLabel = elems[0]
            # durationWindNum = 1
            stageLabel = stageLabel.replace('*','').upper()
            if stageLabel == 'NR':
                stageLabel = 'S'
            elif stageLabel == '2':
                stageLabel = 'S'
            elif stageLabel == 'W':
                stageLabel = 'L'
            elif stageLabel == 'HH':
                stageLabel = 'H'
            stagesL.append(stageLabel)#.upper())
            durationWindNumsL = [1]*len(stagesL)

        return np.array(stagesL)
    def readEEG(self, filePath,emg_ind=False):
            self.metaDataLineNumUpperBound4eeg=50
            self.cueWhereEEGDataStarts='Time'
            self.samplingFreq_from_params=128
            eeg_fp = open(filePath, 'r',encoding= 'utf-8')
            for i in range(self.metaDataLineNumUpperBound4eeg):    # skip lines that describes metadata
                line = eeg_fp.readline()
                # print('line = ' + line)
                if line.startswith(self.cueWhereEEGDataStarts):
                    break
                if i == self.metaDataLineNumUpperBound4eeg - 1:
                    # print('eeg file without metadata header, but it\'s okay.')
                    eeg_fp.close()
                    eeg_fp = open(filePath, 'r', encoding='utf-8')

            #-----------
            # read text file
            print('---------------------')
            print('Started to read ' + filePath)# + '. It may take a few minutes before starting to classify. Please wait.')
            print('---------------------')
            timeStampsL = []
            eegL = []
            emgL = []
            timeStamp = datetime.now()
            for line in eeg_fp:
                line = line.rstrip()
                # print('line = ' + line)
                if ',' in line:
                    elems = line.split(',')
                elif '\t' in line:
                    elems = line.split('\t')
                else:
                    elems = line.split(' ')
                if len(elems) > 1:
                    ### timeStampsL.append(elems[0].split(' ')[2].split(':')[2])
                    # print('  elems[1] = ' + str(elems[1]) + ', elems[2] = ' + str(elems[2]))
                    if ' ' in elems[0]:
                        timeStampsL.append(elems[0].split(' ')[2])
                    else:
                        timeStampsL.append(elems[0])
                    eegL.append(float(elems[1]))
                    if len(elems) > 2:
                        try:
                            emgL.append(float(elems[2]))
                        except ValueError:
                            emgL.append(0)
                elif len(elems) == 1:
                    # when wave data contains no timestamp
                    timeStampsL.append(str(timeStamp).split(' ')[-1])
                    timeStamp += timedelta(seconds=1.0 / self.samplingFreq_from_params)
                    eegL.append(float(elems[0]))
                    # print(timeStampsL[-1], ':', eegL[-1])

            eeg = np.array(eegL)
            emg = np.array(emgL)
            timeStamps = np.array(timeStampsL)
            if emg_ind == True:
                return eeg, emg, timeStamps
            else:
                return eeg,timeStamps
            
    def up_or_down_sampling(self, signal_rawarray, model_samplePointNum, observed_samplePointNum):
        # downsampling
        # print('-------')
        # print('model_samplePointNum =', model_samplePointNum)
        # print('observed_samplePointNum =', observed_samplePointNum)
        if model_samplePointNum < observed_samplePointNum:
            print('-> downsampling')
            print('before downsampling: signal_rawarray.shape =', signal_rawarray.shape)
            epochNum = max(
                1, int(np.floor(1.0 * signal_rawarray.shape[0] / observed_samplePointNum)))
            print('epochNum =', epochNum)
            multiple = int(np.floor(
                1.0 * signal_rawarray.shape[0] / model_samplePointNum)) * model_samplePointNum * epochNum
            split_signal = np.array_split(
                signal_rawarray[:multiple], model_samplePointNum * epochNum)
            # for seg in split_signal:
            #     print('len(seg) =', len(seg))
            signal_rawarray = np.array([seg.mean() for seg in split_signal])
            # print('len(split_signal) =', len(split_signal))
            # print('split_signal[0].shape =', split_signal[0].shape)
            # print('after downsampling: signal_rawarray.shape =', signal_rawarray.shape)

        # upsampling
        if model_samplePointNum > observed_samplePointNum:
            upsample_rate = np.int(
                np.ceil(1.0 * model_samplePointNum / observed_samplePointNum))
            signal_rawarray = np.array(
                [[elem] * upsample_rate for elem in signal_rawarray]).flatten()[:model_samplePointNum]

        return signal_rawarray

    def ds_norm_mne(self, folder_path, prefx, freq):

            
        data = self.readEEG(folder_path+"/"+prefx+".csv")[0]
        label = self.readStageSeq(folder_path+"/"+prefx+"_Trend.csv")

        onset = np.arange(0, 4*len(label), 4)
        fourS = np.array(len(label)*[4])
        my_annot = mne.Annotations(onset, duration=fourS, description=label)
        # print(my_annot)
        print(data.shape)
        data = self.up_or_down_sampling(data, model_samplePointNum=128, observed_samplePointNum = freq)

        inf = mne.create_info(["eeg"], 128, 'eeg')
        data = np.array(data).reshape(1, -1)
        print(data.shape)


        # normalize
        data -= np.mean(data, axis=1, keepdims=True)
        data = data / np.std(data, axis=1, keepdims=True)
        raw = mne.io.RawArray(data, inf)
        raw.set_annotations(my_annot)
        return raw
    
    def to_mne_raw(self,folder_path,prefx=False,norm=False):
        if prefx:
            # prf = os.listdir(folder_path)[0].split(".")[0]
            data, t = self.readEEG(folder_path+"/"+prefx+".csv")              
            label = self.readStageSeq(folder_path+"/"+prefx+"_Trend.csv")
        else:
            data, t = self.readEEG(folder_path+"/"+"raw.csv")
            label = self.readStageSeq(folder_path+"/"+"tlabels.csv")
            
        onset = np.arange(0,4*len(label),4)      
        fourS = np.array(len(label)*[4])      
        my_annot = mne.Annotations(onset, duration=fourS, description=label)      
        # print(my_annot)      
        # print(data)      
        inf = mne.create_info(["eeg"],128,'eeg')      
        data = np.array(data).reshape(1,-1)
        #normalize
        if norm:    
          data -= np.mean(data, axis=1, keepdims=True)
          data = data / np.std(data, axis=1, keepdims=True)
        raw = mne.io.RawArray(data,inf)      
        raw.set_annotations(my_annot)
        return raw

    def twolabel_raw(self, folder_path, prf,ai_path,norm=False):
        
        data, t = self.readEEG(folder_path+"/"+prf+".csv")
        label1 = self.readStageSeq(folder_path+"/"+prf+"_Trend.csv")
        if os.path.exists(ai_path+"/"+prf+".txt"):
          label2 = self.readStageSeq(ai_path+"/"+prf+".txt")
        else:
          label2 = np.array(["?" for i in range(len(label1))])
          print("********************************* ")
          print("********************************* ")
          print("not exists: "+prf)
          print("********************************* ")
          print("********************************* ")
        label = np.char.add(label1, label2)      
        onset = np.arange(0, 4*len(label),4)      
        fourS = np.array(len(label)*[4])
        my_annot = mne.Annotations(onset, duration=fourS, description=label)
        inf = mne.create_info(["eeg"], 128,'eeg')      
        data = np.array(data).reshape(1, -1)
        #normalize
        if norm:    
          data -= np.mean(data, axis=1, keepdims=True)
          data = data / np.std(data, axis=1, keepdims=True)

        raw = mne.io.RawArray(data, inf)      
        raw.set_annotations(my_annot)
        return raw
        
