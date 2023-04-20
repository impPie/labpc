# import sys
# sys.path.insert(1,'..')
import numpy as np
from scipy import signal
from _7utils.parameterSetup import ParameterSetup
from _4extract.featureExtractor import FeatureExtractor


class FeatureExtractorRawDataWithSTFT(FeatureExtractor):

    def __init__(self):
        params = ParameterSetup()
        self.samplingFreq = params.samplingFreq
        self.stft_time_bin_in_seconds = params.stft_time_bin_in_seconds
        self.stft_nperseg = np.int(np.floor(256.0 * self.stft_time_bin_in_seconds))
        self.extractorType = 'rawDataWithSTFT'
        self.lightPeriodStartTime = params.lightPeriodStartTime
        self.wholeBand = params.wholeBand
        self.binNum4spectrum = round(self.wholeBand.getBandWidth() / params.binWidth4freqHisto)
        # self.binArray4spectrum = np.linspace(self.wholeBand.bottom, self.wholeBand.top, self.binNum4spectrum + 1)

    def filtering(self, Zxx, freqs, lowerFreq, upperFreq):
        zipped = list(filter(lambda x: lowerFreq <= x[1] and x[1] < upperFreq, zip(Zxx, freqs)))
        return np.array([e[0] for e in zipped]), np.array([e[1] for e in zipped])

    def binning(self, Zxx, freqs, freqBinNum):
        binSize = np.int(np.floor(1.0 * len(Zxx) / freqBinNum))
        Zxx_binned = np.array([np.sum(np.abs(Zxx[(binID*binSize):((binID+1)*binSize)]),axis=0) for binID in range(freqBinNum)])
        freqs_binned = np.array([np.mean(freqs[(binID*binSize):((binID+1)*binSize)],axis=0) for binID in range(freqBinNum)])
        return Zxx_binned, freqs_binned

    def getFeatures(self, eegSegment, timeStampSegment, time_step):
        # compute power spectrum and sort it
        
        freqs, segment_times, Zxx = signal.stft(eegSegment, fs=self.samplingFreq, nperseg=self.stft_nperseg)
        
        Zxx_filtered, freqs_filtered = self.filtering(Zxx, freqs, self.wholeBand.bottom, self.wholeBand.top)

        Zxx_binned, freqs_binned = self.binning(Zxx_filtered, freqs_filtered, self.binNum4spectrum)
        Zxx_flattened = Zxx_binned.reshape(-1)
        #----------------
        # add time after light period started as a features
        rawDataWithSTFT = np.r_[eegSegment, Zxx_flattened]

        return rawDataWithSTFT
