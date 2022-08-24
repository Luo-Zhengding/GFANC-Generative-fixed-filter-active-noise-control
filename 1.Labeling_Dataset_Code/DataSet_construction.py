import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from scipy.fft import fft, fftfreq, ifft
str(torchaudio.get_audio_backend())
import os, sys
import math
import pandas as pd


def BandlimitedNoise_generation(f_star, Bandwidth, fs, N):
    # f_star indecats the start of frequency band (Hz)
    # Bandwith denots the bandwith of the boradabnd noise 
    # fs denots the sample frequecy (Hz)
    # N represents the number of point
    len_f = 1024
    f_end = f_star + Bandwidth
    b2 = signal.firwin(len_f, [f_star, f_end], pass_zero='bandpass', window ='hamming', fs=fs) # FIR bandpass filter
    xin = np.random.randn(N) # random signal fs+1023
    Re = signal.lfilter(b2,1,xin) # random sigal pass the bandpass filter
    Noise = Re[len_f-1:] # Re[1023:fs+1023] the obtained length is 16000
    #----------------------------------------------------
    return Noise/np.sqrt(np.var(Noise)) #除以标准差


class DatasetSheet:
    def __init__(self, folder, filename):
        self.filename = filename 
        self.folder = folder
        try: 
            os.mkdir(folder, 755)
        except:
            print("folder exists")
        self.path = os.path.join(folder, filename)
    def add_data_to_file(self, wave_file, class_ID):
        dict = {'File_path': [wave_file], 'Class_ID': [class_ID]}
        df = pd.DataFrame(dict)
        
        with open(self.path, mode = 'a') as f:
            df.to_csv(f, header=f.tell()==0)
    def flush(self):
        dc = pd.read_csv(self.path, index_col=0)
        dc.index = range(len(dc))
        dc.to_csv(self.path)


class SoundGenerator:
    def __init__(self, fs, folder):
        self.fs = fs
        self.len = fs + 1023
        self.folder = folder
        self.Num = 0
        try: 
            os.makedirs(folder) #创建多级目录
        except:
            print("folder exists")
    
    def _construct_(self):
        self.Num = self.Num + 1 #第几个声音
        f_star = np.random.uniform(20, 7980, 1) # 随机采样得到一个开始频率
        bandWidth = np.random.uniform(1,7980-f_star,1) # 随机采样得到一个带宽
        f_end = f_star + bandWidth # 结束频率=开始频率+带宽
        filename = f'{self.Num}_Frequency_from_'+ f'{f_star[0]:.0f}_to_{f_end[0]:.0f}_Hz.wav'
        filePath = os.path.join(self.folder, filename)
        noise = BandlimitedNoise_generation(f_star[0], bandWidth[0], self.fs, self.len) # f_star is a list while f_star[0] is a value
        noise = torch.from_numpy(noise).type(torch.float32).unsqueeze(0) # generate a random noise with the freqency limits
        torchaudio.save(filePath, noise, self.fs)
        return f_star[0], f_end[0], filename


#--------------------------------------------------------------------------------------
# Function: Generating Dataset
#--------------------------------------------------------------------------------------
def Generating_Synthetic_NoiseDataset(N_sample, Folder_name):
    import progressbar
    bar = progressbar.ProgressBar(maxval=N_sample, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    Generator = SoundGenerator(fs=16000, folder=Folder_name)
    
    bar.start()
    for ii in range(N_sample):
        f_star, f_end, filePath = Generator._construct_() # N_sample个随机产生的声音
        bar.update(ii+1)
    bar.finish()