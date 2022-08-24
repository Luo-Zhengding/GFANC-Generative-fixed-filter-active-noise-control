import os 
from torch.utils.data import Dataset
import pandas as pd 
import torchaudio
import torch
import matplotlib
import matplotlib.pyplot as plt
import json
import numpy as np

#------------------------------------------------------------------------
# Class: minmaxscaler()
# Description: Shrink the data
#------------------------------------------------------------------------
def minmaxscaler(data):
    min = data.min()
    max = data.max()    
    return (data)/(max-min)

#------------------------------------------------------------------------
# Load the noise tracks and labels 
#------------------------------------------------------------------------
class MyNoiseDataset(Dataset):

    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
    
    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,_ = torchaudio.load(os.path.join(self.folder, audio_sample_path))
        signal = minmaxscaler(signal) # min-max normalization
        return signal, label
    
    def _get_audio_sample_path(self, index):
        path = self.annotations_file.iloc[index, 1]
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations_file.iloc[index,2]
        label = json.loads(label) # transform str to numpy float32
        label = np.array(label)
        label = label.astype(np.float32)
        return label
    
class MyNoiseDataset1(Dataset):

    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
    
    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal,_ = torchaudio.load(os.path.join(self.folder, audio_sample_path))
        signal = minmaxscaler(signal) # minmax normalization
        return audio_sample_path, signal, label # change
    
    def _get_audio_sample_path(self, index):
        path = self.annotations_file.iloc[index, 1]
        return path

    def _get_audio_sample_label(self, index):
        label = self.annotations_file.iloc[index,2]
        return label