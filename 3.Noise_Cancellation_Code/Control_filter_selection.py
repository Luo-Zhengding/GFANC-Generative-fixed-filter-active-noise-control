import torch
import os
import numpy as np
from torch import nn
import scipy.signal as signal
import scipy.io as sio

from M5_Network import m6_res


#-------------------------------------------------------------
# Function: load_weight_for_model()
# Loading pre-trained weights to model
#-------------------------------------------------------------
def load_weigth_for_model(model, pretrained_path, device):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)

    
def Load_Pretrained_filters_to_tensor(MAT_FILE):
    mat_contents = sio.loadmat(MAT_FILE)
    Wc_vectors = mat_contents['Wc_v']
    return torch.from_numpy(Wc_vectors).type(torch.float)

def minmaxscaler(data):
    min = data.min()
    max = data.max()    
    return (data)/(max-min)


def Construt_filter(threshold, sub_filters, soft_labels):
    labels = soft_labels >= threshold
    hard_labels = np.expand_dims(labels, axis=0)
    novel_filter = np.matmul(hard_labels, sub_filters) # reconstructed filter based on hard outputs
    return novel_filter, hard_labels


#-------------------------------------------------------------
# Function: multiple length of samples
#-------------------------------------------------------------
def Casting_multiple_time_length_of_primary_noise(primary_noise, fs):
    assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
    cast_len = primary_noise.shape[1] - primary_noise.shape[1]%fs
    return primary_noise[:,:cast_len] # make the length of primary_noise is an integer multiple of fs

def Casting_single_time_length_of_training_noise(filter_training_noise, fs):
    assert filter_training_noise.dim() == 3, 'The dimension of the training noise should be 3 !!!'
    print(filter_training_noise[:,:,:fs].shape)
    return filter_training_noise[:,:,:fs]


#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y = signal.lfilter(bandpass_filter,1,xin)
    yout = y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)


#-------------------------------------------------------------
# Class : Control_filter_Index_predictor
#-------------------------------------------------------------
class Control_filter_Index_predictor():
    
    def __init__(self, MODEL_PATH, path_mat, device, fs, threshold):
        model = m6_res
        load_weigth_for_model(model, MODEL_PATH, device)
        model = model.to(device)
        model.eval()
        
        self.device = device
        self.model = model
        self.fs = fs
        self.sub_filters_T = Load_Pretrained_filters_to_tensor(path_mat)
        self.threshold = threshold
    
    def predic_ID(self, noise): # predict the noise index
        noise = noise.to(self.device) # torch.Size([1, 16000])
        noise = noise.unsqueeze(0) # torch.Size([1, 1, 16000])
        noise = minmaxscaler(noise) # minmax normalization
        prediction = self.model(noise) # torch.Size([15])
        construt_filter, hard_labels = Construt_filter(self.threshold, self.sub_filters_T.detach().numpy(), soft_labels=prediction.detach().cpu().numpy())
        return construt_filter, hard_labels
    
    def predic_ID_vector(self, primary_noise):
        # Checking the length of the primary noise.
        assert  primary_noise.shape[0] == 1, 'The dimension of the primary noise should be [1 x samples] !!!'
        assert  primary_noise.shape[1] % self.fs == 0, 'The length of the primary noise is not an integral multiple of fs.'
        
        # Computing how many seconds the primary noise contain.
        Time_len = int(primary_noise.shape[1]/self.fs) 
        print(f'The primary nosie has {Time_len} seconds !!!')
        
        # Bulding the matric of the primary noise [times x 1 x fs]
        primary_noise_vectors = primary_noise.reshape(Time_len, self.fs).unsqueeze(1)
        
        # Get the control filter for each frame whose length is 1 second.
        Filter_vector = []
        for ii in range(Time_len):
            construt_filter, hard_labels = self.predic_ID(primary_noise_vectors[ii]) # construt_filter:(1, 1024)
            hard_labels = hard_labels.squeeze()
            hard_labels = hard_labels.astype(int)
            print(ii+1, hard_labels)
            construt_filter = construt_filter.squeeze()
            Filter_vector.append(construt_filter)
        Filter_vector = np.array(Filter_vector) # list to np.array
        return Filter_vector


def Control_filter_selection(fs, MODEL_PTH, path_mat, Primary_noise, threshold):
    device = torch.device('cuda')
    
    Pre_trained_control_filter_ID_pridector = Control_filter_Index_predictor(MODEL_PATH=MODEL_PTH, path_mat=path_mat, device=device, fs=fs, threshold=threshold)
    
    Primary_noise = Casting_multiple_time_length_of_primary_noise(Primary_noise, fs=fs)
    
    Filter_vector = Pre_trained_control_filter_ID_pridector.predic_ID_vector(Primary_noise)
    
    return Filter_vector