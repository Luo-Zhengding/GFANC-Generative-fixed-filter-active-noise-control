import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import progressbar

#------------------------------------------------------------------------------
# Class: FxLMS algorithm
#------------------------------------------------------------------------------
class FxLMS():
    
    def __init__(self, Len):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float) # initial coefficients of filter
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
    
    def feedforward(self, Xf): # fixed reference signal passes the control filter to output the control signal
        self.Xd = torch.roll(self.Xd, shifts=1, dims=1) # roll the tensor along the given dimension
        self.Xd[0,0] = Xf # Xf: reference signal
        yt = self.Wc @ self.Xd.t() #矩阵相乘
        return yt
    
    def LossFunction(self, y, d):
        e = d-y
        return e**2, e
    
    def _get_coeff_(self):
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# Function: train_fxlms_algorithm()
#------------------------------------------------------------------------------
def train_fxlms_algorithm(Model, Ref, Disturbance, Stepsize=0.00000005): 

    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
        widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    optimizer = optim.SGD([Model.Wc], lr=Stepsize) # Stepsize is learning_rate
    
    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    for itera in range(len_data):
        # Feedfoward
        xin = Ref[itera]
        dis = Disturbance[itera]
        y = Model.feedforward(xin)
        loss,e = Model.LossFunction(y, dis)
        
        # Progress shown
        bar.update(2*itera+1)
            
        # Backward
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())
        
        # Progress shown 
        bar.update(2*itera+2)
    bar.finish()
    return Erro_signal

#------------------------------------------------------------
# Function: Generating the testing broadband noise
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)