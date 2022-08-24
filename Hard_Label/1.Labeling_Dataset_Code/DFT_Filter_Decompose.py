from cmath import pi
from scipy.fft import fft, fftfreq, ifft
import numpy as np 
import scipy.signal as signal 
import matplotlib.pyplot as plt


#---------------------------------------------------------------------------------
# Description : Creating a boardband filter
#---------------------------------------------------------------------------------
def Creating_Filter(Len=1024, low_cut_normal_fre=20, high_cut_normal_fre=7980, fs=16000):
    """
    Args:
        Len (int, optional): The length of the control filter
        low_cut_normal_fre (float, optional): The low cut-off normalized frequency
        high_cut_normal_fre (float, optional): The high cut-off normalized frequency

    Returns:
        float: The coefficients of the designed filter.
    """
    b1 = signal.firwin(Len, [low_cut_normal_fre, high_cut_normal_fre], pass_zero='bandpass', window ='hamming',fs=fs)
    w1, h1 = signal.freqz(b1)
    
    plt.title('Frequnecy Response of Original Filter ')
    plt.plot(w1*fs/(2*pi), 20*np.log10(np.abs(h1)),'b')
    plt.ylabel('Amplitude Response (dB)')
    plt.xlabel('Frequency')
    plt.grid()
    plt.show()
    return b1 


#---------------------------------------------------------------------------------
# Function: Drawing frequency spectrum via fft
#---------------------------------------------------------------------------------
def Spectrum_FFT(filter_response, fs=160000):
    N = len(filter_response)
    T = 1/fs 
    Fre_response = fft(filter_response)
    xf = fftfreq(N,T)
    yf =np.abs(Fre_response)**2
    plt.plot(yf)
    plt.grid()
    plt.show()


#---------------------------------------------------------------------------------
# Function: Drawing filter reponse in frequency spectrum and time domain
#---------------------------------------------------------------------------------
def Group_Filters_Responses(sub_filters, fs=16000):
    N_filter = sub_filters.shape[0]
    filter_total = np.sum(sub_filters, axis=0)
    
    
    for i in range(N_filter):
        plt.subplot(N_filter+1, 2, 2*i+1)
        plt.plot(sub_filters[i,:])
        
        w1, h1 = signal.freqz(sub_filters[i,:])
        plt.subplot(N_filter+1, 2, 2*(i+1))
        plt.plot(w1*fs/(2*pi), np.abs(h1)**2,'b')
    
    plt.subplot(N_filter+1, 2, 2*(N_filter+1)-1)
    plt.plot(filter_total)
        
    w1, h1 = signal.freqz(filter_total)
    plt.subplot(N_filter+1, 2, 2*(N_filter+1))
    plt.plot(w1*fs/(2*pi), np.abs(h1)**2,'b')
    
    plt.tight_layout() #调整subplot子图间的间距
    plt.show()
    

#---------------------------------------------------------------------------------
# Function: Perfectly decompose a filter into subfilters
#---------------------------------------------------------------------------------
def Filter_Decompose(Filter, Num_subfilters, fs=16000):
    """
    Args:
        Filter (float64): The impulse resoponse of the control filter
        Num_subfilters (int): The number of the sub filters
        fs (int, optional): The system sampling rate
    Returns:
        float64: The control filter's group.
    """
    N = len(Filter)
    sub_filters = np.zeros((Num_subfilters, N))
    sub_num = N//(Num_subfilters*2) # ???
    Fre_filter = fft(Filter)
    
    for ii in range(Num_subfilters):
        Temper_spectrum = np.zeros_like(Fre_filter)
        start_index = ii*sub_num+1
        end_index = (ii+1)*sub_num+1
        start_n_index = -(ii+1)*sub_num 
        end_n_index = -start_index+1
        
        if ii != Num_subfilters-1:
            Temper_spectrum[start_index:end_index] = Fre_filter[start_index:end_index]
            if end_n_index == 0 :
                Temper_spectrum[start_n_index:] = Fre_filter[start_n_index:]
            else:
                Temper_spectrum[start_n_index:end_n_index] = Fre_filter[start_n_index:end_n_index]
            
        else:
            Temper_spectrum[start_index:end_n_index] = Fre_filter[start_index:end_n_index]
        sub_filters[ii,:] = ifft(Temper_spectrum).real # the coefficients of 15 subfilters in time domain
    return sub_filters