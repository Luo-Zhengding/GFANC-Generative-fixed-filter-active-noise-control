import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.io as sio

from Disturbance_generation import Disturbance_reference_generation_from_Afilter
from FxLMS_algorithm import FxLMS, train_fxlms_algorithm
from DFT_Filter_Decompose import Creating_Filter, Filter_Decompose


def save_mat__(FILE_NAME_PATH, Wc):
    mdict = {'Wc_v': Wc}
    savemat(FILE_NAME_PATH, mdict)


#----------------------------------------------------
# Function loading_paths_from_MAT（）
#----------------------------------------------------
def loading_paths_from_MAT(folder, subfolder, Pri_path_file_name, Sec_path_file_name):
    Primay_path_file, Secondary_path_file = os.path.join(folder, subfolder, Pri_path_file_name), os.path.join(folder,subfolder, Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path

def main():
    FILE_NAME_PATH = 'models/Pretrained_Sub_Control_filters.mat'
    fs = 16000
    control_filter = Creating_Filter(Len=1024, low_cut_normal_fre=20, high_cut_normal_fre=7980, fs=fs)

    # Configurating the pre-trained control filter parameters
    T = 30
    Len_control = 1024
    Pri_path, Secon_path = loading_paths_from_MAT(folder='Pz and Sz', subfolder='Dongyuan', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')

    # Get the filtered-x and disturbance, train the main control filter
    Dis, Fx = Disturbance_reference_generation_from_Afilter(fs=fs, T=T, f_vector=control_filter, Pri_path=Pri_path, Sec_path=Secon_path)
    controller = FxLMS(Len=Len_control)
    Erro = train_fxlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, Stepsize=0.0001) # !!! train step size
    Wc_main = np.squeeze(controller._get_coeff_())
    save_mat__('models/Pretrained_Main_Control_filter.mat', Wc_main)

    # Drawing the noise reduction error of the main control filter
    plt.title('The error signal of main control filter')
    plt.plot(Erro)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.grid()
    plt.show()

    # Devide the pretrained main control filter into 15 sub filters
    sub_filters = Filter_Decompose(Filter=Wc_main, Num_subfilters=15, fs=16000)
    save_mat__(FILE_NAME_PATH, sub_filters)
    
if __name__ == "__main__":
    main()