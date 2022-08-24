import os 
import scipy.io as sio
import matplotlib.pyplot as plt
import torchaudio
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy import signal
import numpy as np 
import math
import progressbar

from DataSet_construction import DatasetSheet
from Adaptive_control_filter_generator_batch import adaptive_control_filter_batch, train_adaptive_gain_batch
from Disturbance_generation import Disturbance_generation_from_real_noise

# Save hard labels


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader


def loading_paths_from_MAT(folder, subfolder, Pri_path_file_name, Sec_path_file_name):
    Primay_path_file, Secondary_path_file = os.path.join(folder, subfolder, Pri_path_file_name), os.path.join(folder,subfolder, Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path


# Get the file with desire sufix
def read_all_file_from_folder(folder_path, file_sufix):
    f_names = []
    for (dirpath, dirnames, filenames) in os.walk(folder_path):
        for file_name in filenames:
            root_ext = os.path.splitext(file_name)
            if root_ext[1] == file_sufix:
                f_names.append(file_name)
    return f_names


def Load_Pretrained_filters_to_tensor(MAT_FILE): # Loading the control filter from the mat file
    mat_contents = sio.loadmat(MAT_FILE)
    Wc_vectors = mat_contents['Wc_v']
    return torch.from_numpy(Wc_vectors).type(torch.float) # converse from numpy to tensor
    

#--------------------------------------------------------------------
# Function : Construt a novel filter from sub filters based on hard decision
#--------------------------------------------------------------------
def Construt_filter_from_labels(threshold, sub_filters, soft_labels):
    labels = soft_labels >= threshold
    
    hard_labels = np.expand_dims(labels, axis=0)
    novel_filter = np.matmul(hard_labels, sub_filters) # reconstructed filter

    pre_vector = np.expand_dims(soft_labels, axis=0)
    const_filter = np.matmul(pre_vector, sub_filters) # perfect filter

    return novel_filter, const_filter, hard_labels


def additional_noise(signal, snr_db):
    signal_power = signal.norm(p=2)
    length = signal.shape[1]
    additional_noise = np.random.randn(length)
    additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
    noise_power = additional_noise.norm(p=2)
    snr = math.exp(snr_db / 10)
    scale = snr * noise_power / signal_power
    noisy_signal = signal + additional_noise/scale
    return noisy_signal


class NoiseDataset(Dataset):

    def __init__(self, folder_path, sufix, Pri_path, Sec_path):
        self.folder = folder_path
        self.f_names = read_all_file_from_folder(folder_path=folder_path, file_sufix=sufix)
        self.Pri_path = Pri_path
        self.Sec_path = Sec_path
    
    def __len__(self):
        return len(self.f_names)

    def __getitem__(self, index):
        filePath = os.path.join(self.folder, self.f_names[index])
        signal,_ = torchaudio.load(filePath)
        Dis, Fx, _ = Disturbance_generation_from_real_noise(fs=16000, Repet=3, wave_form=signal, Pri_path=self.Pri_path, Sec_path=self.Sec_path)
        Fx = additional_noise(signal=Fx.unsqueeze(0), snr_db=30)[0,:] # !!! SNR
        return self.f_names[index], Fx, Dis # the name of noise, filtered-x, disturbance


# Label noise dataset using the 15 pre-trained sub filters
class Automatic_label():
    
    def __init__(self, sufix, folder_path, path_mat, Index_file, threshold, **kwargs):
        '''
        Parameters:
            param1 - folder_path: the dirctory of the dataset 
            param2 - path_mat: the file of the pre-trained sub control filters
            param3 - Index_file_name: the output file name 
        '''
        fs = 16000
        self.BATCH_SIZE = 1000 # !!! batch_size
        self.fs = fs
        self.sufix = sufix
        self.folder = folder_path
        self.f_names = read_all_file_from_folder(folder_path=folder_path, file_sufix=sufix)
        self.sub_filters_T = Load_Pretrained_filters_to_tensor(path_mat)
        self.lable_index = DatasetSheet(folder=folder_path, filename=Index_file)
        self.Pri_path, self.Sec_path = loading_paths_from_MAT(folder='Pz and Sz', subfolder='Dongyuan', Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
        self.threshold = threshold

    def label(self):
        bar = progressbar.ProgressBar(maxval=len(self.f_names), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        
        print('=====================Start labeling ============================')
        print(f'The total file numer is {len(self.f_names)}')
        bar.start()
        
        train_data = NoiseDataset(self.folder, self.sufix, self.Pri_path, self.Sec_path)
        train_dataloader = create_data_loader(train_data, self.BATCH_SIZE)
        
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f'<<===This program used {device}====>>')

        for filenames, Fx, Dis in train_dataloader:
            Generator = adaptive_control_filter_batch(self.sub_filters_T, Batch_size=self.BATCH_SIZE, muw=0.001, device=device) # !!! step size
            error = train_adaptive_gain_batch(Generator, Fx, Dis, device=device)
            plt.plot(error)
            plt.title('The residual error of a batch')
            plt.grid()
            plt.show()
            
            wgains = Generator.get_coeffiecients_() # torch.Size([BATCH_SIZE, 15])
            for index, wgain in enumerate(wgains): # torch.Size([15])
                soft_labels = wgain.detach().cpu().numpy() # soft_labels
                # hard_labels
                novel_filter, const_filter, hard_labels = Construt_filter_from_labels(self.threshold, self.sub_filters_T.detach().numpy(), soft_labels)
                hard_labels = hard_labels.squeeze()
                hard_labels = hard_labels.astype(int)
                hard_labels = hard_labels.tolist()
                self.lable_index.add_data_to_file(wave_file=filenames[index], class_ID=hard_labels)
                bar.update(index)
            
        bar.finish()
        self.lable_index.flush()
        print('=======================End labeling ============================')