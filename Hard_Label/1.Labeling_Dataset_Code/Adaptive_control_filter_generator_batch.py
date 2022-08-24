import torch 
import progressbar
import numpy as np

#--------------------------------------------------------------------
# Class: adaptive_control_filter_batch_normal()
#--------------------------------------------------------------------
class adaptive_control_filter_batch():
    
    def __init__(self, Control_filter_groups, Batch_size, muw, device):
        """Adaptive algorithm is used to generate the control filter from the pre-trained filters batch processing.

        Args:
            Control_filter_groups (float32 tensor): The group of control filters [number of filter x length].
            Batch_size (int): The size of the batch 
            muw (float32): The step size value
        """
        self.filter_number = Control_filter_groups.shape[0]
        self.W_gain = torch.zeros(Batch_size, self.filter_number, requires_grad=False, dtype=torch.float, device=device) # [N x C]
        self.Y_outs = torch.zeros(Batch_size, self.filter_number, dtype=torch.float, device=device) # [N x C]
        self.Xd = torch.zeros(Batch_size, Control_filter_groups.shape[1], dtype=torch.float, device=device) # [N x Len_c]
        self.Filters = Control_filter_groups.to(device) # [C x Len_c]
        self.muw = muw
    
    def filter_processing(self, Xin, Dir):
        """Construting the anti-noise by combing the different output signals of the pre-trainned control filter.

        Args:
            Xin (float32 tensor): The dimension of the input vector is [Batch_size, 1].
            Dir (float32 tensor): The dimension of the disturbance is [Batch_size, 1]. 
        
        Returns:
            float32 tensor : [N x 1] the error signal vector 
        """
        
        # feedforward progress
        self.Xd = torch.roll(self.Xd,1,1)
        self.Xd[:,0] = Xin   
        self.Y_outs = self.Xd @ self.Filters.t() 
        y_anti_noise = torch.unsqueeze(torch.einsum('NC,NC->N',self.W_gain,self.Y_outs),dim=1) # [N x 1]
        Err_vec = torch.unsqueeze(Dir,dim=1) - y_anti_noise ; # [N x 1]
        Y_powers = torch.unsqueeze(torch.einsum('NC,NC->N',self.Y_outs,self.Y_outs),dim=1) # [N x 1]
        
        # Back progress
        self.W_gain += self.muw*(1/Y_powers)*self.Y_outs*Err_vec
        
        return Err_vec
    
    def get_coeffiecients_(self):
        """Extrating the coefficients from the generators. 

        Returns:
            float32: the coeffients of the adaptive gain vector.
        """
        return self.W_gain

#--------------------------------------------------------------------
# Class : train_adaptive_gain_aglrithm_based_batch()
#--------------------------------------------------------------------
def train_adaptive_gain_batch(model, filter_ref, Disturbance, device):
    """This code is used to train the gain of the adaptive filter algorithm.

    Args:
        model (class): It is the model constructed from the class of adaptive_control_filter_batch_normal().
        filter_ref (float32 vector): The filtered reference signal vector [Batch_size x Length of the data].
        Disturbance (float32 vector): The disturbance vector has size of [Batch_size x Length of the data]
        device = "cuda" or "cpu"

    Returns:
        The error signal of the adaptive gatin training result.
    """
    filter_ref = filter_ref.to(device)
    Disturbance = Disturbance.to(device)
    Len_data = Disturbance.shape[1]
    
    bar = progressbar.ProgressBar(maxval=Len_data, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    Erro_signal = []
    for itera in range(Len_data):
        xin = filter_ref[:,itera]
        dis = Disturbance[:,itera]
        erro_v = model.filter_processing(xin, dis) # torch.Size([Batch_size, 1])
        erro_v = erro_v.mean([0]) # torch.Size([1])
        Erro_signal.append(erro_v.cpu().detach().numpy())
        bar.update(itera+1)
        
    bar.finish()
    return Erro_signal