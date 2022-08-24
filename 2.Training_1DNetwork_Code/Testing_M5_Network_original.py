import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from MyDataLoader import MyNoiseDataset, MyNoiseDataset1
from M5_Network import m3, m5, m11, m18, m34_res, m6_res

BATCH_SIZE = 250

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader


def load_weigth_for_model(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location="cuda:0")
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)


def validate_single_epoch(model, eva_data_loader, device):
    eval_loss = 0
    eval_acc = 0
    model.eval()

    for input, target in eva_data_loader:
        input, target = input.to(device), target.to(device)
        
        # Calculating the accuracy
        prediction = model(input)
        num_correct = sum(row.all().int().item() for row in (prediction.ge(0.5) == target))# !!! Threshold
        acc = num_correct / input.shape[0]
        eval_acc += acc

    return eval_acc/len(eva_data_loader)


def Test_model_accuracy_original(TESTING_DATASET_FILE, MODLE_PTH, File_sheet):
    testing_dataset = MyNoiseDataset(TESTING_DATASET_FILE, File_sheet)
    testing_loader = create_data_loader(testing_dataset, int(BATCH_SIZE/10))
    
    # set the model
    model = m6_res
    device = torch.device('cuda')
    model = model.to(device)
    # loading coefficients 
    load_weigth_for_model(model, MODLE_PTH)
    model.eval()
    
    # testing model
    acc_validate = validate_single_epoch(model, testing_loader, device)
    
    return acc_validate

def Output_Test_Error_Samples(TESTING_DATASET_FILE, MODLE_PTH, File_sheet):
    testing_dataset = MyNoiseDataset1(TESTING_DATASET_FILE, File_sheet)
    
    # set the model
    model = m6_res
    device = torch.device('cuda')
    model = model.to(device)
    # loading coefficients 
    load_weigth_for_model(model, MODLE_PTH)
    model.eval()
    
    j=0
    print('length of testing_dataset:%d'%len(testing_dataset))
    for i in range(len(testing_dataset)):
        audio_sample_path, signal, label = testing_dataset[i]
        signal = signal.to(device)
        signal = signal.unsqueeze(0)
        prediction = model(signal)
        pre = prediction.ge(0.5) # !!! threshold
        pre = pre.detach().cpu().numpy()
        pre = pre.astype(int)
        pre = pre.tolist()
        label = eval(label) # str: '[1, 0, 1]' to list: [1, 0, 1]
        if pre == label:
            j += 1
        else:
            print(audio_sample_path, pre, label)
    accuracy = j/len(testing_dataset) # test_accuracy
    return accuracy