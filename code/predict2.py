from typing import List

import torch

from models import *

def mean_stdev(xlist):
    return torch.std_mean(xlist.reshape(-1), unbiased=True)

if __name__ == '__main__':

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    epochs = 200

    music_dataset = DEAM()
    ecg_dataset = DREAMER()

    train_dataset = MyDataset(ecg_dataset.Xtrain, ecg_dataset.ytrain, music_dataset.Xtrain, music_dataset.ytrain)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = MyDataset(ecg_dataset.Xtest, ecg_dataset.ytest, music_dataset.Xtest, music_dataset.ytest)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model_ecg = resnet1d50(num_classes=2, input_channels=1, inplanes=128, use_top=True).to(device)
    model_ecg.load_state_dict(torch.load('freeze_model/ecg_128.pth', map_location=device))
    model_music = resnet1d50(num_classes=2, input_channels=88, inplanes=128, use_top=True).to(device)
    model_music.load_state_dict(torch.load('freeze_model/music_128.pth', map_location=device))

    with torch.no_grad():
        model_ecg.eval()
        model_music.eval()
        music_outputs = None
        ecg_outputs = None
        sim_label = None
        music_labels = None
        ecg_labels = None
        loss_total = 0
        for i, (ecg, ecg_label, music, music_label, distance) in enumerate(test_data_loader):
            ecg, ecg_label, music, music_label, distance = ecg.to(device), ecg_label.to(device), music.to(
                device), music_label.to(device), distance.to(device)

            ecg_out = model_ecg(ecg)
            music_out = model_music(music)

            if music_outputs is None :
                ecg_outputs = ecg_out
                music_outputs = music_out
                ecg_labels = ecg_label
                music_labels = music_label
                sim_label = distance
            else:
                music_outputs = torch.cat([music_outputs, music_out], dim=0)
                ecg_outputs = torch.cat([ecg_outputs, ecg_out], dim=0)
                ecg_labels = torch.cat([ecg_labels, ecg_label], dim=0)
                music_labels = torch.cat([music_labels, music_label], dim=0)
                sim_label = torch.cat([sim_label, distance], dim=0)

        predeict = torch.exp(-torch.sum((music_outputs - ecg_outputs).pow(2), dim=1).sqrt() / test_dataset.mean).reshape((-1,1))
        loss = torch.nn.MSELoss()(predeict, sim_label)
        loss2 = torch.nn.MSELoss()(ecg_outputs, ecg_labels)
        loss3 = torch.nn.MSELoss()(music_outputs, music_labels)

        var_sim_m = mean_stdev((predeict - sim_label).pow(2))

        var_ecg = mean_stdev((ecg_outputs - ecg_labels).pow(2))
        var_music = mean_stdev((music_outputs - music_labels).pow(2))
        print(loss.item(), loss2.item(), loss3.item())
        print(var_sim_m, var_ecg, var_music)








