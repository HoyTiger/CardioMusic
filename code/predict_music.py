def mean_stdev(xlist):
    return torch.std_mean(xlist.reshape(-1), unbiased=True)


import torch

from dataset import *
from models import *

if __name__ == '__main__':

    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    epochs = 200

    music_dataset = DEAM()
    ecg_dataset = DREAMER()

    train_dataset = MyDataset(ecg_dataset.Xtrain, ecg_dataset.ytrain, music_dataset.Xtrain, music_dataset.ytrain)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = MyDataset(ecg_dataset.Xtest, ecg_dataset.ytest, music_dataset.Xtest, music_dataset.ytest)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = DMLNet(pretrain=False).to(device)
    model.load_state_dict(torch.load('freeze_model/model_no_s.pth', map_location=device))

    # model_ecg = resnet1d50(num_classes=2, input_channels=1, inplanes=128, use_top=True).to(device)
    # model_ecg.load_state_dict(torch.load('ecg_128.pth', map_location=device))
    # model_music = resnet1d50(num_classes=2, input_channels=88, inplanes=128, use_top=True).to(device)
    # model_music.load_state_dict(torch.load('music_128.pth', map_location=device))

    with torch.no_grad():
        model.eval()
        music_outputs = None
        music_labels = None
        music_feature = model.music_model(torch.from_numpy(music_dataset.Xtest).to(device))
        music_out = model.music_reg(music_feature)

        if music_outputs is None:
            music_outputs = music_out
            music_labels = torch.from_numpy(music_dataset.ytest).to(device)
        else:
            music_outputs = torch.cat([music_outputs, music_out], dim=0)
            music_labels = torch.cat([music_labels, torch.from_numpy(music_dataset.ytest).to(device)], dim=0)

        loss3 = torch.nn.MSELoss()(music_outputs, music_labels)

        var_music = mean_stdev((music_outputs - music_labels).pow(2))

        print(loss3.item())
        print(var_music)
