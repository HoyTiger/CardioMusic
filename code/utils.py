import numpy as np

from models import *
import neurokit2 as nk

device = 'cuda:3'

music_dataset = DEAM()
ecg_dataset = DREAMER()

lead_index = list(range(4))
map(lambda x: str(x), lead_index)

# ecg = ecg_dataset.Xtest[0:10]
music = music_dataset.Xtest[[28,
                             75,
                             81,
                             177,
                             199,
                             217,
                             222,
                             236,
                             278,
                             288,
                             ]]


def get_index(ytest, v, a):
    l=[]
    for i, (x, y) in enumerate(ytest):
        if x == v and y == a:
            return i
            l.append(i)
    return l


ecg_index = [get_index(ecg_dataset.ytest, 0.25, 0), get_index(ecg_dataset.ytest, 0.25, 0.25),
             get_index(ecg_dataset.ytest, 0.75, 0.75), get_index(ecg_dataset.ytest, 0.75, 0.25)]

ecg_data = np.load('2_2.npy')
ecg_label =np.array([0.75,0.75])

music_label = music_dataset.ytest[[28,
                                   75,
                                   81,
                                   177,
                                   199,
                                   217,
                                   222,
                                   236,
                                   278,
                                   288,
                                   ]]
# ecg_data = ecg_dataset.Xtest[ecg_index]
# ecg_label = ecg_dataset.ytest[ecg_index]
model = DMLNet(pretrain=False).to(device)
model.load_state_dict(torch.load('model_final.p th', map_location=device))
ecg, label = ecg_data, ecg_label
np.save('2_2', ecg)
label, music_label = torch.Tensor(label.reshape(1, 2).repeat(10, axis=0)).to(device), torch.Tensor(music_label).to(
    device)
ecg = torch.Tensor(ecg.reshape((1, 1, 2560)).repeat(10, axis=0)).to(device)
music = torch.Tensor(music).to(device)
with torch.no_grad():
    model.eval()
    ecg_f = model.ecg_model(ecg)
    music_f = model.music_model(music)

    ecg_music_pair = torch.cat([ecg_f, music_f], dim=1)
    predict_distance = model.fusion(ecg_music_pair)

    truth = torch.exp(-torch.sum((label - music_label).pow(2), dim=1).sqrt() / 0.46829593).reshape((-1, 1))

    music_out = torch.cat([model.music_reg_v(music_f), model.music_reg_a(music_f)], dim=1)
    ecg_out = torch.cat([model.ecg_reg_v(ecg_f), model.ecg_reg_a(ecg_f)], dim=1)

    print(predict_distance.detach().cpu().numpy(), truth.detach().cpu().numpy())
    print(music_label,label)
    print(music_out.detach().cpu().numpy(), ecg_out.detach().cpu().numpy())
