import os
import subprocess
from typing import List

import numpy as np
import torch
import pandas as pd

from models import *

if __name__ == '__main__':
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        model = DMLNet(pretrain=False).to(device)
        model.load_state_dict(torch.load('model_final.pth', map_location=device))
        model.eval()
        music_dataset = DEAM()
        path = 'data/原始/'
        for file in os.listdir(path):
            if file.endswith('wav'):
                filename = path + file
                base = os.path.basename(file)
                command = 'SMILExtract -C IS13_ComParE_lld-func.conf -I ' + filename + " -O " + path + base + ".csv"
                subprocess.run(command, shell=True)
        all_data = []
        for file in natsorted(glob.glob(path + "*.csv")):
            if file.endswith('csv'):
                filename = file
                data = pd.read_csv(filename,  sep=';')
                df_features_noframe = np.array(data.iloc[:88, 1:].values)
                if len(df_features_noframe) != 88:
                    df_features_noframe = np.tile(df_features_noframe, (88 // len(df_features_noframe) + 1, 1))
                df_features_noframe = df_features_noframe[:88]
                print(df_features_noframe.shape)
                all_data.append(df_features_noframe)
                print(file)

        # X_scaler = StandardScaler()
        # X_list = X_scaler.fit_transform(np.vstack(np.array(music_dataset.Xtest)).flatten()[:, np.newaxis].astype(float))
        # X_list = X_list.reshape((-1, 88, 260))
        inputs = torch.from_numpy(np.array(music_dataset.Xtest).astype('float32')).to(device)
        feature = model.music_model(inputs)
        out = torch.cat([model.music_reg_v(feature), model.music_reg_a(feature)], dim=1)
        for i in out.detach().cpu().numpy():
            print(i[0], i[1])

        print('break')
        for i in music_dataset.ytest:
            print(i[0], i[1])

