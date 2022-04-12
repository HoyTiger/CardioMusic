from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from dataset import *
from models import *

if __name__ == '__main__':

    # default 'log_dir' is
    writer = SummaryWriter("runs/{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()))

    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    epochs = 200

    music_dataset = DEAM()
    ecg_dataset = DREAMER()

    train_dataset = MyDataset(ecg_dataset.Xtrain, ecg_dataset.ytrain, music_dataset.data, music_dataset.label)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = MyDataset(ecg_dataset.Xtest, ecg_dataset.ytest, music_dataset.data, music_dataset.label)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = L3Net(pretrain=False).to(device)
    # model.load_state_dict(torch.load('model_pretrain.pth', map_location=device))
    # print(model)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True,
                                                           threshold=0.005, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)

    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    cost = torch.nn.CrossEntropyLoss()
    best_loss = 1.0
    es = 0
    for epoch in range(0, epochs):
        model.train()
        for i, (ecg, ecg_label, music, music_label, distance) in enumerate(train_data_loader):
            ecg, ecg_label, music, music_label, distance = ecg.to(device), ecg_label.to(device), music.to(
                device), music_label.to(device), distance.to(device)
            opt.zero_grad()
            loss, predict_distance = model(ecg, ecg_label, music, music_label, distance)
            loss.backward()
            opt.step()

            if i % 10 == 0:
                print(f"epoch: {epoch}, batch: {i}, loss:{loss}")

        with torch.no_grad():
            model.eval()
            music_outputs = None
            ecg_outputs = None
            sim_outputs = None
            sim_label = None
            music_labels = None
            ecg_labels = None
            loss_total = 0
            for i, (ecg, ecg_label, music, music_label, distance) in enumerate(test_data_loader):
                ecg, ecg_label, music, music_label, distance = ecg.to(device), ecg_label.to(device), music.to(
                    device), music_label.to(device), distance.to(device)

                loss, predict_distance = model(ecg,
                                               ecg_label,
                                               music,
                                               music_label,
                                               distance)

                if music_outputs is None:
                    sim_outputs = predict_distance
                    sim_label = distance
                else:
                    sim_label = torch.cat([sim_label, distance], dim=0)
                    sim_outputs = torch.cat([sim_outputs, predict_distance], dim=0)

            loss = torch.nn.MSELoss()(sim_outputs, sim_label)
            scheduler.step(loss)

            print(f"Test\tepoch: {epoch}, loss smi:{loss}, ")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'freeze_model/model_final_L3NET.pth')
                es = es // 2
            else:
                es += 1
            if es > 10:
                break
