from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from models import *
from dataset import *
import torch


if __name__ == '__main__':

    # default 'log_dir' is
    writer = SummaryWriter("runs/{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()))

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    lr = 1e-3
    epochs = 200

    music_dataset = DEAM()
    ecg_dataset = DREAMER()

    train_dataset = MyDataset(ecg_dataset.Xtrain, ecg_dataset.ytrain, music_dataset.Xtrain, music_dataset.ytrain)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = MyDataset(ecg_dataset.Xtest, ecg_dataset.ytest, music_dataset.Xtest, music_dataset.ytest)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = DMLNet(pretrain=False).to(device)
    model.load_state_dict(torch.load('model6.pth'))

    opt = torch.optim.SGD(model.parameters(), lr=lr)
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
            ecg, ecg_label, music, music_label, distance = ecg.to(device), ecg_label.to(device), music.to(device), music_label.to(device), distance.to(device)
            opt.zero_grad()
            loss, loss1, loss2, loss3, _, _, _= model(ecg, ecg_label, music, music_label, distance)
            loss.backward()
            opt.step()

            if i % 10 == 0:
                print(
                    f"epoch: {epoch}, batch: {i}, loss:{loss}, ecg loss:{loss1}, music loss: {loss2}, loss smi:{loss3}"
                    f"")
                writer.add_scalar(
                    "ecg loss",
                    loss1,
                    epoch * len(train_data_loader) + i
                )

                # writer.add_scalar(
                #     "ecg loss a",
                #     loss2,
                #     epoch * len(train_data_loader) + i
                # )

                writer.add_scalar(
                    "music loss",
                    loss2,
                    epoch * len(train_data_loader) + i
                )

                # writer.add_scalar(
                #     "music loss a",
                #     loss4,
                #     epoch * len(train_data_loader) + i
                # )

                writer.add_scalar(
                    "similarity loss",
                    loss3,
                    epoch * len(train_data_loader) + i
                )
                # writer.add_scalar(
                #     "cfr loss",
                #     loss4,
                #     epoch * len(train_data_loader) + i
                # )
                # writer.add_scalar(
                #     "cfm loss",
                #     loss5,
                #     epoch * len(train_data_loader) + i
                # )

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

                loss, loss1, loss2, loss3, ecg_out, music_out, predict_distance = model(ecg,
                                                                                                      ecg_label,
                                                                                                      music,
                                                                                                      music_label,
                                                                                                      distance)
                writer.add_scalar(
                    "test loss",
                    loss,
                    epoch * len(test_data_loader) + i
                )
                writer.add_scalar(
                    "test ecg loss",
                    loss1,
                    epoch * len(test_data_loader) + i
                )
                # writer.add_scalar(
                #     "test ecg loss a",
                #     loss2,
                #     epoch * len(test_data_loader) + i
                # )
                writer.add_scalar(
                    "test music loss",
                    loss2,
                    epoch * len(test_data_loader) + i
                )
                # writer.add_scalar(
                #     "test music loss a",
                #     loss4,
                #     epoch * len(test_data_loader) + i
                # )

                writer.add_scalar(
                    "test similarity loss",
                    loss3,
                    epoch * len(test_data_loader) + i
                )

                if music_outputs is None:
                    ecg_outputs = ecg_out
                    music_outputs = music_out
                    ecg_labels = ecg_label
                    music_labels = music_label
                    sim_label = distance
                    sim_outputs = predict_distance
                else:
                    music_outputs = torch.cat([music_outputs, music_out], dim=0)
                    ecg_outputs = torch.cat([ecg_outputs, ecg_out], dim=0)
                    ecg_labels = torch.cat([ecg_labels, ecg_label], dim=0)
                    music_labels = torch.cat([music_labels, music_label], dim=0)
                    sim_label = torch.cat([sim_label, distance], dim=0)
                    sim_outputs = torch.cat([sim_outputs, predict_distance], dim=0)

            predeict = torch.exp(-torch.sum((music_outputs - ecg_outputs).pow(2), dim=1).sqrt() / test_dataset.mean).reshape((-1, 1))
            loss = torch.nn.MSELoss()(sim_outputs, sim_label)
            loss1 = torch.nn.MSELoss()(predeict, sim_label)
            loss2 = torch.nn.MSELoss()(ecg_outputs, ecg_labels)
            loss3 = torch.nn.MSELoss()(music_outputs, music_labels)
            scheduler.step(loss + loss2 + loss3)

            print(f"Test\tepoch: {epoch}, loss sim:{loss}, loss me:{loss1}, ecg loss: {loss2}, music loss:{loss3}")

            # print(loss.item(), loss1.item(), loss2.item(), loss3.item())

                # writer.add_scalar(
                #     "test cfr loss",
                #     loss4,
                #     epoch * len(test_data_loader) + i
                # )
                # writer.add_scalar(
                #     "test cfm loss",
                #     loss5,
                #     epoch * len(test_data_loader) + i
                # )

                #     if test_output is None and test_label is None:
                #         test_output = output
                #         test_label = label
                #     else:
                #         test_output = torch.cat([test_output, output], dim=0)
                #         test_label = torch.cat([test_label, label], dim=0)
                #
                # loss = cost(test_output, test_label)
                # acc = (test_output.argmax(1) == test_label).float().mean()
                # print(classification_report(test_label.cpu(), test_output.cpu().argmax(1)))

            if loss + loss2 + loss3 < best_loss:
                best_loss = loss + loss2 + loss3
                torch.save(model.state_dict(), 'model5.pth')
            else:
                es += 1
            if es > 20:
                break
