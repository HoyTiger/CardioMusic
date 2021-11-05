import torch

from dataset import *
from model.resnet1d import *


epochs = 100
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
lr = 1e-2

ecg_dataset = DEAM()

train_dataset = DEAMDataset(ecg_dataset.Xtrain, ecg_dataset.ytrain)
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = DEAMDataset(ecg_dataset.Xtest, ecg_dataset.ytest)
test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

model = resnet1d50(num_classes=2, input_channels=88, inplanes=128, use_top=True).to(device)

# model_dict = model.state_dict()
# pretrain = torch.load('music_128.pth')
# pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

opt = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

cost = torch.nn.MSELoss()
es = 0
min_loss = 1
print(len(train_dataset))
for epoch in range(0, epochs):
    model.train()
    for i, (ecg_data, ecg_label,) in enumerate(train_data_loader):
        ecg_data, ecg_label = ecg_data.to(device), ecg_label.to(device),
        opt.zero_grad()
        output = model(ecg_data)
        loss = cost(output, ecg_label)
        loss.backward()
        opt.step()

        if i % 10 == 0:
            print(f"epoch: {epoch}, batch: {i}, loss:{loss}")

    with torch.no_grad():
        model.eval()
        test_label = None
        test_output = None
        loss_total = 0
        for i, (ecg_data, ecg_label) in enumerate(test_data_loader):
            ecg_data, ecg_label = ecg_data.to(device), ecg_label.to(device)
            output = model(ecg_data)
            loss = cost(output, ecg_label)
            loss_total += loss.item()

            # print(ecg_label, output)

            if test_output is None and test_label is None:
                test_output = output
                test_label = ecg_label
            else:
                test_output = torch.cat([test_output, output], dim=0)
                test_label = torch.cat([test_label, ecg_label], dim=0)

        loss = cost(test_output, test_label)
        scheduler.step(loss)
        print(f"Test\tepoch: {epoch}, loss:{loss_total / len(test_data_loader)}")

        if loss < min_loss:
            torch.save(model.state_dict(), 'music_128.pth')
            min_loss = loss
            es = 0
        else:
            es += 1
        if es > 20:
            break

