import torch.optim
import torch.optim

from model.resnet1d import *


class DMLNet(nn.Module):

    def __init__(self, num_class=4, inplanes=128, use_top=False, pretrain=False, **kwargs):
        super(DMLNet, self).__init__()

        self.ecg_model = nn.Sequential(
            resnet1d50(num_classes=num_class, input_channels=1, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
        )
        self.music_model = nn.Sequential(
            nn.Conv1d(in_channels=88, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            AdaptiveConcatPool1d(),
            Flatten(),
        )
        if pretrain:
            self.load_pretrain_model(self.ecg_model[0], 'freeze_model/ecg_128.pth')
            # self.load_pretrain_model(self.music_model[0], 'freeze_model/music_128.pth')

            for p in self.ecg_model[0].parameters():
                p.requires_grad = False

        self.ecg_reg = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, inplanes * 2),
            nn.BatchNorm1d(inplanes * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 2, 2),
            nn.Sigmoid()
        )

        self.ecg_reg_a = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            nn.Sigmoid()
        )

        self.ecg_reg_v = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            nn.Sigmoid()
        )
        self.music_reg = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, inplanes * 2),
            nn.BatchNorm1d(inplanes * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 2, 2),
            nn.Sigmoid()
        )

        self.music_reg_a = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            nn.Sigmoid()
        )

        self.music_reg_v = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            # Flatten(),
            # nn.BatchNorm1d(inplanes * 4 * 2),
            nn.Linear(inplanes * 8 * 2, inplanes * 4 * 2),
            nn.BatchNorm1d(inplanes * 4 * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 4 * 2, inplanes * 2 * 2),
            nn.BatchNorm1d(inplanes * 2 * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 2 * 2, 1),
            nn.Sigmoid()
        )

        self.loss_fuc = nn.MSELoss()

    def load_pretrain_model(self, model, path):
        model_dict = model.state_dict()
        pretrain = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, ecg, ecg_label, music, music_label, distance, test=False):
        ecg_feature = self.ecg_model(ecg)
        music_feature = self.music_model(music)

        ecg_music_pair = torch.cat([ecg_feature, music_feature], dim=1)
        predict_distance = self.fusion(ecg_music_pair)
        # ecg_out = torch.cat([self.ecg_reg_v(ecg_feature), self.ecg_reg_a(ecg_feature)], dim=1)
        # music_out = torch.cat([self.music_reg_v(music_feature), self.music_reg_a(music_feature)], dim=1)
        ecg_out = self.ecg_reg(ecg_feature)
        music_out = self.music_reg(music_feature)
        loss = self.loss_fuc(ecg_out, ecg_label) + self.loss_fuc(music_out, music_label) + \
               self.loss_fuc(distance, predict_distance)
        # loss = self.loss_fuc(distance, predict_distance)

        if test:
            print(ecg_out.detach().cpu().numpy())
            print(music_out.detach().cpu().numpy())
            print(predict_distance.detach().cpu().numpy())

        return loss, self.loss_fuc(ecg_out, ecg_label), self.loss_fuc(music_out, music_label), self.loss_fuc(distance,
                                                                                                             predict_distance), ecg_out, music_out, predict_distance


class DMLNet2(nn.Module):

    def __init__(self, num_class=4, inplanes=128, use_top=False, pretrain=True, **kwargs):
        super(DMLNet2, self).__init__()

        self.ecg_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            AdaptiveConcatPool1d(),
            Flatten(),

        )
        self.music_model = nn.Sequential(
            nn.Conv1d(in_channels=88, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            AdaptiveConcatPool1d(),
            Flatten(),
        )
        if pretrain:
            self.load_pretrain_model(self.ecg_model[0], 'freeze_model/ecg_128.pth')
            self.load_pretrain_model(self.music_model[0], 'freeze_model/music_128.pth')

            for p in self.ecg_model[0].parameters():
                p.requires_grad = False

            for p in self.music_model[0].parameters():
                p.requires_grad = False

        self.ecg_reg = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, inplanes * 2),
            nn.BatchNorm1d(inplanes * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 2, 2),
            nn.Sigmoid()
        )

        self.music_reg = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, inplanes * 2),
            nn.BatchNorm1d(inplanes * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 2, 2),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            # Flatten(),
            # nn.BatchNorm1d(inplanes * 4 * 2),
            nn.Linear(inplanes * 8 * 2, inplanes * 4 * 2),
            nn.BatchNorm1d(inplanes * 4 * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 4 * 2, inplanes * 2 * 2),
            nn.BatchNorm1d(inplanes * 2 * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 2 * 2, 1),
            nn.Sigmoid()
        )

        self.loss_fuc = nn.MSELoss()

    def load_pretrain_model(self, model, path):
        model_dict = model.state_dict()
        pretrain = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, ecg, ecg_label, music, music_label, distance, test=False):
        ecg_feature = self.ecg_model(ecg)
        music_feature = self.music_model(music)

        ecg_music_pair = torch.cat([ecg_feature, music_feature], dim=1)
        predict_distance = self.fusion(ecg_music_pair)
        # ecg_out = torch.cat([self.ecg_reg_v(ecg_feature), self.ecg_reg_a(ecg_feature)], dim=1)
        # music_out = torch.cat([self.music_reg_v(music_feature), self.music_reg_a(music_feature)], dim=1)
        ecg_out = self.ecg_reg(ecg_feature)
        music_out = self.music_reg(music_feature)
        loss = self.loss_fuc(ecg_out, ecg_label) + self.loss_fuc(music_out, music_label) + \
               self.loss_fuc(distance, predict_distance)
        # loss = self.loss_fuc(distance, predict_distance)

        if test:
            print(ecg_out.detach().cpu().numpy())
            print(music_out.detach().cpu().numpy())
            print(predict_distance.detach().cpu().numpy())

        return loss, self.loss_fuc(ecg_out, ecg_label), self.loss_fuc(music_out, music_label), self.loss_fuc(distance,
                                                                                                             predict_distance), ecg_out, music_out, predict_distance


class L3Net(nn.Module):
    def __init__(self, inplanes=128, use_top=False, pretrain=True):
        super(L3Net, self).__init__()

        self.ecg_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),

            Flatten(),
        )

        self.music_model = nn.Sequential(
            nn.Conv1d(in_channels=88, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),

            Flatten(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(inplanes * 8, inplanes),
            nn.ReLU(),
            nn.Linear(inplanes, 1),
            nn.Sigmoid()
        )

        self.loss_fuc = nn.MSELoss()

    def load_pretrain_model(self, model, path):
        model_dict = model.state_dict()
        pretrain = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, ecg, ecg_label, music, music_label, distance, test=False):
        ecg_feature = self.ecg_model(ecg)
        music_feature = self.music_model(music)

        ecg_music_pair = torch.cat([ecg_feature, music_feature], dim=1)
        predict_distance = self.fusion(ecg_music_pair)
        # ecg_out = torch.cat([self.ecg_reg_v(ecg_feature), self.ecg_reg_a(ecg_feature)], dim=1)
        # music_out = torch.cat([self.music_reg_v(music_feature), self.music_reg_a(music_feature)], dim=1)
        loss = self.loss_fuc(distance, predict_distance)
        # loss = self.loss_fuc(distance, predict_distance)

        return loss, predict_distance


class ACPNet(nn.Module):
    def __init__(self, inplanes=128, use_top=False, pretrain=True):
        super(ACPNet, self).__init__()

        self.ecg_model = nn.Sequential(
            resnet1d50(num_classes=2, input_channels=1, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
        )

        self.music_model = nn.Sequential(
            Flatten(),
            nn.Linear(88 * 260, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        if pretrain:
            self.load_pretrain_model(self.ecg_model[0], 'freeze_model/ecg_128.pth')

            for p in self.ecg_model[0].parameters():
                p.requires_grad = False

        self.loss_fuc = nn.MSELoss()

    def load_pretrain_model(self, model, path):
        model_dict = model.state_dict()
        pretrain = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, ecg, ecg_label, music, music_label, distance, test=False):
        ecg_feature = self.ecg_model(ecg)
        music_feature = self.music_model(music)

        ecg_music_pair = torch.cat([ecg_feature, music_feature], dim=1)
        predict_distance = self.fusion(ecg_music_pair)
        # ecg_out = torch.cat([self.ecg_reg_v(ecg_feature), self.ecg_reg_a(ecg_feature)], dim=1)
        # music_out = torch.cat([self.music_reg_v(music_feature), self.music_reg_a(music_feature)], dim=1)
        loss = self.loss_fuc(distance, predict_distance)
        # loss = self.loss_fuc(distance, predict_distance)

        return loss, predict_distance
