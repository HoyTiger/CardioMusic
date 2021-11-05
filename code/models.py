from datetime import datetime

import torch.optim
from torch.utils.tensorboard import SummaryWriter

from dataset import *
from model.resnet1d import *
from transfer_losses import TransferLoss

class EDLoss(nn.Module):
    def __init__(self):
        super(EDLoss, self).__init__()

    def forward(self, input, target):
        return torch.sum((input - target).pow(2), dim=1).mean()


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ConvBlockForward(nn.Module):
    """ Forward block performing 1d convolution, batchnorm and relu"""

    def __init__(self, in_dim: int, out_dim: int):
        super(ConvBlockForward, self).__init__()

        self.f = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.f.forward(x)


class ResidualConvBlockForward(nn.Module):
    def __init__(self, in_dim):
        super(ResidualConvBlockForward, self).__init__()
        self.f = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, 3, stride=1, padding=1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.f.forward(x)


class Encoder(nn.Module):
    """
        VAE Encoder module
    """

    def __init__(self, dims, sample_dims, in_channels: int = 1, in_samples: int = 2560, repeat_convs: int = 8):
        super(Encoder, self).__init__()
        self.layers = []

        for channel_dim, sample_dim in zip(dims, sample_dims):
            self.layers.append(
                nn.Sequential(
                    *[
                        ResidualConvBlockForward(in_channels) for i in range(repeat_convs - 1)]
                )
            )
            self.layers.append(
                nn.Sequential(
                    ConvBlockForward(in_channels, channel_dim),
                    nn.Linear(in_samples, sample_dim),
                    nn.ReLU()
                )
            )
            in_channels = channel_dim
            in_samples = sample_dim
        self.layers.append(SelfAttention(4, sample_dims[-1], sample_dims[-1]))

        self.layers.append(nn.Flatten())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        for l in self.layers:
            # print(x.shape)
            # print(l)
            x = l.forward(x)

        # print(x.shape)
        return x


class NET(nn.Module):
    def __init__(self, num_class, transfer_loss='coral', max_iter=1000, hidden_dims=None,
                 hidden_sample_dims=None, **kwargs):
        super(NET, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        if hidden_sample_dims is None:
            hidden_sample_dims = [1024, 512, 256, 128]
        self.num_class = num_class
        self.base_network = Encoder(hidden_dims, hidden_sample_dims, **kwargs)

        feature_dim = hidden_dims[-1] * hidden_sample_dims[-1]

        self.classifier_layer = nn.Linear(feature_dim, num_class)

        self.transfer_loss = transfer_loss
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, ecg):

        out = self.base_network(ecg)

        out = self.classifier_layer(out)

        return out


class TransferNet(nn.Module):
    def __init__(self, num_class, transfer_loss='coral', max_iter=1000, hidden_dims=None,
                 hidden_sample_dims=None, **kwargs):
        super(TransferNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        if hidden_sample_dims is None:
            hidden_sample_dims = [1000, 500, 250, 125]
        self.num_class = num_class
        self.base_network_ecg = NET(num_class)
        if os.path.exists('ecg.h5'):
            self.base_network_ecg.load_state_dict(torch.load('ecg.h5'))
            for layer in self.base_network_ecg.base_network.layers:
                layer.trainable = False
        self.base_network_ecg = self.base_network_ecg.base_network

        self.base_network_music = NET(num_class, **kwargs)

        if os.path.exists('music.h5'):
            self.base_network_music.load_state_dict(torch.load('music.h5'))
            for layer in self.base_network_music.base_network.layers:
                layer.trainable = False
        self.base_network_music = self.base_network_music.base_network

        feature_dim = hidden_dims[-1] * hidden_sample_dims[-1]

        self.latent = nn.Linear(feature_dim, 4096)

        # self.latent_music = self.base_network_music.latent

        self.classifier_layer_ecg = nn.Sequential()
        self.classifier_layer_music = nn.Sequential()

        # self.classifier_layer.add_module('att', nn.Att)
        self.classifier_layer_ecg.add_module('dense_ecg', nn.Linear(feature_dim, num_class))
        self.classifier_layer_music.add_module('dense_music', nn.Linear(feature_dim, num_class))

        self.transfer_loss = transfer_loss
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, ecg, music):

        ecg_out = self.base_network_ecg(ecg)
        music_out = self.base_network_music(music)
        kwargs = {}

        # classification
        # ecg_out = self.latent(ecg_out)
        # music_out = self.latent(music_out)
        ecg_out = self.classifier_layer_ecg(ecg_out)
        music_out = self.classifier_layer_music(music_out)

        # if self.transfer_loss == "coral":
        #     transfer_loss = self.adapt_loss(ecg_out, music_out, **kwargs)
        # else:
        #     transfer_loss =
        transfer_loss = 0

        return ecg_out, music_out, transfer_loss


class DMLNet(nn.Module):

    def __init__(self, num_class=4, inplanes=128, use_top=False, pretrain=True, **kwargs):
        super(DMLNet, self).__init__()

        self.ecg_model = nn.Sequential(
            resnet1d50(num_classes=num_class, input_channels=1, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
            # nn.Linear(inplanes * 8, inplanes * 4),
            # nn.BatchNorm1d(inplanes * 4),
            # nn.ReLU()
        )
        self.music_model = nn.Sequential(
            resnet1d50(num_classes=num_class, input_channels=88, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
            # nn.Linear(inplanes * 8, inplanes * 4),
            # nn.BatchNorm1d(inplanes * 4),
            # nn.ReLU()
        )
        if pretrain:
            self.load_pretrain_model(self.ecg_model[0], 'freeze_model/ecg_128.pth')
            self.load_pretrain_model(self.music_model[0], 'freeze_model/music_128.pth')

            for p in self.ecg_model[0].parameters():
                p.requires_grad = False

            for p in self.music_model[0].parameters():
                p.requires_grad = False

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
            nn.Linear(inplanes * 4 * 2, 1),
            nn.Sigmoid()
        )

        self.loss_fuc = nn.MSELoss()
        self.loss_fuc_2 = EDLoss()


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
        ecg_out = torch.cat([self.ecg_reg_v(ecg_feature), self.ecg_reg_a(ecg_feature)], dim=1)
        music_out = torch.cat([self.music_reg_v(music_feature), self.music_reg_a(music_feature)], dim=1)

        loss = self.loss_fuc(ecg_out, ecg_label) + self.loss_fuc(music_out, music_label) + \
               self.loss_fuc(distance, predict_distance)
        # loss = self.loss_fuc(distance, predict_distance)


        if test:
            print(ecg_out.detach().cpu().numpy())
            print(music_out.detach().cpu().numpy())
            print(predict_distance.detach().cpu().numpy())

        return loss, self.loss_fuc(ecg_out, ecg_label), self.loss_fuc(music_out, music_label),  self.loss_fuc(distance, predict_distance), ecg_out, music_out, predict_distance


class TripletNet(nn.Module):

    def __init__(self, num_class=2, inplanes=128, use_top=False, pretrain=True, **kwargs):
        super(TripletNet, self).__init__()

        self.ecg_model = nn.Sequential(
            resnet1d50(num_classes=num_class, input_channels=1, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU()
        )
        self.music_model = nn.Sequential(
            resnet1d50(num_classes=num_class, input_channels=88, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU()
        )
        if pretrain:
            self.ecg_model = self.load_pretrain_model(self.ecg_model, 'ecg_128.pth')
            self.music_model = self.load_pretrain_model(self.music_model, 'music_128.pth')

            # for p in self.ecg_model[0].parameters():
            #     p.requires_grad = False
            #
            # for p in self.music_model[0].parameters():
            #     p.requires_grad = False

        self.ecg_reg = nn.Sequential(
            nn.Linear(inplanes * 4, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 2),
            # nn.Sigmoid()
        )

        self.music_reg = nn.Sequential(
            nn.Linear(inplanes * 4, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 2),
            # nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            # Flatten(),
            # nn.BatchNorm1d(inplanes * 4 * 2),
            nn.Linear(inplanes * 4 * 2, inplanes * 4 * 2),
            nn.BatchNorm1d(inplanes * 4 * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 4 * 2, 1),
            # nn.Sigmoid()
        )

        self.loss_fuc = nn.MSELoss()
        self.loss_fuc_2 = EDLoss()

    def load_pretrain_model(self, model, path):
        model_dict = model.state_dict()
        pretrain = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrain.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, ecg, ecg_label, music, music_label, an_ecg, an_ecg_label, an_music, an_music_label, s1, s2, s3):
        ecg_feature = self.ecg_model(ecg)
        music_feature = self.music_model(music)

        an_ecg_feature = self.ecg_model(an_ecg)
        an_music_feature = self.music_model(an_music)

        # ecg_feature = AdaptiveConcatPool1d()(ecg_feature)
        # music_feature = AdaptiveConcatPool1d()(music_feature)
        # an_ecg_feature = AdaptiveConcatPool1d()(an_ecg_feature)
        # an_music_feature = AdaptiveConcatPool1d()(an_music_feature)

        ecg_music_pair = torch.cat([ecg_feature, music_feature], dim=1)
        predict_distance = self.fusion(ecg_music_pair)
        ecg_out = self.ecg_reg(ecg_feature)
        music_out = self.ecg_reg(music_feature)

        feature_distance_i_i = torch.sum((ecg_feature - music_feature).pow(2), dim=1) + 0.00001
        feature_distance_i_j = torch.sum((ecg_feature - an_music_feature).pow(2), dim=1) + 0.00001
        feature_distance_j_i = torch.sum((an_ecg_feature - music_feature).pow(2), dim=1) + 0.00001

        loss_cfr = (torch.sum(
            (torch.log(feature_distance_i_i / feature_distance_i_j) - torch.log(s1 / s2)).pow(2)) + torch.sum(
            (torch.log(feature_distance_i_i / feature_distance_j_i) - torch.log(s1 / s3)).pow(2))) / ecg.size()[0]

        index = torch.where(feature_distance_i_i - 25 > 0)
        loss_cfm = torch.sum(feature_distance_i_i[index] - 25) / ecg.size()[0]

        # loss_sfr = torch.sum(
        #     (torch.log(feature_distance_i_i / feature_distance_i_j) - torch.log(torch.pairwise_distance() / s2)).pow(2)) + torch.sum()

        loss = self.loss_fuc(ecg_out, ecg_label) + self.loss_fuc(music_out, music_label) + \
               self.loss_fuc(s1, predict_distance) + loss_cfr + loss_cfm

        return loss, self.loss_fuc(ecg_out, ecg_label), self.loss_fuc(music_out, music_label), self.loss_fuc(s1,
                                                                                                             predict_distance), loss_cfr, loss_cfm


class DMLNet2(nn.Module):

    def __init__(self, num_class=4, inplanes=128, use_top=False, pretrain=True, **kwargs):
        super(DMLNet2, self).__init__()

        self.ecg_model = nn.Sequential(
            resnet1d50(num_classes=num_class, input_channels=1, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU()
        )
        self.music_model = nn.Sequential(
            resnet1d50(num_classes=num_class, input_channels=88, inplanes=inplanes, use_top=use_top),
            AdaptiveConcatPool1d(),
            Flatten(),
            nn.Linear(inplanes * 8, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU()
        )
        if pretrain:
            self.ecg_model = self.load_pretrain_model(self.ecg_model, 'ecg_128.pth')
            self.music_model = self.load_pretrain_model(self.music_model, 'music_128.pth')

            # for p in self.ecg_model[0].parameters():
            #     p.requires_grad = False
            #
            # for p in self.music_model[0].parameters():
            #     p.requires_grad = False

        self.ecg_reg_a = nn.Sequential(
            nn.Linear(inplanes * 4, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            # nn.Sigmoid()
        )

        self.ecg_reg_v = nn.Sequential(
            nn.Linear(inplanes * 4, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            # nn.Sigmoid()
        )

        self.music_reg_a = nn.Sequential(
            nn.Linear(inplanes * 4, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            # nn.Sigmoid()
        )

        self.music_reg_v = nn.Sequential(
            nn.Linear(inplanes * 4, inplanes * 4),
            nn.BatchNorm1d(inplanes * 4),
            nn.ReLU(),
            nn.Linear(inplanes * 4, 1),
            # nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            # Flatten(),
            # nn.BatchNorm1d(inplanes * 4 * 2),
            nn.Linear(inplanes * 4 * 2, inplanes * 4 * 2),
            nn.BatchNorm1d(inplanes * 4 * 2),
            nn.ReLU(),
            nn.Linear(inplanes * 4 * 2, 1),
            # nn.Sigmoid()
        )

        self.loss_fuc = nn.MSELoss()
        self.loss_fuc_2 = EDLoss()


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

        # an_ecg_feature = self.ecg_model(an_ecg)
        # an_music_feature = self.music_model(an_music)

        # ecg_feature = AdaptiveConcatPool1d()(ecg_feature)
        # music_feature = AdaptiveConcatPool1d()(music_feature)
        # an_ecg_feature = AdaptiveConcatPool1d()(an_ecg_feature)
        # an_music_feature = AdaptiveConcatPool1d()(an_music_feature)

        ecg_music_pair = torch.cat([ecg_feature, music_feature], dim=1)
        predict_distance = self.fusion(ecg_music_pair)
        ecg_out = torch.cat([self.ecg_reg_v(ecg_feature), self.ecg_reg_a(ecg_feature)], dim=1)
        music_out = torch.cat([self.music_reg_v(music_feature), self.music_reg_a(music_feature)], dim=1)

        # feature_distance_i_i = torch.sum((ecg_feature - music_feature).pow(2), dim=1) + 0.00001
        # feature_distance_i_j = torch.sum((ecg_feature - an_music_feature).pow(2), dim=1) + 0.00001
        # feature_distance_j_i = torch.sum((an_ecg_feature - music_feature).pow(2), dim=1) + 0.00001

        # loss_cfr = (torch.sum(
        #     (torch.log(feature_distance_i_i / feature_distance_i_j) - torch.log(s1 / s2)).pow(2)) + torch.sum(
        #     (torch.log(feature_distance_i_i / feature_distance_j_i) - torch.log(s1 / s3)).pow(2))) / ecg.size()[0]
        #
        # index = torch.where(feature_distance_i_i - 25 > 0)
        # loss_cfm = torch.sum(feature_distance_i_i[index] - 25) / ecg.size()[0]

        # loss_sfr = torch.sum(
        #     (torch.log(feature_distance_i_i / feature_distance_i_j) - torch.log(torch.pairwise_distance() / s2)).pow(2)) + torch.sum()

        loss = self.loss_fuc(ecg_out, ecg_label) + self.loss_fuc(music_out, music_label) \
                + self.loss_fuc(distance, predict_distance)
        # loss = self.loss_fuc(distance, predict_distance)


        if test:
            print(ecg_out.detach().cpu().numpy())
            print(music_out.detach().cpu().numpy())
            # print(predict_distance.detach().cpu().numpy())

        return loss, self.loss_fuc(ecg_out, ecg_label), self.loss_fuc(music_out, music_label),  ecg_out, music_out

