import math

import torch as t
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DSEncoder(nn.Module):
    def __init__(self, args):
        super(DSEncoder, self).__init__()  # 初始化参数
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.n_head = args['n_head']
        self.att_out = args['att_out']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['lstm_encoder_size']
        self.train_flag = args['train_flag']
        self.use_elu = args['use_elu']

        self.dropout = args['dropout']
        # traj encoder
        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)  # in = 5 out = 32
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)  # in = 32 out = 64
        # activation function
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        #  attention embeding

        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.project0 = nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.project1 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 4)
        self.project2 = nn.Linear(self.lstm_encoder_size * 4, self.lstm_encoder_size)
        self.qtt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.ktt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vtt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.project10 = nn.Linear(self.n_head * self.att_out, self.lstm_encoder_size)
        self.project11 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 4)

        self.glu = GLU(
            input_size=self.lstm_encoder_size * 4,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)
        self.Dropout = nn.Dropout(p=self.dropout)
        #  addAndNorm
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)  # layer norm
        #  style pred
        self.style_pred = nn.Linear(self.lstm_encoder_size, 3)
        #  style attention
        self.mapping = t.nn.Parameter(
            t.Tensor(3, self.lstm_encoder_size))  # 3*s_l
        nn.init.xavier_uniform_(self.mapping, gain=1.414)

        self.d_model = self.traj_linear_hidden
        self.max_seq_len = self.in_length + 1
        self.positional_encoding = self.generate_positional_encoding()
        self.ds_aw = t.zeros([3, self.in_length], dtype=t.float32)

    def generate_positional_encoding(self):
        # 生成位置编码矩阵
        pe = t.zeros(self.max_seq_len, self.d_model)
        position = t.arange(0, self.max_seq_len, dtype=t.float32).unsqueeze(1)
        div_term = t.exp(t.arange(0, self.d_model, 2, dtype=t.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)
        return pe

    def forward(self, Hist):
        # lane = lane.unsqueeze(-1)
        # Hist = t.cat((Hist, lane), -1)
        Hist = Hist.permute(1, 0, 2)
        if self.train_flag is True:
            hist_batch = t.zeros([self.in_length, (self.out_length + 1) * Hist.shape[1], self.f_length])
            hist_batch = hist_batch.to(self.device)
            # DS = t.zeros([self.out_length+1, Hist.shape[1],self.lstm_encoder_size])

            for i in range(self.out_length + 1):
                hist_batch[:, i * Hist.shape[1]:(i + 1) * Hist.shape[1], :] = Hist[i:self.in_length + i, :, :]
        else:
            hist_batch = Hist

            # hist = Hist[i:self.in_length+i,:,:]

        hist_enc = self.activation(self.linear1(hist_batch))  # 轨迹编码 输出 历史时刻长度、batch*in_length、特征维度

        hist_enc += self.positional_encoding[:, :self.in_length, :].clone().detach().permute(1, 0, 2).repeat(1,hist_enc.shape[
                                                                                                                        1],
                                                                                                                    1)

        
        hist_hidden_enc = hist_enc.permute(1, 0, 2)
        # m_head attention begin---------------------------
        qt = t.cat(t.split(self.qt(hist_hidden_enc), int(hist_hidden_enc.shape[-1] / self.n_head), -1),
                   0)  # batch * n_head, sq_l, f_head
        kt = t.cat(t.split(self.kt(hist_hidden_enc), int(hist_hidden_enc.shape[-1] / self.n_head), -1), 0).permute(0, 2,
                                                                                                                   1)
        vt = t.cat(t.split(self.vt(hist_hidden_enc), int(hist_hidden_enc.shape[-1] / self.n_head), -1), 0)
        a = t.matmul(qt, kt)
        a /= math.sqrt(self.att_out)
        a = t.softmax(a, -1)
        values = t.matmul(a, vt)
        values1 = t.cat(t.split(values, int(hist_enc.shape[1]), 0), -1)
        values1 = self.activation(self.project0(values1))
        values1 = self.addAndNorm(hist_hidden_enc, values1)
        values2 = self.Dropout(self.activation(self.project1(values1)))
        values3 = self.Dropout(self.activation(self.project2(values2)))
        values3 = self.addAndNorm(values1, values3)

        token_ds = self.positional_encoding[:, -1:, :].clone().detach().repeat(values3.shape[0], 1, 1)
        query_ds = t.cat((values3, token_ds), 1)

        qtt = t.cat(t.split(self.qtt(query_ds), int(hist_hidden_enc.shape[-1] / self.n_head), -1),
                    0)  # batch * n_head, sq_l, f_head
        ktt = t.cat(t.split(self.ktt(values3), int(hist_hidden_enc.shape[-1] / self.n_head), -1), 0).permute(0, 2,
                                                                                                             1)
        vtt = t.cat(t.split(self.vtt(values3), int(hist_hidden_enc.shape[-1] / self.n_head), -1), 0)
        a = t.matmul(qtt, ktt)
        a /= math.sqrt(self.att_out)
        a = t.softmax(a, -1)
        values = t.matmul(a, vtt)
        values1 = t.cat(t.split(values, int(hist_enc.shape[1]), 0), -1)
        values1 = self.activation(self.project10(values1))
        values1 = self.addAndNorm(query_ds, values1)
        values2 = self.Dropout(self.activation(self.project11(values1)))

        # ------------------------
        # gate
        time_values, _ = self.glu(values2)
        # Residual connection,

        values4 = self.addAndNorm(values1, time_values)  # batch s_l f

        ds_pred = F.softmax(self.style_pred(values4[:, -1, :]), dim=-1).unsqueeze(1)  # batch 1 3
        # ds_pred = t.zeros(values4.shape[0], 1, 3).to(self.device)
        # ds_pred[:, :, 2] = 1
        ds_pred2 = t.zeros_like(ds_pred).to(self.device)
        ds_pred_index = t.argmax(ds_pred.squeeze(), dim=-1)

        for ds_i in range(ds_pred2.shape[0]):
            ds_pred2[ds_i, :, ds_pred_index[ds_i]] = 1
        ds_att = F.softmax(t.matmul(ds_pred2, t.matmul(self.mapping, values4[:, :-1, :].permute(0, 2, 1))),
                           dim=-1)# batch 1 s_l
        for ds_i in range(0, 3):
            self.ds_aw[ds_i, :] = t.sum(ds_att.squeeze()[ds_pred_index == ds_i], dim=0)


        ds = t.matmul(ds_att, values4[:, :-1, :]).permute(1, 0, 2)
        if self.train_flag is True:
            DS = t.cat(t.split(ds, Hist.shape[1], 1), 0)
            return DS
        else:
            # return ds, t.cat((ds_att, ds_pred, ds_pred2), -1).squeeze()
            return ds
        # O_L batch  f 最后一维包含了空间和时间的状态信息


class StateEncoder(nn.Module):
    def __init__(self, args):
        super(StateEncoder, self).__init__()  # 初始化参数
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.train_flag = args['train_flag']
        self.in_length = args['in_length']

        self.f_length = args['f_length']
        self.relu_param = args['relu']
        self.traj_linear_hidden = args['traj_linear_hidden']

        self.use_elu = args['use_elu']

        self.dropout = args['dropout']
        # traj encoder
        self.linear1 = nn.Linear(3, self.traj_linear_hidden)  # in = 5 out = 32
        self.Dropout = nn.Dropout(p=self.dropout)
        # activation function
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        self.project0 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 2)
        self.project1 = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)
        self.project2 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 2)
        self.glu = GLU(
            input_size=self.lstm_encoder_size * 2,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)

    def forward(self, Hist):
        # Hist = t.as_tensor(Hist, dtype=self.linear1.weight.dtype)
        Hist = Hist.permute(1, 0, 2)

        Hist = Hist[self.in_length - 1:, :, :]


        states0 = self.activation(self.linear1(Hist))

        states1 = self.Dropout(self.tanh(self.project0(states0)))
        states2 = self.Dropout(self.tanh(self.project1(states1)))
        states2 = self.addAndNorm(states0, states2)
        states3 = self.Dropout(self.activation(self.project2(states2)))
        # gate
        states, _ = self.glu(states3)

        return states


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = t.add(t.add(x1, x2), x3)
        else:
            x = t.add(x1, x2)
        return self.normalize(x)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.dropout = args['dropout']
        self.glu = GLU(
            input_size=self.lstm_encoder_size * 2,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)

    def forward(self, state, ds):
        # gate = self.sigmoid(state+ds)
        # KS_ALL = gate * state + (1 - gate) * ds
        S = t.cat((state, ds), dim=-1)
        KS_ALL, _ = self.glu(S)
        return KS_ALL


class Koop_space(nn.Module):
    def __init__(self, args):
        super(Koop_space, self).__init__()  # 初始化参数
        self.args = args
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']
        self.train_flag = args['train_flag']
        self.A = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size, bias=False)
        self.B = nn.Linear(1, self.lstm_encoder_size, bias=False)
        self.C = nn.Linear(self.lstm_encoder_size, 3, bias=False)
        self.dec = decoder(self.args)
        self.liner_dec = args['liner_dec']

    def forward(self, KS_ALL, next_vf):
        # S = t.cat((state, ds), dim=-1)
        # KS_ALL, _ = self.glu(S) #O_L+1 BATCH FQ
        next_vf = next_vf.unsqueeze(-1).permute(1, 0, 2)
        if self.train_flag is True:
            ks = KS_ALL[0:1, :, :]  # 1 BATCH F
        # KS_pre = t.zeros_like(KS_ALL)
        else:
            ks = KS_ALL
        KS_pre = ks
        for i in range(next_vf.shape[0]):
            ks = self.A(ks) + self.B(next_vf[i:i + 1, :, :])
            KS_pre = t.cat((KS_pre, ks), dim=0)
        if self.liner_dec:
            OS = self.C(KS_pre)
        else:
            OS = self.dec(KS_pre)

        return KS_pre, OS


# gate
class GLU(nn.Module):
    # Gated Linear Unit+
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 dropout_rate=None,
                 ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return t.mul(activation, gated), gated


class decoder(nn.Module):
    def __init__(self, args):
        super(decoder, self).__init__()  # 初始化参数
        self.device = args['device']
        self.lstm_encoder_size = args['lstm_encoder_size']

        self.tanh = nn.Tanh()

        self.relu_param = args['relu']

        self.use_elu = args['use_elu']

        self.dropout = args['dropout']

        self.Dropout = nn.Dropout(p=self.dropout)
        # activation function
        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        self.project0 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size * 2)
        self.project1 = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)
        self.project2 = nn.Linear(self.lstm_encoder_size, self.lstm_encoder_size)
        self.project3 = nn.Linear(self.lstm_encoder_size, 3)

    def forward(self, x):

        states1 = self.Dropout(self.tanh(self.project0(x)))
        states2 = self.Dropout(self.tanh(self.project1(states1)))
        states2 = self.addAndNorm(x, states2)
        states3 = self.Dropout(self.activation(self.project2(states2)))
        y = self.project3(states3)

        return y
