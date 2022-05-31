import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
from data.data_loader import Dataset_ETT_hour, Dataset_Weather
from utils.tools import adjust_learning_rate, dotdict
from utils.metrics import metric


class Encoder(nn.Module):
    def __init__(self, cell, enc_in, num_hidden, num_layer, dropout=0.0):
        super(Encoder, self).__init__()
        assert cell == "GRU" or cell == "LSTM"
        self.cell = cell
        if self.cell == "GRU":
            self.rnn = nn.GRU(enc_in, num_hidden, num_layer, dropout=dropout)
        elif self.cell == "LSTM":
            self.rnn = nn.LSTM(enc_in, num_hidden, num_layer, dropout=dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, enc_in)
        # 在循环神经网络模型中，第一个轴对应于时间步
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, enc_in)
        if self.cell == "GRU":
            output, state = self.rnn(x)
            # output: (seq_len, batch_size, num_hidden)
            # state: (num_layer, batch_size, num_hidden)
            return output, state
        elif self.cell == "LSTM":
            output, (H, C) = self.rnn(x)
            # output: (seq_len, batch_size, num_hidden)
            # H: (num_layer, batch_size, num_hidden)
            # C: (num_layer, batch_size, num_hidden)
            return output, (H, C)


class Decoder(nn.Module):
    def __init__(self, cell, dec_in, num_hidden, num_layer, dec_out, dropout=0.0):
        super(Decoder, self).__init__()
        assert cell == "GRU" or cell == "LSTM"
        self.cell = cell
        if self.cell == "GRU":
            self.rnn = nn.GRU(dec_in + num_hidden, num_hidden, num_layer, dropout=dropout)
        elif self.cell == "LSTM":
            self.rnn = nn.LSTM(dec_in + num_hidden, num_hidden, num_layer, dropout=dropout)
        self.dense = nn.Linear(num_hidden, dec_out)

    def init_state(self, state):
        # 如果是LSTM, state = (H, C)
        self.enc_hidden_state = state if self.cell == "GRU" else state[0]
        # (num_layer, batch_size, num_hidden)
        self.enc_hidden_state = self.enc_hidden_state[-1]  # Decoder最后一层的最后的隐状态
        # (batch_size, num_hidden)

    def forward(self, x, state):
        # 训练 x: (batch_size, pred_len, dec_in)
        # 预测 x: (batch_size, 1, dec_in)
        x = x.permute(1, 0, 2)
        # 广播context，使其具有与x相同长度的时间步
        context = self.enc_hidden_state.repeat(x.shape[0], 1, 1)
        x_and_context = torch.cat((x, context), dim=2)
        output, state = self.rnn(x_and_context, state)
        # output: (pred_len或1, batch_size, num_hidden)
        # state: (num_layer, batch_size, num_hidden)
        output = self.dense(output).permute(1, 0, 2)
        # output: (batch_size, pred_len或1, dec_out)
        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, args):
        super(EncoderDecoder, self).__init__()
        self.args = args
        self.encoder = Encoder(
            self.args.cell,
            self.args.enc_in,
            self.args.num_hidden,
            self.args.num_layer,
            self.args.dropout,
        )
        self.decoder = Decoder(
            self.args.cell,
            self.args.dec_in,
            self.args.num_hidden,
            self.args.num_layer,
            self.args.dec_out,
            self.args.dropout,
        )

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # batch_x: (batch_size, seq_len, 变量的个数)
        # batch_x_mask: (batch_size, seq_len, 时间特征的维度数) 时间特征的维度数，如月、日、小时
        # batch_y: (batch_size, label_len + pred_len, 变量的个数)
        # batch_x_mask: (batch_size, label_len + pred_len, 时间特征的维度数)

        x_enc = torch.cat([batch_x, batch_x_mark], dim=2).float()
        # x_dec = torch.cat([batch_y, batch_y_mark], dim=2).float()
        enc_out, state = self.encoder(x_enc)
        self.decoder.init_state(state)
        if self.training:  # 训练模式，Decoder输入的是准确的数据
            x_dec = torch.cat([batch_y[:, :-1, :], batch_y_mark[:, :-1, :]], dim=2).float()
            dec_out, state = self.decoder(x_dec, state)
            return dec_out

        # eval模式，Decoder的输入来自上一个Decoder单元的预测结果和Encoder最后的状态
        x_dec = batch_y[:, [0], :]
        out = []
        for i in range(self.args.pred_len):
            x_dec = torch.cat([x_dec, batch_y_mark[:, [i], :]], dim=2).float()
            dec_out, state = self.decoder(x_dec, state)
            out.append(dec_out)
            x_dec = dec_out
        pred = torch.cat(out, dim=1)
        return pred


class Exp_seq2seq:
    def __init__(self, args):
        super(Exp_seq2seq, self).__init__()
        self.args = args
        self.exp_info = {}
        self.model = EncoderDecoder(self.args)
        self.model.to(self.args.device)

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            "ETTh1": Dataset_ETT_hour,
            "Weather_WH": Dataset_Weather,
            "Weather_SZ": Dataset_Weather,
            "Weather_GZ": Dataset_Weather,
        }
        Data = data_dict[self.args.data]
        # timeenc = 0 if args.embed!='timeF' else 1
        timeenc = 1

        if flag == "test":
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last
        )

        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
            # break
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()

        min_vali_loss = 1e10
        bad_train = 0
        iter_count = 0
        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                # batch_x: (batch_size, seq_len, 变量的个数)
                # batch_x_mask: (batch_size, seq_len, 时间特征的维度数) 时间特征的维度数，如月、日、小时
                # batch_y: (batch_size, label_len + pred_len, 变量的个数)
                # batch_x_mask: (batch_size, label_len + pred_len, 时间特征的维度数)
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))

                # break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 对一个epoch中的训练集误差求平均
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            if vali_loss < min_vali_loss:
                min_vali_loss = vali_loss
                bad_train = 0
            else:
                bad_train += 1
                if bad_train == 3:
                    break

            # 每经过一个epoch，学习率变为原来1/2
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            # break

        print("Train, cost time: {}".format(time.time() - time_now))

    def test(self):
        test_data, test_loader = self._get_data(flag="test")

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # batch_x: (batch_size, seq_len, 变量的个数)
        # batch_x_mask: (batch_size, seq_len, 时间特征的维度数) 时间特征的维度数，如月、日、小时
        # batch_y: (batch_size, label_len + pred_len, 变量的个数)
        # batch_x_mask: (batch_size, label_len + pred_len, 时间特征的维度数)

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if self.args.features == "MS":
            batch_y = batch_y[:, :, [-1]]

        pred = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
        true = batch_y[:, -self.args.pred_len :, :]

        return pred, true


if __name__ == "__main__":
    args = dotdict()

    # args.cell = "GRU"  # 编码器和解码器使用的模型
    args.cell = "LSTM"
    args.num_hidden = 16  # 隐藏层的节点个数
    args.num_layer = 2  # 隐藏层的层数
    args.dropout = 0.1
    args.batch_size = 32
    args.seq_len = 30  # 编码器输入长度
    args.label_len = 1
    args.pred_len = 7  # 解码器预测长度
    args.learning_rate = 0.003
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.train_epochs = 2

    # args.data = "ETTh1"
    args.data = "Weather_WH"
    args.root_path = "./DataSet"
    args.freq = "h"
    # args.freq = "wh"

    args.features = "MS"  # M, S, MS

    # ETTh1: HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    # Weather: Month, Day, Hour, Po, P, U, Ff, Td, T
    data_parser = {
        "ETTh1": {"data_path": "ETTh1.csv", "Target": "OT", "M": [11, 11, 11], "S": [5, 5, 1], "MS": [11, 5, 1]},
        "Weather_WH": {"data_path": "Weather_WH.csv", "Target": "T", "M": [9, 9, 6], "S": [4, 4, 1], "MS": [9, 4, 1]},
        "Weather_SZ": {"data_path": "Weather_SZ.csv", "Target": "T", "M": [9, 9, 6], "S": [4, 4, 1], "MS": [9, 4, 1]},
        "Weather_GZ": {"data_path": "Weather_GZ.csv", "Target": "T", "M": [9, 9, 6], "S": [4, 4, 1], "MS": [9, 4, 1]},
    }

    data_info = data_parser[args.data]
    args.data_path = data_info["data_path"]
    args.target = data_info["Target"]
    # 编码器输入变量的个数, 解码器输入变量的个数, 解码器预测变量的个数
    args.enc_in, args.dec_in, args.dec_out = data_info[args.features]

    args.num_workers = 0  # DataLoader()的参数

    exp = Exp_seq2seq(args)

    print(">>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>")
    exp.train()

    # test
    print(">>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    exp.test()

    torch.cuda.empty_cache()
