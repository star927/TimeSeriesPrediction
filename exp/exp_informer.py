from data.data_loader import Dataset_ETT, Dataset_Weather
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings("ignore")


class Exp_Informer:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model_dict = {
            "informer": Informer,
            "informerstack": InformerStack,
        }
        if self.args.model == "informer" or self.args.model == "informerstack":
            e_layers = self.args.e_layers if self.args.model == "informer" else self.args.s_layers
            model = model_dict[self.args.model](
                enc_in=self.args.enc_in,
                dec_in=self.args.dec_in,
                c_out=self.args.c_out,
                seq_len=self.args.seq_len,
                label_len=self.args.label_len,
                out_len=self.args.pred_len,
                factor=self.args.factor,
                d_model=self.args.d_model,
                n_heads=self.args.n_heads,
                e_layers=e_layers,  # self.args.e_layers,
                d_layers=self.args.d_layers,
                d_ff=self.args.d_ff,
                dropout=self.args.dropout,
                attn=self.args.attn,
                dataset_flag=self.args.dataset_flag,
                activation=self.args.activation,
                output_attention=self.args.output_attention,
                distil=self.args.distil,
                mix=self.args.mix,
                device=self.device,
                dec_one_by_one=self.args.dec_one_by_one,
                features=self.args.features,
                inp_lens=self.args.inp_lens,
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            "ETT": Dataset_ETT,
            "Weather_WH": Dataset_Weather,
            "Weather_SZ": Dataset_Weather,
            "Weather_GZ": Dataset_Weather,
        }
        Data = data_dict[self.args.data]

        shuffle_flag = False if flag == "test" else True
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set, batch_size=args.batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=True
        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss.lower() == "huber":
            return nn.HuberLoss()
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping()

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        iter_count = 0
        actual_train_epochs = self.args.train_epochs
        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                # batch_x: (batch_size, seq_len, enc_in)
                # batch_x_mask: (batch_size, seq_len, time_num) time_num时间特征的维度数，如月、日、周几、小时
                # batch_y: (batch_size, label_len + pred_len, dec_in)
                # batch_x_mask: (batch_size, label_len + pred_len, time_num)
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))

                loss.backward()
                model_optim.step()

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
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                actual_train_epochs = epoch + 1
                break

            # 每经过一个epoch，学习率变为原来1/2
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        train_cost_time = time.time() - time_now
        print("Train, cost time: {}".format(train_cost_time))
        return actual_train_epochs, train_cost_time

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
        preds = preds.reshape((-1, preds.shape[-2], preds.shape[-1]))
        trues = trues.reshape((-1, trues.shape[-2], trues.shape[-1]))
        print("test shape:", preds.shape, trues.shape)

        mse, mae = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

        return mse, mae

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # batch_y: (batch_size, label_len + pred_len, dec_in)
        dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

        return outputs, batch_y
