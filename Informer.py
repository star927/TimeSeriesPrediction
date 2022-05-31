from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
from data.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns


args = dotdict()

args.model = "informer"  # model of experiment, options: [informer, informerstack, informerlight(TBD)]
# args.model = 'informerstack'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]

args.data = "ETTh1"  # data
# args.data = 'Weather_WH'
args.root_path = "./DataSet"  # root path of data file
# args.data_path = 'ETTh1.csv'  # data file
args.features = "M"  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
# args.target = 'OT'  # target feature in S or MS task
args.freq = "h"  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
# args.freq = 'wh'  # 天气数据集，小时
args.checkpoints = "./informer_checkpoints"  # location of model checkpoints

args.seq_len = 96  # input sequence length of Informer encoder
args.label_len = 48  # start token length of Informer decoder
args.pred_len = 24  # prediction sequence length
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

# args.enc_in = 7  # encoder input size
# args.dec_in = 7  # decoder input size
# args.c_out = 7  # output size
args.factor = 5  # probsparse attn factor
args.d_model = 512  # dimension of model
args.n_heads = 8  # num of heads
args.e_layers = 3  # num of encoder layers
args.s_layers = [3, 1]  # HX, 对应InformerStack模型
args.inp_lens = [0, 2]
args.d_layers = 2  # num of decoder layers
args.d_ff = 2048  # dimension of fcn in model
args.dropout = 0.05  # dropout
# args.attn = "prob"  # attention used in encoder, options:[prob, full]
args.attn = "full"
args.embed = "timeF"  # time features encoding, options:[timeF, fixed, learned]
# args.embed = 'fixed'
args.activation = "gelu"  # activation
# args.distil = True  # whether to use distilling in encoder
args.distil = False  # whether to use distilling in encoder
args.output_attention = False  # whether to output attention in ecoder
args.mix = True
args.padding = 0
# args.freq = 'h'


# args.dec_one_by_one = False
args.dec_one_by_one = True

args.batch_size = 32
args.learning_rate = 0.0001
args.loss = "huber"  # 模型用的损失函数，默认nn.MSELoss()
args.lradj = "type1"
args.use_amp = False  # whether to use automatic mixed precision training

args.num_workers = 0  # DataLoader()的参数
args.exp_num = 1  # 实验次数
args.train_epochs = 6  # 训练集最大epoch数
args.patience = 3  # 连续patience个epoch的验证集误差比最小验证集误差大，则停止训练
args.des = "exp"

args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0

args.use_multi_gpu = False
args.devices = "0,1,2,3"

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(" ", "")
    device_ids = args.devices.split(",")
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# Set augments by using data name
data_parser = {
    "ETTh1": {"data_path": "ETTh1.csv", "Target": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "ETTh2": {"data_path": "ETTh2.csv", "Target": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "ETTm1": {"data_path": "ETTm1.csv", "Target": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "ETTm2": {"data_path": "ETTm2.csv", "Target": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "Weather_WH": {"data_path": "Weather_WH.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
    "Weather_SZ": {"data_path": "Weather_SZ.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
    "Weather_GZ": {"data_path": "Weather_GZ.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
}

data_info = data_parser[args.data]
args.data_path = data_info["data_path"]
args.target = data_info["Target"]
args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq  # 预测时相关
# args.freq = args.freq[-1:]

if args.dec_one_by_one and (args.features == "MS" or args.features == "S"):
    args.dec_in = 1

print("Args in experiment:")
print(args)

Exp = Exp_Informer

for ii in range(args.exp_num):
    # setting record of experiments
    setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}".format(
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.attn,
        args.factor,
        args.embed,
        args.distil,
        args.mix,
        args.des,
        ii,
    )

    # set experiments
    exp = Exp(args)

    # train
    print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    exp.train(setting)

    # test
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    exp.test(setting)

    torch.cuda.empty_cache()

    # 预测
    # exp.predict(setting, True)
    # prediction = np.load('./results/' + setting + '/real_prediction.npy')
    # print(prediction.shape)
    # 预测

    # 训练集上的结果
    # preds = np.load('./results/' + setting + '/pred.npy')
    # trues = np.load('./results/' + setting + '/true.npy')
    # plt.figure()
    # plt.plot(trues[0, :, -1], label='GroundTruth')
    # plt.plot(preds[0, :, -1], label='Prediction')
    # plt.legend()
    # plt.show()
    # 训练集上的结果

    # 可视化注意力分数图
    # Data = Dataset_ETT_hour
    # timeenc = 0 if args.embed != "timeF" else 1
    # flag = "test"
    # shuffle_flag = False
    # drop_last = True
    # batch_size = 1
    # data_set = Data(
    #     root_path=args.root_path,
    #     data_path=args.data_path,
    #     flag=flag,
    #     size=[args.seq_len, args.label_len, args.pred_len],
    #     features=args.features,
    #     timeenc=timeenc,
    #     freq=args.freq,
    # )
    # data_loader = DataLoader(
    #     data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last
    # )
    #
    # args.output_attention = True
    # exp = Exp(args)
    # model = exp.model
    # path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    # model.load_state_dict(torch.load(path))
    #
    # idx = 0
    # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
    #     if i != idx:
    #         continue
    #     batch_x = batch_x.float().to(exp.device)
    #     batch_y = batch_y.float()
    #
    #     batch_x_mark = batch_x_mark.float().to(exp.device)
    #     batch_y_mark = batch_y_mark.float().to(exp.device)
    #
    #     dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    #     dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
    #
    #     outputs, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #
    # # args.model=informer时, attn是list类型，长度是encoder个数，attn中的每个元素(1, head_num, query_num, key_num)
    # # 例：attn[e][0, h]是二维tensor, 表示第(e+1)个encoder第(h+1)个注意力头的注意力分数，
    # # args.model=informerstack时, attn是list类型，长度是stack个数，attn中每个元素是list类型，长度是该stack的encoder个数
    # # 例：attn[s][e][0, h]是二维tensor, 表示第(s+1)个stack中的第(e+1)个encoder第(h+1)个注意力头的注意力分数，
    #
    # stack = 0
    # layer = 0
    # distil = 'Distil' if args.distil else 'NoDistil'
    # for h in range(0, 8):
    #     plt.figure(figsize=[10, 8])
    #     plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
    #     A = attn[layer][0, h].detach().cpu().numpy()
    #     # A = attn[stack][layer][0, h].detach().cpu().numpy()
    #     ax = sns.heatmap(A, vmin=0, vmax=A.max() + 0.01)
    # plt.show()
    # 可视化注意力分数图
