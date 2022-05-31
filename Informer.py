import argparse
from exp.exp_informer import Exp_Informer
import torch
from data.data_loader import Dataset_ETT_hour
from torch.utils.data import DataLoader
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="[Informer] Long Sequences Forecasting")
parser.add_argument("--model", type=str, default="informer", help="[informer, informerstack]")
parser.add_argument("--data", type=str, default="ETTh1", help="[ETTh1, Weather_WH, Weather_SZ, Weather_GZ]")
parser.add_argument("--root_path", type=str, default="./DataSet", help="root path of the data file")
parser.add_argument("--features", type=str, default="M", help="[M, S, MS]")
parser.add_argument("--freq", type=str, default="h", help="[h, wh], h对应ETTh1, wh对应天气数据集")
parser.add_argument("--checkpoints", type=str, default="./informer_checkpoints/", help="location of model checkpoints")

parser.add_argument("--seq_len", type=int, default=96, help="input sequence length of Informer encoder")
parser.add_argument("--label_len", type=int, default=48, help="start token length of Informer decoder")
parser.add_argument("--pred_len", type=int, default=24, help="prediction sequence length")
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument("--factor", type=int, default=5, help="probsparse attn factor")
parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--s_layers", type=list, default=[3, 1], help="num of stack encoder layers")
parser.add_argument("--inp_lens", type=list, default=[0, 2], help="")
parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
parser.add_argument("--embed", type=str, default="timeF", help="[timeF, fixed, learned]")
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument("--output_attention", type=bool, default=False, help="whether to output attention in ecoder")
parser.add_argument("--mix", type=bool, default=True, help="use mix attention in generative decoder")
parser.add_argument("--padding", type=int, default=0, help="padding type")

parser.add_argument("--attn", type=str, default="prob", help="[prob, full]")
parser.add_argument("--distil", action='store_true', default=False, help="whether to use distilling in encoder")
parser.add_argument("--dec_one_by_one", action='store_true', default=False, help="whether dec_one_by_one")

parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
parser.add_argument("--loss", type=str, default="huber", help="模型用的损失函数，默认nn.MSELoss()")
parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
parser.add_argument("--use_amp", type=bool, default=False, help="use automatic mixed precision training")

parser.add_argument("--num_workers", type=int, default=0, help="DataLoader()的参数")
parser.add_argument("--exp_num", type=int, default=1, help="实验次数")
parser.add_argument("--train_epochs", type=int, default=6, help="train epochs")
parser.add_argument("--patience", type=int, default=3, help="连续patience个epoch的验证集误差比最小验证集误差大，则停止训练")
parser.add_argument("--des", type=str, default="exp", help="exp description")

args = parser.parse_args()
args.inverse = None
args.cols = None

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
    "Weather_WH": {"data_path": "Weather_WH.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
    "Weather_SZ": {"data_path": "Weather_SZ.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
    "Weather_GZ": {"data_path": "Weather_GZ.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
}
data_info = data_parser[args.data]
args.data_path = data_info["data_path"]
args.target = data_info["Target"]
args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq  # 预测时相关

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
