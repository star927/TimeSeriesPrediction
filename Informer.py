import argparse
from exp.exp_informer import Exp_Informer
import torch
import pandas as pd
from data.data_loader import Dataset_ETT, Dataset_Weather
from torch.utils.data import DataLoader
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="[Informer] Long Sequences Forecasting")
parser.add_argument("--model", type=str, default="informer", help="[informer, informerstack]")
parser.add_argument("--data", type=str, default="ETT", help="[ETT, Weather_WH, Weather_SZ, Weather_GZ]")
parser.add_argument("--root_path", type=str, default="./DataSet", help="root path of the data file")
parser.add_argument("--features", type=str, default="MS", help="[M, S, MS]")
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
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument("--output_attention", type=bool, default=False, help="whether to output attention in ecoder")
parser.add_argument("--mix", type=bool, default=True, help="use mix attention in generative decoder")
parser.add_argument("--padding", type=int, default=0, help="padding type")

parser.add_argument("--attn", type=str, default="prob", help="[prob, full]")
parser.add_argument("--distil", action="store_true", default=False, help="whether to use distilling in encoder")
parser.add_argument("--transformer_dec", action="store_true", default=False, help="????????????Transformer???Decoder??????")

parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
parser.add_argument("--loss", type=str, default="huber", help="?????????????????????????????????nn.MSELoss()")

parser.add_argument("--num_workers", type=int, default=0, help="DataLoader()?????????")
parser.add_argument("--exp_num", type=int, default=1, help="????????????")
parser.add_argument("--train_epochs", type=int, default=6, help="train epochs")

args = parser.parse_args()
try:  # ?????????????????????list???????????????????????????list?????????????????????????????????try???????????????????????????
    args.s_layers = eval("".join(args.s_layers))
except:  # ??????????????????????????????????????????try????????????????????????????????????????????????
    pass
try:
    args.inp_lens = eval("".join(args.inp_lens))
except:
    pass


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
    "ETT": {"data_path": "ETT.csv", "Target": "OT", "M": [7, 7, 7], "S": [1, 1, 1], "MS": [7, 7, 1]},
    "Weather_WH": {"data_path": "Weather_WH.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
    "Weather_SZ": {"data_path": "Weather_SZ.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
    "Weather_GZ": {"data_path": "Weather_GZ.csv", "Target": "T", "M": [6, 6, 6], "S": [1, 1, 1], "MS": [6, 6, 1]},
}
data_info = data_parser[args.data]
args.data_path = data_info["data_path"]
args.target = data_info["Target"]
args.enc_in, args.dec_in, args.c_out = data_info[args.features]
if args.data == "ETT":
    args.dataset_flag = "ETT"
else:  # ???????????????
    args.dataset_flag = "Weather"

if args.transformer_dec and (args.features == "MS" or args.features == "S"):
    args.dec_in = 1

print("Args in experiment:")
print(args)

Exp = Exp_Informer

for ii in range(args.exp_num):
    # setting record of experiments
    setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_dt{}_mx{}_{}".format(
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
        args.distil,
        args.mix,
        ii,
    )

    # set experiments
    exp = Exp(args)

    # train
    print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    actual_train_epochs, train_cost_time = exp.train(setting)

    # test
    print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
    mse, mae = exp.test()

    path_result = "./result_informer.csv"
    open(path_result, "a").close()

    result = vars(args)
    result["actual_train_epochs"] = actual_train_epochs
    result["train_cost_time"] = train_cost_time
    result["mse"] = mse
    result["mae"] = mae
    try:
        df = pd.read_csv(path_result, header=0)
        pd.concat([df, pd.DataFrame([result])]).to_csv(path_result, index=False)
    except:
        pd.DataFrame([result]).to_csv(path_result, index=False)

    torch.cuda.empty_cache()

    # ???????????????????????????
    # data_dict = {
    #     'ETT': Dataset_ETT,
    #     'Weather_WH': Dataset_Weather,
    #     'Weather_SZ': Dataset_Weather,
    #     'Weather_GZ': Dataset_Weather,
    # }
    # Data = data_dict[args.data]
    # data_set = Data(
    #     root_path=args.root_path,
    #     data_path=args.data_path,
    #     flag="test",
    #     size=[args.seq_len, args.label_len, args.pred_len],
    #     features=args.features,
    # )
    # data_loader = DataLoader(
    #     data_set, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=True
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
    # # args.model=informer???, attn???list??????????????????encoder?????????attn??????????????????(1, head_num, query_num, key_num)
    # # ??????attn[e][0, h]?????????tensor, ?????????(e+1)???encoder???(h+1)????????????????????????????????????
    # # args.model=informerstack???, attn???list??????????????????stack?????????attn??????????????????list?????????????????????stack???encoder??????
    # # ??????attn[s][e][0, h]?????????tensor, ?????????(s+1)???stack?????????(e+1)???encoder???(h+1)????????????????????????????????????
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
    # ???????????????????????????
