import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        dataset_flag="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
        device=torch.device("cuda:0"),
        dec_one_by_one=False,
        features="MS",
        **kwargs
    ):
        super(Informer, self).__init__()
        self.c_out = c_out
        self.label_len = label_len
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.dec_one_by_one = dec_one_by_one
        self.features = features

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dataset_flag, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dataset_flag, dropout)
        # Attention
        Attn = ProbAttention if attn == "prob" else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: (batch_size, seq_len, enc_in)
        # x_mark_enc: (batch_size, seq_len, time_num) time_num时间特征的维度数，如月、日、周几、小时
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # enc_out: (batch_size, seq_len, d_model) d_model: dimension of model
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        #  enc_out: (batch_size, enc_out_len, d_model) enc_out_len跟Informer的Encoder结构有关

        if not self.training and self.dec_one_by_one:
            # x_enc: (batch_size, seq_len, enc_in), 如果是预测单变量，则预测的是最后一个变量
            x_dec = x_enc[:, [-1], :]
            if self.c_out == 1:
                x_dec = x_dec[:, :, [-1]]
            out = []
            for i in range(self.pred_len):
                # x_mark_dec: (batch_size, label_len + pred_len, time_num)
                dec_out = self.dec_embedding(x_dec, x_mark_dec[:, [self.label_len - 1 + i], :])
                dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
                x_dec = self.projection(dec_out)
                out.append(x_dec)
            dec_result = torch.concat(out, dim=1)
            if self.output_attention:
                return dec_result, attns
            else:
                return dec_result  # [B, L, D]

        if self.training and self.dec_one_by_one and self.features == "MS":
            x_dec = x_dec[:, :, [-1]]

        # x_dec: (batch_size, label_len + pred_len, dec_in)
        # x_mark_dec: (batch_size, label_len + pred_len, time_num)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out: (batch_size, label_len + pred_len, d_model) d_model: dimension of model
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        # dec_out: (batch_size, label_len + pred_len, d_model)
        dec_out = self.projection(dec_out)
        # dec_out: (batch_size, label_len + pred_len, c_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=[3, 2, 1],
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        dataset_flag="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
        device=torch.device("cuda:0"),
        dec_one_by_one=False,
        features="MS",
        inp_lens=[0, 1, 2],
    ):
        super(InformerStack, self).__init__()
        self.c_out = c_out
        self.label_len = label_len
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.dec_one_by_one = dec_one_by_one

        self.features = features

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dataset_flag, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dataset_flag, dropout)
        # Attention
        Attn = ProbAttention if attn == "prob" else FullAttention
        # Encoder

        # inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model,
                            n_heads,
                            mix=False,
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                    )
                    for l in range(el)
                ],
                [ConvLayer(d_model) for l in range(el - 1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model),
            )
            for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, attention_dropout=dropout, output_attention=False), d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        if not self.training and self.dec_one_by_one:
            # x_enc: (batch_size, seq_len, enc_in), 如果是预测单变量，则预测的是最后一个变量
            x_dec = x_enc[:, [-1], :]
            if self.c_out == 1:
                x_dec = x_dec[:, :, [-1]]
            out = []
            for i in range(self.pred_len):
                # x_mark_dec: (batch_size, label_len + pred_len, time_num)
                dec_out = self.dec_embedding(x_dec, x_mark_dec[:, [self.label_len - 1 + i], :])
                dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
                x_dec = self.projection(dec_out)
                out.append(x_dec)
            dec_result = torch.concat(out, dim=1)
            if self.output_attention:
                return dec_result, attns
            else:
                return dec_result  # [B, L, D]

        if self.training and self.dec_one_by_one and self.features == "MS":
            x_dec = x_dec[:, :, [-1]]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
