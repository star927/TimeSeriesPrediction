import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, dec_kv=None,x_mask=None, cross_mask=None):
        if dec_kv is None:
            x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        else:
            x = x + self.dropout(self.self_attention(x, dec_kv, dec_kv, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.dec_kv = [None] * len(self.layers)

    def forward(self, x, cross, x_mask=None, cross_mask=None, transformer_dec=False):
        for i, layer in enumerate(self.layers):
            if not transformer_dec:
                x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            else:
                if self.dec_kv[i] is None:
                    self.dec_kv[i] = x
                else:
                    self.dec_kv[i] = torch.cat([self.dec_kv[i], x], dim=1)
                x = layer(x, cross, self.dec_kv[i], x_mask=x_mask, cross_mask=cross_mask)


        if self.norm is not None:
            x = self.norm(x)

        return x
