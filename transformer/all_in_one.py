from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from x_transformers import ContinuousTransformerWrapper, Decoder, ContinuousAutoregressiveWrapper

from embedding.backbone.restnet_backbone import ResnetBackbone
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from einops import rearrange, repeat

class AllInOne(nn.Module):
    def __init__(self, embed_dim, transformer_cfg, embedder_cfg):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(
            dim_in=embed_dim,
            dim_out=embed_dim,
            max_seq_len=transformer_cfg["ctx"],
            attn_layers=Decoder(
                dim=transformer_cfg["decoder_dim"],
                depth=transformer_cfg["depth"],
                heads=transformer_cfg["heads"],
                macaron=transformer_cfg["macaron"],
                shift_tokens=transformer_cfg["shift_tokens"],
                ff_dropout=transformer_cfg["ff_dropout"],
                attn_dropout=transformer_cfg["attn_dropout"],
            ),
        )
        self.autoencoder = LandauLifshitzGilbertEmbedding(
            embedder_cfg,
        )
        self.max_seq_len = transformer_cfg["ctx"]

    def embed(self, x, field):
        b, t, c, w, h = x.shape
        f = field.unsqueeze(1)
        f = f.repeat((1,t,1))
        # collapse batch and time
        x_collapsed = rearrange(x, 'b t c w h -> (b t) c w h')
        f_collapsed = rearrange(f, 'b t f -> (b t) f')
        x_h_collapsed = self.autoencoder.embed(x_collapsed, f_collapsed)
        # back to batch and time separate
        x_h = rearrange(x_h_collapsed, '(b t) g -> b t g', b=b, t=t)
        return x_h

    def recover(self, x_h):
        b, t, g = x_h.shape
        # collapse batch and time
        x_h_collapsed = rearrange(x_h, 'b t g -> (b t) g')
        x_collapsed = self.autoencoder.recover(x_h_collapsed)
        # back to batch and time separate
        x = rearrange(x_collapsed, '(b t) c h w -> b t c h w', b=b, t=t)
        return x
        
    def forward(self, x, field):
        x_h = self.embed(x, field)
        x_r = self.recover(x_h)
        # xi = x[:,:-1]
        # xo = x[:,1:]
        # xi_r = x_r[:,:-1]
        xi_h = x_h[:,:-1]
        xo_h = x_h[:,1:]
        # xo_r = x_r[:,1:]
        xo_ht = self.transformer(xi_h)

        # return xi, xi_r, xo, xo_r, xo_h, xo_ht
        return x, x_r, xo_h, xo_ht

    def generate(self, x, field, seq_len):
        b, t, c, w, h = x.shape

        self.transformer.eval()
        self.autoencoder.eval()

        with torch.no_grad():
            out_h = self.embed(x, field)
            for _ in range(seq_len):
                in_h = out_h[:, -self.max_seq_len:]

                last = self.transformer(in_h)[:, -1:, :]
                out_h = torch.cat((out_h, last), dim = -2)

        out = self.recover(out_h[:,t:])
        # mse = torch.nn.MSELoss()
        # diffs = []
        # l = np.arange(2,len(out_h[0]) + 1)
        # for i in range(1,len(out_h[0])):
        #     diffs.append(mse(out_h[0,i-1],out_h[0,i]).item())
        # # plt.plot([24,24],[min(diffs),max(diffs)])
        # fig, ax = plt.subplots()
        # # ax.annotate('Comparing initial embedded ground truth s-state representation with first predicted embedded state', 
        # #     xy=(2, diffs[0]*1.2), 
        # #     xytext=(2, diffs[0]*4), 
        # #     arrowprops = dict(facecolor='black', shrink=0.05))
        # # ax.annotate('Initial ground truth magnetization leaves context window', 
        # #     xy=(25, diffs[23]*1.2), 
        # #     xytext=(25, diffs[23]*4), 
        # #     arrowprops = dict(facecolor='black', shrink=0.05))
        # ax.plot(l[0:1],diffs[0:1],'rx')
        # ax.plot(l[1:24],diffs[1:24],'gx')
        # ax.plot(l[24:],diffs[24:],'bx')
        # ax.grid()
        # plt.title('MSE between current and previous embedded representation')
        # plt.yscale('log')
        # plt.show()
        return out