"""
An old script for fitting the cost model for OPT in FlexGen.

Warning:
The script has not been cleaned for release.
It has been placed here for study purposes only. There is no promise of reproduction.
"""

import argparse
import math
import os
import pickle as pkl
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from experiments.run_exp import ExpConfig, cases, get_filename
from flexgen.opt_config import get_opt_config
from flexgen.utils import GB, T

class CostModel(nn.Module):
    def __init__(self):
        super(CostModel, self).__init__()
        a = torch.abs(torch.rand([]))
        b = torch.abs(torch.rand([]))

        self.ctog_bdw        = nn.Parameter(a)
        self.gtoc_bdw_cache  = nn.Parameter(a)
        self.gtoc_bdw_hidden = nn.Parameter(a)

        self.dtoc_bdw          = nn.Parameter(a)
        self.ctod_bdw_cache_p  = nn.Parameter(a)
        self.ctod_bdw_hidden_p = nn.Parameter(a)
        self.ctod_bdw_g        = nn.Parameter(a)

        self.mm_flops_p  = nn.Parameter(b)
        self.mm_flops_g  = nn.Parameter(b)
        self.bmm_flops_p = nn.Parameter(b)
        self.bmm_flops_g = nn.Parameter(b)
        self.cpu_flops   = nn.Parameter(b)

        self.c0 = nn.Parameter(torch.tensor(0.0))
        self.c1 = nn.Parameter(torch.tensor(0.0))
        self.c2 = nn.Parameter(torch.tensor(0.0))
        self.c3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, xs):
        (wi, l, h1, h2, s, n,
         gbs, bls, wg, wc, wn, cg, cc, cn, hg, hc, hn) = xs.split(1, dim=1)

        ctogp = (self.ctog_bdw / GB) * (wi * (wc + wn)
                 + 2 * s * h1 * bls * (hc + hn))
        gtocp = (self.gtoc_bdw_cache / GB) * (4 * (s + 1) * h1 * bls * (cc + cn)) \
                 + (self.gtoc_bdw_hidden / GB) * 2 * s * h1 * bls * (hc + hn)
        dtocp = (self.dtoc_bdw / GB) * (wi * wn + 2 * s * h1 * bls * hn)
        ctodp = (self.ctod_bdw_cache_p / GB) * 4 * bls * (s + 1) * h1 * cn \
                  + (self.ctod_bdw_hidden_p / GB) * 2 * s * h1 * bls * hn
        compp = (self.mm_flops_p / T) * bls * (8 * s * h1 ** 2  + 4 * s * h1 * h2) \
                  + (self.bmm_flops_p / T) * 4 * bls * s ** 2 * h1
        tpre = torch.maximum(ctogp + dtocp, torch.maximum(gtocp + ctodp, compp))

        ctogg = (self.ctog_bdw / GB) * (wi * (wc + wn)
                  + 2 * h1 * bls * (hc + hn))
        gtocg = (self.gtoc_bdw_hidden / GB) * 2 * h1 * bls * (hc + hn)
        dtocg = (self.dtoc_bdw / GB) * (4 * bls * (s + n / 2) * h1 * cn
                                        + 2 * h1 * bls * hn) \
                  + (self.dtoc_bdw / GB / 0.95) * wi * wn 
        ctodg = (self.ctod_bdw_g / GB) * (4 * bls * h1 * cn
                                               + 2 * h1 * bls * hn)


        # non-linear cpu_flops
        cpu_flops_real = self.cpu_flops / torch.clamp(
            1 + self.c1 * torch.log2(64 / gbs).clamp(min=0) * torch.log2(4096 / h1).clamp(min=0)
            - self.c2 * torch.log2(64 / gbs).clamp(min=0)
            - self.c3 * torch.log2(4096 / h1).clamp(min=0),
            min=0.5)
        compg = (self.mm_flops_g / T) * bls * (8 * h1 ** 2 + 4 * h1 * h2) \
             + (self.bmm_flops_g / T) * 4 * bls * (s + n / 2) * h1 * cg \
             + (cpu_flops_real / T) * 4 * bls * (s + n / 2) * h1 * (cc + cn)

        #cpu_time_delta = (
        #    self.c0 +
        #    self.c1 * torch.log2(torch.clamp(gbs, max=64)) +
        #    self.c2 * torch.log2(torch.clamp(h1, max=4096)) +
        #    self.c3 * torch.log2(torch.clamp(gbs, max=64)) * torch.log2(torch.clamp(h1, max=4096))
        #)
        #compg = (self.mm_flops_g / T) * bls * (8 * h1 ** 2 + 4 * h1 * h2) \
        #     + (self.bmm_flops_g / T) * 4 * bls * (s + n / 2) * h1 * cg \
        #     + (self.cpu_flops / T) * 4 * bls * (s + n / 2) * h1 * (cc + cn) * (1 + cpu_time_delta) \

        tgen = ctogg + torch.maximum(gtocg,
                                     torch.maximum(dtocg,
                                     torch.maximum(ctodg, compg)))
        return torch.cat([tpre * l, tgen * (n - 1) * l], dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", type=str, default="../experiments/results")
    args = parser.parse_args()
    torch.manual_seed(0)

    model = CostModel()
    model.double()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-3)

    dataset = []
    for case in cases:
        pkl_file = os.path.join(args.pkl_dir, get_filename(case) + ".pkl")
        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as f:
                stats = pkl.load(f)
            s = case.prompt_len
            n = case.gen_len
            opt_config = get_opt_config(case.model_name)
            l = opt_config.num_hidden_layers
            h1 = opt_config.hidden_size
            h2 = opt_config.ffn_embed_dim
            wi = 8 * h1 ** 2 + 4 * h1 * h2
            gbs = case.gbs
            bls = case.bls
            wg, wc, cg, cc, hg, hc = case.percent
            wn = 100 - wg - wc
            cn = 100 - cg - cc
            hn = 100 - hg - hc
            wg, wc, wn, cg, cc, cn, hg, hc, hn = (
                 wg / 100, wc / 100, wn / 100, cg / 100, cc / 100,
                 cn / 100, hg / 100, hc / 100, hn  / 100)
            x = torch.tensor([[wi, l, h1, h2, s, n,
                             gbs, bls, wg, wc, wn, cg, cc, cn, hg, hc, hn]])
            y = torch.tensor([[stats.prefill_latency, stats.decode_latency]])
            dataset.append((x, y))

    xs = torch.cat([row[0] for row in dataset])
    ys = torch.cat([row[1] for row in dataset])

    indices = torch.randperm(xs.shape[0])
    xs = xs[indices]
    ys = ys[indices]
    split = int(0.9 * len(xs))

    xs_train, xs_test = xs[:split], xs[split:]
    ys_train, ys_test = ys[:split], ys[split:]

    def compute_loss(xs, ys):
        ys_pred = model(xs)
        return loss_fn(ys_pred / ys, torch.ones_like(ys))

    num_epoches = 30000

    def set_update_cpu_delta(flag):
        model.c0.requires_grad = flag
        model.c1.requires_grad = flag
        model.c2.requires_grad = flag
        model.c3.requires_grad = flag

    def freeze_all_params():
        for param in model.parameters():
            param.requires_grad = False

    set_update_cpu_delta(False)

    for i in range(num_epoches):
        reg_loss = compute_loss(xs_train, ys_train)
        penalty_loss = 0
        for name, p in model.named_parameters():
            penalty_loss += F.relu(-p)
        penalty_loss += F.relu(model.gtoc_bdw_hidden - model.gtoc_bdw_cache)
        penalty_loss += F.relu(model.mm_flops_p - model.bmm_flops_p)
        penalty_loss += F.relu(model.mm_flops_g - model.bmm_flops_g)
        penalty_loss += F.relu(model.bmm_flops_p - model.bmm_flops_g)
        penalty_loss += F.relu(model.mm_flops_p - model.mm_flops_g)

        loss = reg_loss + penalty_loss * 100
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == int(num_epoches * 0.8):
            freeze_all_params()
            set_update_cpu_delta(True)
            optimizer.param_groups[0]['lr'] = 1e-3

        if i % 200 == 0:
            eval_loss = compute_loss(xs_train, ys_train)
            print(f"epoch: {i}, train_loss: {loss.item():.6f}, "
                  f"eval_loss: {eval_loss.item():.6f}")

        #for name, p in model.named_parameters():
            #print(name, p.grad)

    for name, param in model.named_parameters():
        if "bdw" in name:
            print(f"{name}:\t {1 / param.item():.4f} GB/s")
        elif "flops" in name:
            print(f"{name}:\t {1 / param.item():.4f} T")
        elif "c" in name:
            print(f"{name}:\t {param.item():.4f}")

    print(f"len(dataset): {len(dataset)}")
