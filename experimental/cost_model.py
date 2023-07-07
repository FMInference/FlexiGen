"""
Cost Model for OPT in FlexGen.

Dependencies:
pip install pulp

Example Usages:
1. Find a policy:
python cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 32 \
                     --gpu-mem 16 --cpu-mem 200 --nvme-mem 1500
2. Estimate the throughput for a given policy:
python cost_model.py --model facebook/opt-30b --prompt-len 512 --gen-len 32 \
                     --gpu-mem 16 --cpu-mem 200 --nvme-mem 1500 \
                     --gpu-batch-size 48 --num-gpu-batches 3 \
                     --percent 20 80 0 100 0 100 \
                     --alpha-g 1.2 --alpha-c 1.2 --alpha-n 1.2

Note:
1. You need to fit the hardware constants for your device,
   see class CostModelConfig.
   (We fit them using gradient descent by collecting real run data points.
    Profiling for primitive modules or take numbers from the internet will not be accurate.)
2. Adjust relaxation ratio alpha_g, alpha_c, and alpha_n carefully,
   a smaller ratio cause a conservative policy,
   and a larger ratio cause an aggresive policy.
3. The cost model does not consider CPU compute delegation,
   and the support for quantization is incomplete.
4. In the second use case of estimating throughput for a fixed policy,
   relax the constraints alpha_g, alpha_c, and alpha_n to allow imprecise peak memory estimation.
"""

import argparse
import dataclasses
import math
import numpy as np
import pulp

from flexgen.compression import CompressionConfig
from flexgen.opt_config import get_opt_config
from flexgen.flex_opt import Policy
from flexgen.utils import GB, T

alpha_g = 0.8
alpha_c = 0.8
alpha_n = 0.8


@dataclasses.dataclass
class CostModelConfig:
    s: int = 512
    n: int = 32

    l: int = 96
    h1: int = 12288
    h2: int = 12288 * 4
    nh: int = 96

    gmem: int = 15 * GB
    cmem: int = 204 * GB
    nmem: int = 1500 * GB

    # hardware constants
    # default value aligned on google cloud T4
    ctog_bdw: float = 12.89 * GB
    gtoc_bdw_cache: float = 0.97 * GB
    gtoc_bdw_hidden: float = 4.82 * GB

    dtoc_bdw: float = 0.473 * GB
    ctod_bdw_cache_p: float = 0.746 * GB
    ctod_bdw_hidden_p: float = 2.015 * GB
    ctod_bdw_g: float = 2.015 * GB

    mm_flops_p: float = 21.24 * T
    mm_flops_g: float = 4.3 * T
    bmm_flops_p: float = 9.97 * T
    bmm_flops_g: float = 0.079 * T
    cpu_flops: float = 0.0123 * T

    c1: float = 0.0168
    c2: float = 0.0328
    c3: float = 0.0621


def solve_lp(config, bls, gbs, compress_w=False, verbose=1, debug=False, percent=None):
    assert bls > 0 and gbs > 0
    assert bls >= gbs and bls % gbs == 0

    ## Constants
    s = config.s
    n = config.n
    l = config.l
    h1 = config.h1
    h2 = config.h2
    nh = config.nh

    gmem = config.gmem * alpha_g
    cmem = config.cmem * alpha_c
    nmem = config.nmem * alpha_n

    ctog_bdw = config.ctog_bdw
    gtoc_bdw_cache = config.gtoc_bdw_cache
    gtoc_bdw_hidden = config.gtoc_bdw_hidden
    dtoc_bdw = config.dtoc_bdw

    ctod_bdw_cache_p = config.ctod_bdw_cache_p
    ctod_bdw_hidden_p = config.ctod_bdw_hidden_p
    ctod_bdw_g = config.ctod_bdw_g

    mm_flops_p = config.mm_flops_p
    mm_flops_g = config.mm_flops_g
    bmm_flops_p = config.bmm_flops_p
    bmm_flops_g = config.bmm_flops_g
    cpu_flops = config.cpu_flops

    c1 = config.c1
    c2 = config.c2
    c3 = config.c3

    ## Create Problem
    prob = pulp.LpProblem('storage', sense=pulp.LpMinimize)

    ## Create variables for cost
    T = pulp.LpVariable("T", lowBound=0)
    Tpre = pulp.LpVariable("Tpre_i", lowBound=0)
    Tgen = pulp.LpVariable("Tgen_i", lowBound=0)
    ctogp = pulp.LpVariable("ctog_i^p", lowBound=0)
    gtocp = pulp.LpVariable("gtoc_i^p", lowBound=0)
    ctodp = pulp.LpVariable("ctod_i^p", lowBound=0)
    dtocp = pulp.LpVariable("dtoc_i^p", lowBound=0)
    compp = pulp.LpVariable("comp_i^p", lowBound=0)
    ctogg = pulp.LpVariable("ctog_i^g", lowBound=0)
    gtocg = pulp.LpVariable("gtoc_i^g", lowBound=0)
    ctodg = pulp.LpVariable("ctod_i^g", lowBound=0)
    dtocg = pulp.LpVariable("dtoc_i^g", lowBound=0)
    compg = pulp.LpVariable("comp_i^g", lowBound=0)

    wg = pulp.LpVariable("wg", lowBound=0)
    wc = pulp.LpVariable("wc", lowBound=0)
    wn = pulp.LpVariable("wn", lowBound=0)
    cg = pulp.LpVariable("cg", lowBound=0)
    cc = pulp.LpVariable("cc", lowBound=0)
    cn = pulp.LpVariable("cn", lowBound=0)
    hg = pulp.LpVariable("hg", lowBound=0)
    hc = pulp.LpVariable("hc", lowBound=0)
    hn = pulp.LpVariable("hn", lowBound=0)


    ## Set objective

    # Minimize T/bls
    prob += T * (1 / bls)

    # layer weight size
    wi = 8 * h1 ** 2 + 4 * h1 * h2
    if compress_w:
        wi = wi / 4

    if debug:
        ## DEBUG
        assert percent is not None
        if percent[0] is not None:
            prob += wg == percent[0] / 100
        if percent[1] is not None:
            prob += wc == percent[1] / 100
        if percent[2] is not None:
            prob += cg == percent[2] / 100
        if percent[3] is not None:
            prob += cc == percent[3] / 100
        if percent[4] is not None:
            prob += hg == percent[4] / 100
        if percent[5] is not None:
            prob += hc == percent[5] / 100

    # --------------- Add constraints -------------------

    prob += wg + wc + wn == 1
    prob += cg + cc + cn == 1
    prob += hg + hc + hn == 1
    ## temporay hack, as the current runtime does not support hidden on disk
    prob += hg + hc == 1

    prob += T == Tpre * l + Tgen * (n-1) * l

#    # Tpre = max(ctogp, gtocp, dtocp, ctodp, compp)
#    prob += Tpre >= ctogp
#    prob += Tpre >= gtocp
#    prob += Tpre >= dtocp
#    prob += Tpre >= ctodp
#    prob += Tpre >= compp

    # Tpre = max(ctogp + dtocp, gtocp + ctodp, compp)
    prob += Tpre >= ctogp + dtocp
    prob += Tpre >= gtocp + ctodp
    prob += Tpre >= compp

    # ctogp = (weight_ctogp + hidden_ctogp) / ctog_bdw
    prob += ctogp == (1 / ctog_bdw) * (wi * (wc + wn)
                     + 2 * s * h1 * bls * (hc + hn))

    # gtocp = (cache_gtocp + hidden_gtocp) / gtoc_bdw
    prob += gtocp == (1 / gtoc_bdw_cache) * (4 * (s + 1) * h1 * bls * (cc + cn)) \
                   + (1 / gtoc_bdw_hidden) * 2 * s * h1 * bls * (hc + hn)

    # dtocp = (weight_dtocp + hidden_dtocp) / dtoc_bdw
    prob += dtocp == (1 / dtoc_bdw) * (wi * wn + 2 * s * h1 * bls * hn)

    # ctodp = (cache_ctodp + hidden_ctodp) / ctod_bdw
    prob += ctodp == (1 / ctod_bdw_cache_p) * 4 * bls * (s + 1) * h1 * cn \
                   + (1 / ctod_bdw_hidden_p) * 2 * s * h1 * bls * hn

    # compp = gpu_compp
    prob += compp == (1 / mm_flops_p) * bls * (8 * s * h1 ** 2  + 4 * s * h1 * h2) \
                     + (1 / bmm_flops_p) * 4 * bls * s ** 2 * h1

    # # Tgen = max(ctogg, gtocg, dtocg, ctodg, compg)
    # prob += Tgen >= ctogg
    # prob += Tgen >= gtocg
    # prob += Tgen >= dtocg
    # prob += Tgen >= ctodg
    # prob += Tgen >= compg

    # Tgen = ctogg + max(gtocg, dtocg, ctodg, compg)
    prob += Tgen >= gtocg + ctogg
    prob += Tgen >= dtocg + ctogg
    prob += Tgen >= ctodg + ctogg
    prob += Tgen >= compg + ctogg

    # ctogg = (weight_ctogg + hidden_ctogg) / ctog_bdw
    prob += ctogg == (1 / ctog_bdw) * (wi * (wc + wn)
                     + 2 * h1 * bls * (hc + hn))

    # gtocg = hidden_gtocg / gtoc_bdw
    prob += gtocg == (1 / gtoc_bdw_hidden) * 2 * h1 * bls * (hc + hn)

    # dtocg = (cache_dtocg + weight_dtocg + hidden_dtocg) / dtoc_bdw
    prob += dtocg == (1 / dtoc_bdw) * (4 * bls * (s + n / 2) * h1 * cn
                                       + 2 * h1 * bls * hn) \
                     + (1 / (dtoc_bdw * 0.95)) * wi * wn 

    # ctodg = (cache_ctodg + hidden_ctodg) / ctod_bdw
    prob += ctodg == (1 / ctod_bdw_g) * (4 * bls * h1 * cn
                     + 2 * h1 * bls * hn)

    # compg = gpu_compg + cpu_compg
    # non-linear cpu_flops
    cpu_flops_real = cpu_flops * np.maximum(0.1,
            1 + c1 * (max(0, math.log2(64 / gbs)) * max(0, math.log2(4096 / h1)))
            - c2 * max(0, math.log2(64 / gbs))
            - c3 * max(0, math.log2(4096 / h1)))
    prob += compg == (1 / mm_flops_g) * bls * (8 * h1 ** 2  + 4 * h1 * h2) \
                     + (1 / bmm_flops_g) * 4 * bls * (s + n / 2) * h1 * cg \
                     + (1 / cpu_flops_real) * 4 * bls * (s + n / 2) * h1 * (cc + cn)

    ## Create variables for peak memory constraints
    gpu_home_p = pulp.LpVariable("gpu_home^p", lowBound=0)
    gpu_w_p = pulp.LpVariable("gpu_w^p", lowBound=0)
    gpu_home_g = pulp.LpVariable("gpu_home^g", lowBound=0)
    gpu_w_g = pulp.LpVariable("gpu_w^g", lowBound=0)

    interp = pulp.LpVariable("inter_gpu_working_p", lowBound=0)
    qkvp = pulp.LpVariable("qkvp", lowBound=0)
    att1p = pulp.LpVariable("att1p", lowBound=0)
    att2p = pulp.LpVariable("att2p", lowBound=0)
    outputp = pulp.LpVariable("outputp", lowBound=0)
    mlp1p = pulp.LpVariable("mlp1p", lowBound=0)
    mlp2p = pulp.LpVariable("mlp2p", lowBound=0)

    interg = pulp.LpVariable("inter_gpu_working_g", lowBound=0)
    qkvg = pulp.LpVariable("qkvg", lowBound=0)
    att1g = pulp.LpVariable("att1g", lowBound=0)
    att2g = pulp.LpVariable("att2g", lowBound=0)
    outputg = pulp.LpVariable("outputg", lowBound=0)
    mlp1g = pulp.LpVariable("mlp1g", lowBound=0)
    mlp2g = pulp.LpVariable("mlp2g", lowBound=0)

    cpu_home_p = pulp.LpVariable("cpu_home^p", lowBound=0)
    cpu_w_p = pulp.LpVariable("cpu_w^p", lowBound=0)
    cpu_home_g = pulp.LpVariable("cpu_home^g", lowBound=0)
    cpu_w_g = pulp.LpVariable("cpu_w^g", lowBound=0)

    nvme_peak = pulp.LpVariable("nvme_peak", lowBound=0)
    
    ## GPU peak memory constaints
    prob += gpu_home_p == wi * l * wg + 2 * s * h1 * bls * hg + 4 * (s + n) * h1 * bls * l * cg
    prob += interp == 8 * gbs * s * h1 \
                    + gbs * (2 * s * h1 + 2 * nh * s ** 2) \
                    + gbs * (2 * s * h1) \
                    + 4 * gbs * s * h1 \
                    + 2 * gbs * s * h2 \
                    + 2 * gbs * s * h1
    prob += gpu_w_p == 2 * wi * (1 - wg) + 2 * s * h1 * gbs * (1 - hg) \
                     + interp
    # prob += (gpu_home_p * 1.2 + gpu_w_p) + 0.5 * GB <= gmem
    prob += gpu_home_p + gpu_w_p <= gmem

    # prob += gpu_home_p == wi * l * wg + 2 * s * h1 * bls * hg + 4 * (s + n) * h1 * bls * l * cg
    # prob += gpu_w_p == 2 * wi * (1 - wg) + 2 * s * h1 * gbs * (1 - hg) + interp
    # prob += interp >= qkvp
    # prob += interp >= att1p
    # prob += interp >= att2p
    # prob += interp >= outputp
    # prob += interp >= mlp1p
    # prob += interp >= mlp2p
    # prob += qkvp == 8 * gbs * s * h1
    # prob += att1p == gbs * (2 * s * h1 + 2 * s * h1 + 2 * nh * s ** 2)
    # prob += att2p == gbs * (2 * nh * s ** 2 + 2 * s * h1 + 2 * s * h1)
    # prob += outputp == 4 * gbs * s * h1
    # prob += mlp1p == 2 * gbs * s * (h1 + h2)
    # prob += mlp2p == 2 * gbs * s * (h2 + h1)
    # prob += gpu_home_p + gpu_w_p <= gmem

    prob += gpu_home_g == wi * l * wg + 2 * h1 * bls * hg + 4 * (s + n) * h1 * bls * l * cg
    prob += interg == 8 * gbs * h1 \
                    + gbs * (2 * h1 + 2 * (s + n) * h1 + 2 * nh * (s + n)) * cg \
                    + gbs * (2 * (s + n) * h1 + 2 * h1) * cg \
                    + 4 * gbs * h1 \
                    + 2 * gbs * h2 \
                    + 2 * gbs * h1
    prob += gpu_w_g == 2 * wi * (1 - wg) + 2 * h1 * gbs * (1 - hg) \
                     + 2 * 2 * gbs * (s + n) * h1 * cg \
                     + interg
    # prob += (gpu_home_g * 1.2 + gpu_w_g) + 0.5 * GB <= gmem
    prob += gpu_home_g + gpu_w_g <= gmem

    # prob += gpu_home_g == wi * l * wg + 2 * h1 * bls * hg + 4 * (s + n) * h1 * bls * l * cg
    # prob += gpu_w_g == 2 * wi * (1 - wg) + 2 * s * h1 * gbs * (1 - hg) + interg
    # prob += interg >= qkvg
    # prob += interg >= att1g
    # prob += interg >= att2g
    # prob += interg >= outputg
    # prob += interg >= mlp1g
    # prob += interg >= mlp2g
    # prob += qkvg == 8 * gbs * h1
    # prob += att1g == gbs * (2 * h1 + 2 * (s + n) * h1 + 2 * nh * (s + n)) * cg
    # prob += att2g == gbs * (2 * nh * (s + n) + 2 * (s + n) * h1 + 2 * h1) * cg
    # prob += outputg == 4 * gbs * h1
    # prob += mlp1g == 2 * gbs * (h1 + h2)
    # prob += mlp2g == 2 * gbs * (h2 + h1)
    # prob += gpu_home_g + gpu_w_g <= gmem

    ## CPU peak memory constraints
    prob += cpu_home_p == wi * l * wc + 2 * s * h1 * bls * hc + 4 * s * h1 * bls * l * cc
    prob += cpu_w_p == wi * (1 - wg) + 2 * s * h1 * gbs * (1 - hg)
    prob += cpu_home_p + cpu_w_p <= cmem

    prob += cpu_home_g == wi * l * wc + 2 * h1 * bls * hc + 4 * (s + n) * h1 * bls * l * cc
    prob += cpu_w_g == wi * wn + 4 * h1 * gbs * hn + 8 * (s + n) * h1 * gbs * cn + 2 * nh * (s + n) * gbs + 2 * h1 * gbs
    prob += cpu_home_g + cpu_w_g <= cmem

    ## NVMe peak memory constraints
    prob += nvme_peak == wi * l * wn + 2 * s * h1 * bls * hn + 4 * (s + n) * h1 * bls  * l * cn
    prob += nvme_peak <= nmem

    # ------------ Finish add constraints ---------------

    ## Optimize model
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if status == -1:
        return status, None, (0, -1, -1), None

    # gpu_peak_p = pulp.value(gpu_home_p) * 1.2 + pulp.value(gpu_w_p) + 0.5 * GB
    # gpu_peak_g = pulp.value(gpu_home_g) * 1.2 + pulp.value(gpu_w_g) + 0.5 * GB

    gpu_peak_p = pulp.value(gpu_home_p) + pulp.value(gpu_w_p)
    gpu_peak_g = pulp.value(gpu_home_g) + pulp.value(gpu_w_g)

    cpu_peak_p = pulp.value(cpu_home_p) + pulp.value(cpu_w_p)
    cpu_peak_g = pulp.value(cpu_home_g) + pulp.value(cpu_w_g)
 
    throughput = bls * n / pulp.value(T)

    tpre = pulp.value(Tpre)
    tpre_tot = tpre * l

    tgen = pulp.value(Tgen)
    tgen_tot = tgen * (n-1) * l

    ## print solution
    if verbose:
        print(f"status: {status}")
        print(f"weights size: {wi * l / GB:.4f} GB")
        print(f"ctogp = {pulp.value(ctogp):.4f} s  "
              f"gtocp = {pulp.value(gtocp):.4f} s  "
              f"dtocp = {pulp.value(dtocp):.4f} s  "
              f"ctodp = {pulp.value(ctodp):.4f} s  "
              f"compp = {pulp.value(compp):.4f} s")
        print(f"Tpre = {pulp.value(Tpre):.3f} s")
        print(f"Tpre * l: {tpre:.4f} * {l} = {tpre_tot:.4f}")
        print(f"ctogg = {pulp.value(ctogg):.4f} s  "
              f"gtocg = {pulp.value(gtocg):.4f} s  "
              f"dtocg = {pulp.value(dtocg):.4f} s  "
              f"ctodg = {pulp.value(ctodg):.4f} s  "
              f"compg = {pulp.value(compg):.4f} s")
        print(f"cache dtocg: {4 * bls * (s + n / 2) * h1 * pulp.value(cn) / dtoc_bdw:.2f}  "
              f"weights dtocg: {wi * pulp.value(wn) / dtoc_bdw:.2f}")
        print(f"Tgen = {pulp.value(Tgen):.3f} s")
        print(f"Tgen * (n-1) * l: "
                f"{tgen:.4f} * {n-1} * {l} = {tgen_tot:.4f}")

        print(f"gpu peak mem (prefill): {gpu_peak_p / GB:.3f} GB / {gmem / alpha_g / GB:.3f} GB")
        print(f"gpu peak mem (gen):     {gpu_peak_g / GB:.3f} GB / {gmem / alpha_g / GB:.3f} GB")

        print(f"cpu peak mem (prefill): {cpu_peak_p / GB:.3f} GB / {cmem / alpha_c / GB:.3f} GB")
        print(f"cpu peak mem (gen):     {cpu_peak_g / GB:.3f} GB / {cmem / alpha_c / GB:.3f} GB")
        print(f"cpu_home_g: {pulp.value(cpu_home_g) / GB:.2f} GB")
        print(f"cpu_w_g: {pulp.value(cpu_w_g) / GB:.2f} GB")

        print(f"nvme peak mem:          {pulp.value(nvme_peak) / GB:.3f} GB / {nmem / alpha_n / GB:.3f} GB")

        print(f"wg = {pulp.value(wg):.2f}  "
              f"wc = {pulp.value(wc):.2f}  "
              f"wn = {pulp.value(wn):.2f}")
        print(f"cg = {pulp.value(cg):.2f}  "
              f"cc = {pulp.value(cc):.2f}  "
              f"cn = {pulp.value(cn):.2f}")
        print(f"hg = {pulp.value(hg):.2f}  "
              f"hc = {pulp.value(hc):.2f}  "
              f"hn = {pulp.value(hn):.2f}")
        print(f"T = {pulp.value(T)} s  "
              f"generated = {bls * n} tokens")
        print(f"throughput = {throughput:.2f} token/s")
    
    policy = Policy(gbs, bls // gbs,
                    pulp.value(wg), pulp.value(wc),
                    pulp.value(cg), pulp.value(cc),
                    pulp.value(hg), pulp.value(hc),
                    overlap=True, sep_layer=False, pin_weight=False,
                    cpu_cache_compute=True, attn_sparsity=1,
                    compress_weight=False, comp_weight_config=
                        CompressionConfig(num_bits=4, group_size=64,
                                          group_dim=0, symmetric=False),
                    compress_cache=False, comp_cache_config=
                        CompressionConfig(num_bits=4, group_size=64,
                                          group_dim=2, symmetric=False))
    return status, policy, (throughput, tpre_tot, tgen_tot), (gpu_peak_p, gpu_peak_g)


def get_nb_ub(config, gbs, solve_lp, compress_w=False, debug=None, percent=None):
    nb = 1
    while True:
        status, _, _, _ = solve_lp(config, gbs * nb, gbs, compress_w=compress_w, verbose=0, debug=debug, percent=percent)
        if status == -1: break
        nb *= 2

    left = max(nb // 2, 1)
    right = nb
    while left < right:
        mid = (left + right) // 2
        status, _, _, _ = solve_lp(config, gbs * mid, gbs, compress_w=compress_w, verbose=0, debug=debug, percent=percent)
        if status == 1:
            left = mid + 1
        elif status == -1:
            right = mid
    assert left == right
    nb_ub = left

    return nb_ub - 1


def best(policy1, throughput1, policy2, throughput2):
    if throughput2 > throughput1:
        return policy2, throughput2
    else:
        return policy1, throughput1


def solve(config, solve_lp, args):
    # # debug
    # code = []
    # policies = []
    # throughputs = []
    # for nb in range(1, 240, 5):
    #     gbs = 8
    #     bls = gbs * nb
    #     status, policy, throughput, _ = solve_lp(config, bls, gbs, verbose=0)
    #     code.append(status)
    #     policies.append(policy)
    #     throughputs.append(throughput)
    # print(code)
    # print(policies[-3:])
    # print(["{0:0.2f}".format(x[0]) for x in throughputs])

    debug = False
    if args["wg"] or args["wc"] or args["cg"] or \
       args["cc"] or args["hg"] or args["hc"]:
        debug = True
    percent = [args["wg"], args["wc"], args["cg"], args["cc"],
               args["hg"], args["hc"]]
    if args["percent"] is not None:
        debug = True
        percent = args["percent"]
    compress_w = args["compress_w"]

    best_policy = None
    max_throughput = 0
    gbs = 1
    while True:
        if args["gbs"] is not None:
            gbs = args["gbs"]
        if args["num_gb"] is not None:
            status, policy, (throughput, _, _), _ = solve_lp(
                config, gbs * args["num_gb"], gbs, compress_w=compress_w, verbose=0, debug=debug, percent=percent)
            if status == -1:
                break
            if status == 1:
                best_policy, max_throughput = best(best_policy, max_throughput, policy, throughput)
        else:
            nb_ub = get_nb_ub(config, gbs, solve_lp, compress_w=compress_w, debug=debug, percent=percent)
            if nb_ub == 0: break

            prev_throughput = 0
            for nb in range(1, nb_ub + 1):
                _, policy, (throughput, _, _), _ = solve_lp(config, gbs * nb, gbs, compress_w=compress_w, verbose=0, debug=debug, percent=percent)
                if throughput < prev_throughput:
                    break
                prev_throughput = throughput
                best_policy, max_throughput = best(best_policy, max_throughput, policy, throughput)
        if args["gbs"] is not None:
            break
        if gbs < 4:
            gbs += 1
        else:
            gbs *= 2

    if best_policy is not None:
        _, _, _, _ = solve_lp(config, best_policy.gpu_batch_size * best_policy.num_gpu_batches,
                           best_policy.gpu_batch_size, compress_w=compress_w, verbose=True, debug=debug, percent=percent)
    return best_policy, max_throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/opt-175b")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--gpu-mem", type=int, default=15)
    parser.add_argument("--cpu-mem", type=int, default=200)
    parser.add_argument("--nvme-mem", type=int, default=1500)
    
    parser.add_argument("--gbs", "--gpu-batch-size", type=int)
    parser.add_argument("--num-gb", "--num-gpu-batches", type=int)
    parser.add_argument("--percent", nargs="+", type=int)
    parser.add_argument("--wg", type=int)
    parser.add_argument("--wc", type=int)
    parser.add_argument("--cg", type=int)
    parser.add_argument("--cc", type=int)
    parser.add_argument("--hg", type=int)
    parser.add_argument("--hc", type=int)
    parser.add_argument("--compress-w", action="store_true")

    parser.add_argument("--alpha-g", type=float)
    parser.add_argument("--alpha-c", type=float)
    parser.add_argument("--alpha-n", type=float)
    args = parser.parse_args()
    assert not (args.percent and (args.wg or args.wc or args.cg or args.cc or args.hg or args.hc))

    if args.alpha_g:
        alpha_g = args.alpha_g
    if args.alpha_c:
        alpha_c = args.alpha_c
    if args.alpha_n:
        alpha_n = args.alpha_n

    config = CostModelConfig()

    opt_config = get_opt_config(args.model)
    config.l = opt_config.num_hidden_layers
    config.h1 = opt_config.hidden_size
    config.h2 = opt_config.ffn_embed_dim
    config.nh = opt_config.n_head

    config.s = args.prompt_len
    config.n = args.gen_len

    config.gmem = args.gpu_mem * GB
    config.cmem = args.cpu_mem * GB
    config.nmem = args.nvme_mem * GB

    best_policy, max_throughput = solve(config, solve_lp, vars(args))
    print(best_policy)
    print(f"max_throughput: {max_throughput:.2f} token/s")
 
