import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random
import csv
import os

from new_func.myfunc import *

def proposed(q1=150, q2=100, b=100, ga=-16, d=120, w=1.2, v=40, a_s=1.2, a_h=1.5, miu=2, turn=5000):

    Q_s = q1
    Q_h = q2
    B_s = b
    B_h = b
    U=[]
    Umean=[]
    bmean = []
    emean=[]
    Qmean_s = []
    Qmean_h = []
    Qver = []
    Qver1=[]
    Bmean_s = []
    Bmean_h = []
    t0mean=[]
    t1mean=[]
    t2mean=[]
    t3mean=[]
    t4mean=[]
    t5mean=[]
    t6mean=[]
    tao0_list = []
    # p1_list = []
    # p2_list = []
    # p3_list = []

    # Q_pic = [Q]
    # B1_pic = [0]
    # B2_pic = [0]
    # u_pic = []
    # umean_pic = []
    # d2_pic = []
    # d4_pic = []
    # t0_pic = []
    # t1_pic = []
    # t2_pic = []
    # dmean = 0
    # emean = 0

    # 设置参数
    # 节点之间距离
    dsa = 130
    dsh = d
    dha = 120

    T = 1
    P0 = 10
    Bmax = 200
    Pbach = Pbacs = 0.1
    Phtth = Phtts = 0.7
    Pmax = 1
    sigma2 = 10 ** (-3)  # 噪声功率
    W = w
    gap = 10 ** (ga / 10)
    phis = 490
    phih = 470
    fs = 500  # 计算频率
    fh = 480
    kai1 = 10 ** (-8)
    kai2 = kai1
    gsa = 3 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsa)) ** 3 * 10 ** 10  # 信道增益
    gsh = 2.8 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsh)) ** 3 * 10 ** 10
    gha = 2.5 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dha)) ** 3 * 10 ** 10
    miu1 = miu / (miu + 1)
    miu2 = 1 / (miu + 1)
    u = 0
    V = v
    # print(gsa, gsh, gha)

    # 新任务序列均值为3的指数分布
    A_s = produce_A(a_s)
    A_h = produce_A(a_h)
    # print(A)
    # A = np.ones(5000) * 2.58

    # 决策变量
    t0 = cp.Variable()
    t1 = cp.Variable()
    t2 = cp.Variable()
    t3 = cp.Variable()
    t4 = cp.Variable()
    t5 = cp.Variable()
    t6 = cp.Variable()
    ts = cp.Variable()
    th = cp.Variable()
    d1 = cp.Variable()
    d2 = cp.Variable()
    d3 = cp.Variable()
    d4 = cp.Variable()
    d5 = cp.Variable()
    d6 = cp.Variable()
    tao0 = cp.Variable()
    tao1 = cp.Variable()
    tao2 = cp.Variable()
    tao3 = cp.Variable()
    tao4 = cp.Variable()
    tao5 = cp.Variable()

    for i in range(turn):

        obj = cp.Minimize(-(Q_s + V) * fs * ts / phis - (Q_s + V * miu1) * (d1 + d2) - (Q_h + V) *
                          fh * th / phih - (Q_h + V * miu2) * (d5 + d6) +
                          (V * u - B_s + Bmax) * (Pbacs * t1 + Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) + (
                                  V * u - B_h + Bmax) * (
                                  Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th)
                          + ((B_s - Bmax) * gsa * P0 + (B_h - Bmax) * gha * P0) * (
                                   t0 + t1 + t3 + t5) - (B_s - Bmax) * gsa * P0 * tao0 - (B_h - Bmax) * gha * P0 * (
                                  tao1 + tao2))
        constraints = [
            t0 + t1 + t2 + t3 + t4 + t5 + t6 <= T,
            gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) - gsa * P0 * (
                    t0 + t1 + t3 + t5) <= B_s,
            gha * P0 * (tao1 + tao2) + (
                    Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th) - gha * P0 * (
                    t0 + t1 + t3 + t5) <= B_h,
            d1 <= - W * cp.rel_entr(t1 * sigma2, tao0 * gap * P0 * gsa * gsh + t1 * sigma2) / sigma2,
            d2 <= - W * cp.rel_entr(t2 * sigma2, tao3 * gsh + t2 * sigma2) / sigma2,
            d3 <= - W * cp.rel_entr(t3 * sigma2, tao1 * gap * P0 * gha * gha + t3 * sigma2) / sigma2,
            d4 <= - W * cp.rel_entr(t4 * sigma2, tao4 * gha + t4 * sigma2) / sigma2,
            d5 <= - W * cp.rel_entr(t5 * sigma2, tao2 * gap * P0 * gha * gha + t5 * sigma2) / sigma2,
            d6 <= - W * cp.rel_entr(t6 * sigma2, tao5 * gha + t6 * sigma2) / sigma2,
            d1 <= d3,
            d2 <= d4,
            fs * ts / phis + d1 + d2 <= Q_s,
            fh * th / phih + d5 + d6 <= Q_h,
            t0 >= 0,
            t1 >= 0,
            t2 >= 0,
            t3 >= 0,
            t4 >= 0,
            t5 >= 0,
            t6 >= 0,
            ts >= 0,
            th >= 0,
            ts <= T,
            th <= T,
            tao0 >= 0,
            tao1 >= 0,
            tao2 >= 0,
            tao3 >= 0,
            tao4 >= 0,
            tao5 >= 0,
            d1 >= 0,
            d2 >= 0,
            d3 >= 0,
            d4 >= 0,
            d5 >= 0,
            d6 >= 0,
            tao0 <= t1,
            tao1 <= t3,
            tao2 <= t5,
            tao3 <= t2 * Pmax,
            tao4 <= t4 * Pmax,
            tao5 <= t6 * Pmax,
        ]

        prob = cp.Problem(obj, constraints)
        # prob.solve(verbose=True)
        prob.solve()
        a=w*t1.value*math.log(1+(tao0.value * gap * P0 * gsa * gsh) /( t1.value * sigma2))
        b=w*t4.value*math.log(1+(tao4.value*gha)/(t4.value*sigma2))
        print("a=",a,"b=",b)
        # 更新u
        # print("value=", prob.value)
        # print(t0.value, t1.value, t2.value, tao1.value, tao2.value)

        # ds_off = t1.value * W * math.log(1 + tao1.value * gsh / t1.value / sigma2)
        # dh_off_s = t2.value * W * math.log(1 + tao2.value * gha / t2.value / sigma2)
        # dh_off_h = t3.value * W * math.log(1 + tao3.value * gha / t3.value / sigma2)
        ds_loc = fs * ts.value / phis
        dh_loc = fh * th.value / phih

        # a=B_h- gha * P0 * (tao1 + tao2) - (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5)
        #  +gsa * P0 * (t0 + t1 + t3 + t5)
        # print("a=",a.value)

        d_tot = ds_loc + miu1 * (d1.value + d2.value) + dh_loc + miu2 * (d5.value + d6.value)
        es_loc = kai1 * fs ** 3 * ts.value
        es_off = Pbacs * t1.value + Phtts * t2.value + tao3.value
        eh_loc = kai2 * fh ** 3 * th.value
        eh_off1 = Pbach * t3.value + Phtth * t4.value + tao4.value
        eh_off2 = Pbach * t5.value + Phtth * t6.value + tao5.value
        e_s = es_loc + es_off
        e_h = eh_loc + eh_off1 + eh_off2
        es_hv = (t0.value + t1.value + t3.value + t5.value - tao0.value) * P0 * gsa
        eh_hv = (t0.value + t1.value + t3.value + t5.value - tao1.value - tao2.value) * P0 * gha
        e_tot = e_s + e_h

        # 更新任务队列和电池队列
        Q_s = Q_s - d1.value - d2.value - ds_loc + A_s[i]
        Q_h = Q_h - d5.value - d6.value - dh_loc + A_h[i]
        B_s = B_s- gsa * P0 * tao0.value - (Pbacs * t1.value + Phtts * t2.value + tao3.value+kai1 * fs ** 3 * ts.value) + gsa * P0 * (t0.value + t1.value + t3.value + t5.value)
        if B_s <= Bmax:
           B_s=B_s
        else:
           B_s=Bmax
        B_h = B_h-gha * P0 * (tao1.value + tao2.value) - (Pbach * t3.value + Phtth * t4.value + tao4.value + Pbach * t5.value + Phtth * t6.value + tao5.value+ kai2 * fh ** 3 * th.value)+gha * P0 * (t0.value + t1.value + t3.value + t5.value)
        if B_h <= Bmax:
            B_h =B_h
        else:
           B_h=Bmax
        # gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3) - gsa * P0 * (t0 + t1 + t3 + t5) <= B_s,
        # gha * P0 * (tao1 + tao2) + (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5) - gsa * P0 * (
        #         t0 + t1 + t3 + t5) <= B_h,
        Beta1 = tao0.value / t1.value
        Beta2 = tao1.value / t3.value
        Beta3 = tao2.value / t5.value
        Ps = tao3.value / t2.value
        Ph1 = tao4.value / t4.value
        Ph2 = tao5.value / t6.value
        # 记录队列及目标函数
        # Q_pic.append(Q)
        # B1_pic.append(B1)
        # B2_pic.append(B2)
        # u_pic.append(u)
        tao0_list.append(tao0.value)
        if i == 0:
            bmean.append(d_tot)
            emean.append(e_tot)
        else:
            bmean.append(bmean[i - 1] * (i) / (i + 1) + d_tot / (i + 1))
            emean.append(emean[i - 1] * (i) / (i + 1) + e_tot / (i + 1))

        u =bmean[i] / emean[i]

        if i == 0:
            U.append(d_tot / e_tot)
            Umean.append(u)
            Qmean_s.append(Q_s)
            Qmean_h.append(Q_h)
            Bmean_h.append(B_h)
            Bmean_s.append(B_s)
        else:
            U.append(u)
            Umean.append(Umean[i - 1] * (i) / (i + 1) + u / (i + 1))
            Qmean_h.append(Qmean_h[i - 1] * (i) / (i + 1) + Q_h / (i + 1))
            Qmean_s.append(Qmean_s[i - 1] * (i) / (i + 1) + Q_s / (i + 1))
            Bmean_h.append(Bmean_h[i - 1] * (i) / (i + 1) + B_h / (i + 1))
            Bmean_s.append(Bmean_s[i - 1] * (i) / (i + 1) + B_s / (i + 1))

        if i == 0:
            Qver1.append((Q_s + Q_h)/2)
            Qver.append((Q_s + Q_h)/2)
            t0mean.append(t0.value)
            t1mean.append(t1.value)
            t2mean.append(t2.value)
            t3mean.append(t3.value)
            t4mean.append(t4.value)
            t5mean.append(t5.value)
            t6mean.append(t6.value)
        else:
            Qver1.append((Q_s + Q_h) / 2)
            Qver.append((Qmean_s[i] + Qmean_h[i]) / 2)
            t0mean.append(t0mean[i - 1] * (i) / (i + 1) + t0.value / (i + 1))
            t1mean.append(t1mean[i - 1] * (i) / (i + 1) + t1.value / (i + 1))
            t2mean.append(t2mean[i - 1] * (i) / (i + 1) + t2.value / (i + 1))
            t3mean.append(t3mean[i - 1] * (i) / (i + 1) + t3.value / (i + 1))
            t4mean.append(t4mean[i - 1] * (i) / (i + 1) + t4.value / (i + 1))
            t5mean.append(t5mean[i - 1] * (i) / (i + 1) + t5.value / (i + 1))
            t6mean.append(t6mean[i - 1] * (i) / (i + 1) + t6.value / (i + 1))

        # d2_pic.append(d2)
        # d4_pic.append(d4)
        # t0_pic.append(t0.value)
        # t1_pic.append(t1.value)
        # t2_pic.append(t2.value)
        print('slot=', i)
        print("us_loc=", ds_loc / es_loc, "uh_loc=", dh_loc / eh_loc, "us_off=", (d1.value + d2.value) / e_s, "uh_off=",
              (d3.value + d4.value + d5.value + d6.value) / e_h, )
        print("t0=", t0.value, ", t1=", t1.value, ", t2=", t2.value, "t3=", t3.value, "t4=", t4.value, "t5=", t5.value,
              "t6=", t6.value, "ts=", ts.value, "th=", th.value)
        print("ds_loc=", ds_loc, ", d1=", d1.value, ", d2= ", d2.value, ",dh_loc=", dh_loc, ", d3=", d3.value, ", d4=",
              d4.value, ",d5=", d5.value, ",d6=", d6.value)
        print("e_tot=", e_tot, "e_s=", e_s, "es_loc=", es_loc, "es_off=", es_off, ", es_hv=", es_hv, ", e_h=", e_h,
              "eh_loc=", eh_loc,"eh_off1=", eh_off1,"eh_off2=", eh_off2, ", eh_hv=", eh_hv)
        print("d_tot=", d_tot, "Q_s=", Q_s, "Q_h=", Q_h, ", B_s=", B_s, ", B_h=", B_h)
        print("bmean=", bmean[i], "emean=", emean[i], "Qver=",Qver[i],"Qmean_s=", Qmean_s[i], "Qmean_h=", Qmean_h[i], "Bmean_s=",
              Bmean_s[i], "Bmean_h=",Bmean_h[i])
        print("tao4=",tao4.value,)
        print("U=",U[i],'Umean=',Umean[i])
        print('Beta1=',Beta1,"Beta2=",Beta2,"Beta3=",Beta3)
        print("t0mean=",t0mean[i],"t1mean=",t1mean[i],"t2mean=",t2mean[i],"t3mean=",t3mean[i],"t4mean=",t4mean[i],"t5mean=",t5mean[i],"t6mean=",t6mean[i])
        print("--------------------------------------------------------------------------------------")

    Beta1 = tao0.value / t1.value
    Beta2 = tao1.value / t3.value
    Beta3 = tao2.value / t5.value
    Ps = tao3.value / t2.value
    Ph1 = tao4.value / t4.value
    Ph2 = tao5.value / t6.value

    data = [U,Umean,Qver,Qver1, Qmean_s, Qmean_h, t0mean,t1mean, t2mean, t3mean,t4mean, t5mean,t6mean]

    return data

def UC_BAC(q1=150, q2=100, b=100, ga=-16, d=120, w=1.2, v=40, a_s=1.2, a_h=1.5, miu=2, turn=5000):
    Q_s = q1
    Q_h = q2
    B_s = b
    B_h = b
    U=[]
    Umean=[]
    bmean = []
    emean=[]
    Qmean_s = []
    Qmean_h = []
    Qver = []
    Bmean_s = []
    Bmean_h = []
    t0mean=[]
    t1mean=[]
    t2mean=[]
    t3mean=[]
    t4mean=[]
    t5mean=[]
    t6mean=[]
    tao0_list = []
    # p1_list = []
    # p2_list = []
    # p3_list = []

    # Q_pic = [Q]
    # B1_pic = [0]
    # B2_pic = [0]
    # u_pic = []
    # umean_pic = []
    # d2_pic = []
    # d4_pic = []
    # t0_pic = []
    # t1_pic = []
    # t2_pic = []
    # dmean = 0
    # emean = 0

    # 设置参数
    # 节点之间距离
    dsa = 130
    dsh = d
    dha = 120

    T = 1
    P0 = 10
    Bmax = 200
    Pbach = Pbacs = 0.1
    Phtth = Phtts = 0.7
    Pmax = 1
    sigma2 = 10 ** (-3)  # 噪声功率
    W = w
    gap = 10 ** (ga / 10)
    phis = 490
    phih = 470
    fs = 500  # 计算频率
    fh = 480
    kai1 = 10 ** (-8)
    kai2 = kai1
    gsa = 3 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsa)) ** 3 * 10 ** 10  # 信道增益
    gsh = 2.8 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsh)) ** 3 * 10 ** 10
    gha = 2.5 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dha)) ** 3 * 10 ** 10
    miu1 = miu / (miu + 1)
    miu2 = 1 / (miu + 1)
    u = 0
    V = v

    # 新任务序列均值为3的指数分布
    A_s = produce_A(a_s)
    A_h = produce_A(a_h)
    # print(A)
    # A = np.ones(5000) * 2.58

    # 决策变量
    t0 = cp.Variable()
    t1 = cp.Variable()
    t3 = cp.Variable()
    t5 = cp.Variable()
    ts = cp.Variable()
    th = cp.Variable()
    d1 = cp.Variable()
    d3 = cp.Variable()
    d5 = cp.Variable()
    tao0 = cp.Variable()
    tao1 = cp.Variable()
    tao2 = cp.Variable()

    for i in range(turn):

        obj = cp.Minimize(
            -(Q_s + V ) * fs * ts / phis - (Q_s + V * miu1) * d1 - (Q_h + V ) * fh * th / phih - (Q_h + V * miu2) *d5 +
            (V * u - B_s + Bmax) * (Pbacs * t1 + kai1 * fs ** 3 * ts) + (V * u - B_h + Bmax) * (
                        Pbach * t3 + Pbach * t5 + kai2 * fh ** 3 * th)
            + ((B_s - Bmax) * gsa * P0 + (B_h - Bmax) * gha * P0) * ( t0+ t1 + t3 + t5) - (
                        B_s - Bmax) * gsa * P0 * tao0 - (B_h - Bmax) * gha * P0 * (tao1 + tao2))
        constraints = [
            t0 + t1 + t3 + t5 <= T,
            gsa * P0 * tao0 + (Pbacs * t1 + kai1 * fs ** 3 * ts) - gsa * P0 * ( t0+ t1 + t3 + t5) <= B_s,
            gha * P0 * (tao1 + tao2) + (Pbach * t3 + Pbach * t5 + kai2 * fh ** 3 * th) - gha * P0 * (
                    t0 + t1 + t3 +t5) <= B_h,
            d1 <= - W * cp.rel_entr(t1 * sigma2, tao0 * gap * P0 * gsa * gsh + t1 * sigma2) / sigma2,
            d3 <= - W * cp.rel_entr(t3 * sigma2, tao1 * gap * P0 * gha * gha + t3 * sigma2) / sigma2,
            d5 <= - W * cp.rel_entr(t5 * sigma2, tao2 * gap * P0 * gha * gha + t5 * sigma2) / sigma2,
            d1 <= d3,
            fs * ts / phis + d1 <= Q_s,
            fh * th / phih + d5 <= Q_h,
            t0 >= 0,
            t1 >= 0,
            t3 >= 0,
            t5 >= 0,
            ts >= 0,
            th >= 0,
            ts <= T,
            th <= T,
            tao0 >= 0,
            tao1 >= 0,
            tao2 >= 0,
            d1 >= 0,
            d3 >= 0,
            d5 >= 0,
            tao0 <= t1,
            tao1 <= t3,
            tao2 <= t5,
        ]

        prob = cp.Problem(obj, constraints)
        # prob.solve(verbose=True)
        prob.solve()

        # 更新u
        # print("value=", prob.value)
        # print(t0.value, t1.value, t2.value, tao1.value, tao2.value)

        # ds_off = t1.value * W * math.log(1 + tao1.value * gsh / t1.value / sigma2)
        # dh_off_s = t2.value * W * math.log(1 + tao2.value * gha / t2.value / sigma2)
        # dh_off_h = t3.value * W * math.log(1 + tao3.value * gha / t3.value / sigma2)
        ds_loc = fs * ts.value / phis
        dh_loc = fh * th.value / phih

        # a=B_h- gha * P0 * (tao1 + tao2) - (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5)
        #  +gsa * P0 * (t0 + t1 + t3 + t5)
        # print("a=",a.value)

        d_tot = ds_loc + miu1 * d1.value + dh_loc +miu2 *d5.value
        es_loc = kai1 * fs ** 3 * ts.value
        es_off = Pbacs * t1.value
        eh_loc = kai2 * fh ** 3 * th.value
        eh_off1 = Pbach * t3.value
        eh_off2 = Pbach * t5.value
        e_s = es_loc + es_off
        e_h = eh_loc + eh_off1 + eh_off2
        es_hv = (t0.value + t1.value + t3.value + t5.value - tao0.value) * P0 * gsa
        eh_hv = (t0.value + t1.value + t3.value + t5.value - tao1.value - tao2.value) * P0 * gha
        e_tot = e_s + e_h

        # 更新任务队列和电池队列
        Q_s = Q_s - d1.value - ds_loc + A_s[i]
        Q_h = Q_h - d5.value - dh_loc + A_h[i]
        B_s = B_s - gsa * P0 * tao0.value - (Pbacs * t1.value + kai1 * fs ** 3 * ts.value) + gsa * P0 * (
                        t0.value + t1.value + t3.value + t5.value)
        if B_s <= Bmax:
            B_s =B_s
        else:
            B_s = Bmax

        B_h = B_h - gha * P0 * (tao1.value + tao2.value) - (
                        Pbach * t3.value + Pbach * t5.value + kai2 * fh ** 3 * th.value) + gha * P0 * (
                              t0.value + t1.value + t3.value + t5.value)
        if B_h <= Bmax:
            B_h = B_h
        else:
            B_h = Bmax
        # gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3) - gsa * P0 * (t0 + t1 + t3 + t5) <= B_s,
        # gha * P0 * (tao1 + tao2) + (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5) - gsa * P0 * (
        #         t0 + t1 + t3 + t5) <= B_h,

        Beta1 = tao0.value / t1.value
        Beta2 = tao1.value / t3.value
        Beta3 = tao2.value / t5.value
        # Ps = tao3.value / t2.value
        # Ph1 = tao4.value / t4.value
        # Ph2 = tao5.value / t6.value
        # 记录队列及目标函数
        # Q_pic.append(Q)
        # B1_pic.append(B1)
        # B2_pic.append(B2)
        # u_pic.append(u)
        tao0_list.append(tao0.value)
        if i == 0:
            bmean.append(d_tot)
            emean.append(e_tot)
        else:
            bmean.append(bmean[i - 1] * (i) / (i + 1) + d_tot / (i + 1))
            emean.append(emean[i - 1] * (i) / (i + 1) + e_tot / (i + 1))

        u = bmean[i] / emean[i]

        if i == 0:
            U.append(d_tot / e_tot)
            Umean.append(u)
            Qmean_s.append(Q_s)
            Qmean_h.append(Q_h)
            Bmean_h.append(B_h)
            Bmean_s.append(B_s)
        else:
            U.append(u)
            Umean.append(Umean[i - 1] * (i) / (i + 1) + u / (i + 1))
            Qmean_h.append(Qmean_h[i - 1] * (i) / (i + 1) + Q_h / (i + 1))
            Qmean_s.append(Qmean_s[i - 1] * (i) / (i + 1) + Q_s / (i + 1))
            Bmean_h.append(Bmean_h[i - 1] * (i) / (i + 1) + B_h / (i + 1))
            Bmean_s.append(Bmean_s[i - 1] * (i) / (i + 1) + B_s / (i + 1))

        if i == 0:
            Qver.append((Q_s + Q_h)/2)
            t0mean.append(t0.value)
            t1mean.append(t1.value)
            t3mean.append(t3.value)
            t5mean.append(t5.value)
        else:
            Qver.append((Qmean_s[i] + Qmean_h[i]) / 2)
            t0mean.append(t0mean[i - 1] * (i) / (i + 1) + t0.value / (i + 1))
            t1mean.append(t1mean[i - 1] * (i) / (i + 1) + t1.value / (i + 1))
            t3mean.append(t3mean[i - 1] * (i) / (i + 1) + t3.value / (i + 1))
            t5mean.append(t5mean[i - 1] * (i) / (i + 1) + t5.value / (i + 1))

        # d2_pic.append(d2)
        # d4_pic.append(d4)
        # t0_pic.append(t0.value)
        # t1_pic.append(t1.value)
        # t2_pic.append(t2.value)
        print('slot=', i)
        print("us_loc=", ds_loc / es_loc, "uh_loc=", dh_loc / eh_loc, "us_off=", d1.value / e_s, "uh_off=",
              (d3.value + d5.value) / e_h, )
        print("t0=", t0.value, ", t1=", t1.value, "t3=", t3.value, "t5=", t5.value, "ts=", ts.value, "th=", th.value)
        print("ds_loc=", ds_loc, ", d1=", d1.value, ",dh_loc=", dh_loc, ", d3=", d3.value, ",d5=", d5.value)
        print("e_tot=", e_tot, "e_s=", e_s, "es_loc=", es_loc, "es_off=", es_off, ", es_hv=", es_hv, ", e_h=", e_h,
              "eh_loc=", eh_loc, "eh_off1=", eh_off1, "eh_off2=", eh_off2, ", eh_hv=", eh_hv)
        print("d_tot=", d_tot, "Q_s=", Q_s, "Q_h=", Q_h, ", B_s=", B_s, ", B_h=", B_h)
        print("bmean=", bmean[i], "emean=", emean[i], " Qavr=",Qver[i],"Qmean_s=", Qmean_s[i], "Qmean_h=", Qmean_h[i], "Bmean_s=",
              Bmean_s[i], "Bmean_h=", Bmean_h[i])
        print("U=", U[i], 'Umean=', Umean[i])
        print("t0mean=", t0mean[i], "t1mean=", t1mean[i], "t3mean=", t3mean[i], "t5mean=", t5mean[i])
        print("--------------------------------------------------------------------------------------")

    Beta1 = tao0.value / t1.value
    Beta2 = tao1.value / t3.value
    Beta3 = tao2.value / t5.value
    # Ps = tao3.value / t2.value
    # Ph1 = tao4.value / t4.value
    # Ph2 = tao5.value / t6.value

    data = [U,Umean, Qver,Qmean_s, Qmean_h, t0mean,t1mean, t2mean, t3mean,t4mean, t5mean,t6mean]


    return data


def UC_AC(q1=150, q2=100, b=100, ga=-16, d=120, w=1.2, v=40, a_s=1.2, a_h=1.5, miu=2, turn=5000):
    Q_s = q1
    Q_h = q2
    B_s = b
    B_h = b
    U=[]
    Umean=[]
    bmean = []
    emean=[]
    Qmean_s = []
    Qmean_h = []
    Qver = []
    Bmean_s = []
    Bmean_h = []
    t0mean=[]
    t1mean=[]
    t2mean=[]
    t3mean=[]
    t4mean=[]
    t5mean=[]
    t6mean=[]
    tao3_list = []
    # p1_list = []
    # p2_list = []
    # p3_list = []

    # Q_pic = [Q]
    # B1_pic = [0]
    # B2_pic = [0]
    # u_pic = []
    # umean_pic = []
    # d2_pic = []
    # d4_pic = []
    # t0_pic = []
    # t1_pic = []
    # t2_pic = []
    # dmean = 0
    # emean = 0

    # 设置参数
    # 节点之间距离
    dsa = 130
    dsh = d
    dha = 120

    T = 1
    P0 = 5
    Bmax = 200
    Pbach = Pbacs = 0.1
    Phtth = Phtts = 0.7
    Pmax = 1
    sigma2 = 10 ** (-3)  # 噪声功率
    W = w
    gap = 10 ** (ga / 10)
    phis = 490
    phih = 470
    fs = 500  # 计算频率
    fh = 480
    kai1 = 10 ** (-8)
    kai2 = kai1
    gsa = 3 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsa)) ** 3 * 10 ** 10  # 信道增益
    gsh = 2.8 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsh)) ** 3 * 10 ** 10
    gha = 2.5 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dha)) ** 3 * 10 ** 10
    miu1 = miu / (miu + 1)
    miu2 = 1 / (miu + 1)
    u = 0
    V = v
    # print(gsa, gsh, gha)

    # 新任务序列均值为3的指数分布
    A_s = produce_A(a_s)
    A_h = produce_A(a_h)
    # print(A)
    # A = np.ones(5000) * 2.58

    # 决策变量
    t0 = cp.Variable()
    t2 = cp.Variable()
    t4 = cp.Variable()
    t6 = cp.Variable()
    ts = cp.Variable()
    th = cp.Variable()
    d2 = cp.Variable()
    d4 = cp.Variable()
    d6 = cp.Variable()
    tao3 = cp.Variable()
    tao4 = cp.Variable()
    tao5 = cp.Variable()

    for i in range(turn):

        obj = cp.Minimize(-(Q_s + V) * fs * ts / phis - (Q_s + V * miu1) * d2 - (Q_h + V) *
                          fh * th / phih - (Q_h + V * miu2) * d6 + (V * u - B_s + Bmax) * (
                                      Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) + (V * u - B_h + Bmax) * (
                                  Phtth * t4 + tao4 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th)
                          + ((B_s - Bmax) * gsa * P0 + (B_h - Bmax) * gha * P0) * t0)
        constraints = [
            t0 + t2 + t4 + t6 <= T,
            (Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) - gsa * P0 * t0 <= B_s,
            (Phtth * t4 + tao4 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th) - gha * P0 * t0 <= B_h,
            d2 <= - W * cp.rel_entr(t2 * sigma2, tao3 * gsh + t2 * sigma2) / sigma2,
            d4 <= - W * cp.rel_entr(t4 * sigma2, tao4 * gha + t4 * sigma2) / sigma2,
            d6 <= - W * cp.rel_entr(t6 * sigma2, tao5 * gha + t6 * sigma2) / sigma2,
            d2 <= d4,
            fs * ts / phis + d2 <= Q_s,
            fh * th / phih + d6 <= Q_h,
            t0 >= 0,
            t2 >= 0,
            t4 >= 0,
            t6 >= 0,
            ts >= 0,
            th >= 0,
            ts <= T,
            th <= T,
            tao3 >= 0,
            tao4 >= 0,
            tao5 >= 0,
            d2 >= 0,
            d4 >= 0,
            d6 >= 0,
            tao3 <= t2 * Pmax,
            tao4 <= t4 * Pmax,
            tao5 <= t6 * Pmax,
        ]

        prob = cp.Problem(obj, constraints)
        # prob.solve(verbose=True)
        prob.solve()

        # 更新u
        # print("value=", prob.value)
        # print(t0.value, t1.value, t2.value, tao1.value, tao2.value)

        # ds_off = t1.value * W * math.log(1 + tao1.value * gsh / t1.value / sigma2)
        # dh_off_s = t2.value * W * math.log(1 + tao2.value * gha / t2.value / sigma2)
        # dh_off_h = t3.value * W * math.log(1 + tao3.value * gha / t3.value / sigma2)
        ds_loc = fs * ts.value / phis
        dh_loc = fh * th.value / phih

        # a=B_h- gha * P0 * (tao1 + tao2) - (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5)
        #  +gsa * P0 * (t0 + t1 + t3 + t5)
        # print("a=",a.value)

        d_tot = ds_loc + miu1 * d2.value + dh_loc + miu2*d6.value
        es_loc = kai1 * fs ** 3 * ts.value
        es_off = Phtts * t2.value + tao3.value
        eh_loc = kai2 * fh ** 3 * th.value
        eh_off1 = Phtth * t4.value + tao4.value
        eh_off2 = Phtth * t6.value + tao5.value
        e_s = es_loc + es_off
        e_h = eh_loc + eh_off1 + eh_off2
        es_hv = t0.value * P0 * gsa
        eh_hv = t0.value * P0 * gha
        e_tot = e_s + e_h

        # 更新任务队列和电池队列
        Q_s = Q_s - d2.value - ds_loc + A_s[i]
        Q_h = Q_h - d6.value - dh_loc + A_h[i]
        B_s = B_s - (Phtts * t2.value + tao3.value + kai1 * fs ** 3 * ts.value) + gsa * P0 * t0.value
        if B_s <= Bmax:
            B_s = B_s
        else:
            B_s = Bmax
        B_h = B_h - (
                        Phtth * t4.value + tao4.value + Phtth * t6.value + tao5.value + kai2 * fh ** 3 * th.value) + gha * P0 * t0.value
        if B_h <= Bmax:
            B_h = B_h
        else:
            B_h = Bmax
        # gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3) - gsa * P0 * (t0 + t1 + t3 + t5) <= B_s,
        # gha * P0 * (tao1 + tao2) + (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5) - gsa * P0 * (
        #         t0 + t1 + t3 + t5) <= B_h,
        # Beta1 = tao0.value / t1.value
        # Beta2 = tao1.value / t3.value
        # Beta3 = tao2.value / t5.value
        Ps = tao3.value / t2.value
        Ph1 = tao4.value / t4.value
        Ph2 = tao5.value / t6.value
        # 记录队列及目标函数
        # Q_pic.append(Q)
        # B1_pic.append(B1)
        # B2_pic.append(B2)
        # u_pic.append(u)
        tao3_list.append(tao3.value)
        if i == 0:
            bmean.append(d_tot)
            emean.append(e_tot)
        else:
            bmean.append(bmean[i - 1] * (i) / (i + 1) + d_tot / (i + 1))
            emean.append(emean[i - 1] * (i) / (i + 1) + e_tot / (i + 1))

        u = bmean[i] / emean[i]

        if i == 0:
            U.append(d_tot / e_tot)
            Umean.append(u)
            Qmean_s.append(Q_s)
            Qmean_h.append(Q_h)
            Bmean_h.append(B_h)
            Bmean_s.append(B_s)
        else:
            U.append(u)
            Umean.append(Umean[i - 1] * (i) / (i + 1) + u / (i + 1))
            Qmean_h.append(Qmean_h[i - 1] * (i) / (i + 1) + Q_h / (i + 1))
            Qmean_s.append(Qmean_s[i - 1] * (i) / (i + 1) + Q_s / (i + 1))
            Bmean_h.append(Bmean_h[i - 1] * (i) / (i + 1) + B_h / (i + 1))
            Bmean_s.append(Bmean_s[i - 1] * (i) / (i + 1) + B_s / (i + 1))

        if i == 0:
            Qver.append((Q_s + Q_h) / 2)
            t0mean.append(t0.value)
            t2mean.append(t2.value)
            t4mean.append(t4.value)
            t6mean.append(t6.value)
        else:
            Qver.append((Qmean_s[i] + Qmean_h[i]) / 2)
            t0mean.append(t0mean[i - 1] * (i) / (i + 1) + t0.value / (i + 1))
            t2mean.append(t2mean[i - 1] * (i) / (i + 1) + t2.value / (i + 1))
            t4mean.append(t4mean[i - 1] * (i) / (i + 1) + t4.value / (i + 1))
            t6mean.append(t6mean[i - 1] * (i) / (i + 1) + t6.value / (i + 1))
        # d2_pic.append(d2)
        # d4_pic.append(d4)
        # t0_pic.append(t0.value)
        # t1_pic.append(t1.value)
        # t2_pic.append(t2.value)
        print('slot=', i)
        print("us_loc=", ds_loc / es_loc, "uh_loc=", dh_loc / eh_loc, "us_off=", miu1*d2.value / (e_s+eh_off1), "uh_off=",
              miu2*(d6.value) / eh_off2, )
        print("t0=", t0.value, ", t2=", t2.value, "t4=", t4.value, "t6=", t6.value, "ts=", ts.value, "th=", th.value)
        print("ds_loc=", ds_loc, ", d2= ", d2.value, ",dh_loc=", dh_loc, ", d4=", d4.value, ",d6=", d6.value)
        print("e_tot=", e_tot, "e_s=", e_s, "es_loc=", es_loc, "es_off=", es_off, ", es_hv=", es_hv, ", e_h=", e_h,
              "eh_loc=", eh_loc, "eh_off1=", eh_off1, "eh_off2=", eh_off2, ", eh_hv=", eh_hv)
        print("d_tot=", d_tot, "Q_s=", Q_s, "Q_h=", Q_h, ", B_s=", B_s, ", B_h=", B_h)
        print("bmean=", bmean[i], "emean=", emean[i], "Qver=",Qver[i],"Qmean_s=", Qmean_s[i], "Qmean_h=", Qmean_h[i], "Bmean_s=",
              Bmean_s[i], "Bmean_h=", Bmean_h[i])
        print("U=", U[i], 'Umean=', Umean[i])
        print("t0mean=", t0mean[i], "t2mean=", t2mean[i], "t4mean=", t4mean[i], "t6mean=", t6mean[i])
        print("--------------------------------------------------------------------------------------")

    # Beta1 = tao0.value / t1.value
    # Beta2 = tao1.value / t3.value
    # Beta3 = tao2.value / t5.value
    Ps = tao3.value / t2.value
    Ph1 = tao4.value / t4.value
    Ph2 = tao5.value / t6.value

    data = [U,Umean, Qver,Qmean_s, Qmean_h, t0mean, t2mean, t4mean, t6mean]


    return data


def nocoop(q1=150, q2=100, b=100, ga=-16, d=120, w=1.2, v=40, a_s=1.2, a_h=1.5, miu=2, turn=5000):
    Q_s = q1
    Q_h = q2
    B_s = b
    B_h = b
    U=[]
    Umean=[]
    bmean = []
    emean=[]
    Qmean_s = []
    Qmean_h = []
    Qver = []
    Bmean_s = []
    Bmean_h = []
    t0mean=[]
    t1mean=[]
    t2mean=[]
    t5mean=[]
    t6mean=[]
    tao0_list = []
    # p1_list = []
    # p2_list = []
    # p3_list = []

    # Q_pic = [Q]
    # B1_pic = [0]
    # B2_pic = [0]
    # u_pic = []
    # umean_pic = []
    # d2_pic = []
    # d4_pic = []
    # t0_pic = []
    # t1_pic = []
    # t2_pic = []
    # dmean = 0
    # emean = 0

    # 设置参数
    # 节点之间距离
    dsa = 130
    dsh = d
    dha = 120

    T = 1
    P0 = 5
    Bmax = 200
    Pbach = Pbacs = 0.1
    Phtth = Phtts = 0.7
    Pmax = 1
    alpha=0.02
    sigma2 = 10 ** (-3)  # 噪声功率
    W = w
    gap = 10 ** (ga / 10)
    phis = 490
    phih = 470
    fs = 500  # 计算频率
    fh = 480
    kai1 = 10 ** (-8)
    kai2 = kai1
    gsa = 3 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsa)) ** 3 * 10 ** 10  # 信道增益
    gsh = 2.8 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsh)) ** 3 * 10 ** 10
    gha = 2.5 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dha)) ** 3 * 10 ** 10
    miu1 = miu / (miu + 1)
    miu2 = 1 / (miu + 1)
    u = 0
    V = v
    a=(2.0150534691784663-1.5420720223590247)/2.0150534691784663*100
    b=(2.0150534691784663-1.3344988757471887)/2.0150534691784663*100
    c=(2.0150534691784663-1.0857563325780388)/2.0150534691784663*100
    # print(gsa, gsh, gha)

    # 新任务序列均值为3的指数分布
    A_s = produce_A(a_s)
    A_h = produce_A(a_h)
    # print(A)
    # A = np.ones(5000) * 2.58

    # 决策变量
    t0 = cp.Variable()
    t1 = cp.Variable()
    t2 = cp.Variable()
    t5 = cp.Variable()
    t6 = cp.Variable()
    ts = cp.Variable()
    th = cp.Variable()
    d1 = cp.Variable()
    d2 = cp.Variable()
    d5 = cp.Variable()
    d6 = cp.Variable()
    tao0 = cp.Variable()
    tao2 = cp.Variable()
    tao3 = cp.Variable()
    tao5 = cp.Variable()

    for i in range(turn):

        obj = cp.Minimize(-(Q_s + V) * fs * ts / phis - (Q_s + V * miu1) * (d1 + d2) - (Q_h + V) *
                          fh * th / phih - (Q_h + V * miu2) * (d5 + d6) +
                          (V * u - B_s + Bmax) * (Pbacs * t1 + Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) + (
                                  V * u - B_h + Bmax) * (
                                   Pbach * t5 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th)
                          + ((B_s - Bmax) * gsa * P0 + (B_h - Bmax) * gha * P0) * (
                                   t0 + t1 + t5) - (B_s - Bmax) * gsa * P0 * tao0 - (B_h - Bmax) * gha * P0 * (
                                   tao2))
        constraints = [
            t0 + t1 + t2 + t5 + t6 <= T,
            gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) - gsa * P0 * (
                    t0 + t1 + t5) <= B_s,
            gha * P0 * tao2 + (
                     Pbach * t5 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th) - gha * P0 * (
                    t0 + t1 + t5) <= B_h,
            d1 <= - W * cp.rel_entr(t1 * sigma2, tao0 * gap * P0 * gsa * gsa*alpha + t1 * sigma2) / sigma2,
            d2 <= - W * cp.rel_entr(t2 * sigma2, tao3 * gsa*alpha + t2 * sigma2) / sigma2,
            d5 <= - W * cp.rel_entr(t5 * sigma2, tao2 * gap * P0 * gha * gha + t5 * sigma2) / sigma2,
            d6 <= - W * cp.rel_entr(t6 * sigma2, tao5 * gha + t6 * sigma2) / sigma2,
            fs * ts / phis + d1 + d2 <= Q_s,
            fh * th / phih + d5 + d6 <= Q_h,
            t0 >= 0,
            t1 >= 0,
            t2 >= 0,
            t5 >= 0,
            t6 >= 0,
            ts >= 0,
            th >= 0,
            ts <= T,
            th <= T,
            tao0 >= 0,
            tao2 >= 0,
            tao3 >= 0,
            tao5 >= 0,
            d1 >= 0,
            d2 >= 0,
            d5 >= 0,
            d6 >= 0,
            tao0 <= t1,
            tao2 <= t5,
            tao3 <= t2 * Pmax,
            tao5 <= t6 * Pmax,
        ]

        prob = cp.Problem(obj, constraints)
        # prob.solve(verbose=True)
        prob.solve()

        # 更新u
        # print("value=", prob.value)
        # print(t0.value, t1.value, t2.value, tao1.value, tao2.value)

        # ds_off = t1.value * W * math.log(1 + tao1.value * gsh / t1.value / sigma2)
        # dh_off_s = t2.value * W * math.log(1 + tao2.value * gha / t2.value / sigma2)
        # dh_off_h = t3.value * W * math.log(1 + tao3.value * gha / t3.value / sigma2)
        ds_loc = fs * ts.value / phis
        dh_loc = fh * th.value / phih

        # a1= w * t1.value * math.log(1 + ( tao0.value * gap * P0 * gsa * gsa) / (t1.value * sigma2))
        a2= w * t2.value * math.log(1 + (tao3.value * gsa) / (t2.value * sigma2))
        # a3= w * t5.value * math.log(1 + ( tao2.value * gap * P0 * gha * gha) / (t5.value * sigma2))
        a4 = w * t6.value * math.log(1 + (tao5.value * gha) / (t6.value * sigma2))
        print("a2=",a2,"a4=",a4)

        # a=B_h- gha * P0 * (tao1 + tao2) - (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5)
        #  +gsa * P0 * (t0 + t1 + t3 + t5)
        # print("a=",a.value)

        d_tot = ds_loc + miu1 * (d1.value + d2.value) + dh_loc + miu2 * (d5.value + d6.value)
        es_loc = kai1 * fs ** 3 * ts.value
        es_off = Pbacs * t1.value + Phtts * t2.value + tao3.value
        eh_loc = kai2 * fh ** 3 * th.value
        eh_off2 = Pbach * t5.value + Phtth * t6.value + tao5.value
        e_s = es_loc + es_off
        e_h = eh_loc  + eh_off2
        es_hv = (t0.value + t1.value + t5.value - tao0.value) * P0 * gsa
        eh_hv = (t0.value + t1.value + t5.value - tao2.value) * P0 * gha
        e_tot = e_s + e_h

        # 更新任务队列和电池队列
        Q_s = Q_s - d1.value - d2.value - ds_loc + A_s[i]
        Q_h = Q_h - d5.value - d6.value - dh_loc + A_h[i]
        B_s = B_s- gsa * P0 * tao0.value - (Pbacs * t1.value + Phtts * t2.value + tao3.value+kai1 * fs ** 3 * ts.value) + gsa * P0 * (t0.value + t1.value + t5.value)
        if B_s <= 50:
           B_s=B_s
        else:
           B_s=50
        B_h = B_h-gha * P0 * (tao2.value) - ( Pbach * t5.value + Phtth * t6.value + tao5.value+ kai2 * fh ** 3 * th.value)+gha * P0 * (t0.value + t1.value + t5.value)
        if B_h <= 50:
            B_h =B_h
        else:
           B_h=50
        # gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3) - gsa * P0 * (t0 + t1 + t3 + t5) <= B_s,
        # gha * P0 * (tao1 + tao2) + (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5) - gsa * P0 * (
        #         t0 + t1 + t3 + t5) <= B_h,
        Beta1 = tao0.value / t1.value
        Beta3 = tao2.value / t5.value
        Ps = tao3.value / t2.value
        Ph2 = tao5.value / t6.value
        # 记录队列及目标函数
        # Q_pic.append(Q)
        # B1_pic.append(B1)
        # B2_pic.append(B2)
        # u_pic.append(u)
        tao0_list.append(tao0.value)
        if i == 0:
            bmean.append(d_tot)
            emean.append(e_tot)
        else:
            bmean.append(bmean[i - 1] * (i) / (i + 1) + d_tot / (i + 1))
            emean.append(emean[i - 1] * (i) / (i + 1) + e_tot / (i + 1))

        u =bmean[i] / emean[i]

        if i == 0:
            U.append(d_tot / e_tot)
            Umean.append(u)
            Qmean_s.append(Q_s)
            Qmean_h.append(Q_h)
            Bmean_h.append(B_h)
            Bmean_s.append(B_s)
        else:
            U.append(u)
            Umean.append(Umean[i - 1] * (i) / (i + 1) + u / (i + 1))
            Qmean_h.append(Qmean_h[i - 1] * (i) / (i + 1) + Q_h / (i + 1))
            Qmean_s.append(Qmean_s[i - 1] * (i) / (i + 1) + Q_s / (i + 1))
            Bmean_h.append(Bmean_h[i - 1] * (i) / (i + 1) + B_h / (i + 1))
            Bmean_s.append(Bmean_s[i - 1] * (i) / (i + 1) + B_s / (i + 1))

        if i == 0:
            Qver.append((Q_s + Q_h)/2)
            t0mean.append(t0.value)
            t1mean.append(t1.value)
            t2mean.append(t2.value)
            t5mean.append(t5.value)
            t6mean.append(t6.value)
        else:
            Qver.append((Qmean_s[i] + Qmean_h[i]) / 2)
            t0mean.append(t0mean[i - 1] * (i) / (i + 1) + t0.value / (i + 1))
            t1mean.append(t1mean[i - 1] * (i) / (i + 1) + t1.value / (i + 1))
            t2mean.append(t2mean[i - 1] * (i) / (i + 1) + t2.value / (i + 1))
            t5mean.append(t5mean[i - 1] * (i) / (i + 1) + t5.value / (i + 1))
            t6mean.append(t6mean[i - 1] * (i) / (i + 1) + t6.value / (i + 1))

        # d2_pic.append(d2)
        # d4_pic.append(d4)
        # t0_pic.append(t0.value)
        # t1_pic.append(t1.value)
        # t2_pic.append(t2.value)
        print('slot=', i)
        print("us_loc=", ds_loc / es_loc, "uh_loc=", dh_loc / eh_loc, "us_off=", (d1.value + d2.value) / e_s, "uh_off=",
              (d5.value + d6.value) / e_h, )
        print("t0=", t0.value, ", t1=", t1.value, ", t2=", t2.value,  "t5=", t5.value,
              "t6=", t6.value, "ts=", ts.value, "th=", th.value)
        print("ds_loc=", ds_loc, ", d1=", d1.value, ", d2= ", d2.value, ",dh_loc=", dh_loc,"d5=", d5.value, ",d6=", d6.value)
        print("e_tot=", e_tot, "e_s=", e_s, "es_loc=", es_loc, "es_off=", es_off, ", es_hv=", es_hv, ", e_h=", e_h,
              "eh_loc=", eh_loc,"eh_off2=", eh_off2, ", eh_hv=", eh_hv)
        print("d_tot=", d_tot, "Q_s=", Q_s, "Q_h=", Q_h, ", B_s=", B_s, ", B_h=", B_h)
        print("bmean=", bmean[i], "emean=", emean[i], "Qver=",Qver[i],"Qmean_s=", Qmean_s[i], "Qmean_h=", Qmean_h[i], "Bmean_s=",
              Bmean_s[i], "Bmean_h=",Bmean_h[i])
        print("U=",U[i],'Umean=',Umean[i])
        print('Beta1=',Beta1,"Beta3=",Beta3)
        print("t0mean=",t0mean[i],"t1mean=",t1mean[i],"t2mean=",t2mean[i],"t3mean=","t5mean=",t5mean[i],"t6mean=",t6mean[i])
        print("--------------------------------------------------------------------------------------")

    Beta1 = tao0.value / t1.value
    Beta3 = tao2.value / t5.value
    Ps = tao3.value / t2.value
    Ph2 = tao5.value / t6.value

    data = [U,Umean,Qver, Qmean_s, Qmean_h, t0mean,t1mean, t2mean, t5mean,t6mean]

    return data
def nolyp(q1=150, q2=100, b=100, ga=-16, d=120, w=1.2, v=40, a_s=1.2, a_h=1.5, miu=2, turn=5000):
    Q_s = q1
    Q_h = q2
    B_s = b
    B_h = b
    U=[]
    Umean=[]
    bmean = []
    emean=[]
    Qmean_s = []
    Qmean_h = []
    Qver = []
    Qver1=[]
    Bmean_s = []
    Bmean_h = []
    t0mean=[]
    t1mean=[]
    t2mean=[]
    t3mean=[]
    t4mean=[]
    t5mean=[]
    t6mean=[]
    tao0_list = []
    # p1_list = []
    # p2_list = []
    # p3_list = []

    # Q_pic = [Q]
    # B1_pic = [0]
    # B2_pic = [0]
    # u_pic = []
    # umean_pic = []
    # d2_pic = []
    # d4_pic = []
    # t0_pic = []
    # t1_pic = []
    # t2_pic = []
    # dmean = 0
    # emean = 0

    # 设置参数
    # 节点之间距离
    dsa = 130
    dsh = d
    dha = 120

    T = 1
    P0 = 5
    Bmax = 200
    Pbach = Pbacs = 0.1
    Phtth = Phtts = 0.7
    Pmax = 1
    sigma2 = 10 ** (-3)  # 噪声功率
    W = w
    gap = 10 ** (ga / 10)
    phis = 490
    phih = 470
    fs = 500  # 计算频率
    fh = 480
    kai1 = 10 ** (-8)
    kai2 = kai1
    gsa = 3 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsa)) ** 3 * 10 ** 10  # 信道增益
    gsh = 2.8 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dsh)) ** 3 * 10 ** 10
    gha = 2.5 * (3 * 10 ** 8 / (4 * math.pi * 915 * 10 ** 6 * dha)) ** 3 * 10 ** 10
    miu1 = miu / (miu + 1)
    miu2 = 1 / (miu + 1)
    u = 0
    V = v
    # print(gsa, gsh, gha)

    # 新任务序列均值为3的指数分布
    A_s = produce_A(a_s)
    A_h = produce_A(a_h)
    # print(A)
    # A = np.ones(5000) * 2.58

    # 决策变量
    t0 = cp.Variable()
    t1 = cp.Variable()
    t2 = cp.Variable()
    t3 = cp.Variable()
    t4 = cp.Variable()
    t5 = cp.Variable()
    t6 = cp.Variable()
    ts = cp.Variable()
    th = cp.Variable()
    d1 = cp.Variable()
    d2 = cp.Variable()
    d3 = cp.Variable()
    d4 = cp.Variable()
    d5 = cp.Variable()
    d6 = cp.Variable()
    tao0 = cp.Variable()
    tao1 = cp.Variable()
    tao2 = cp.Variable()
    tao3 = cp.Variable()
    tao4 = cp.Variable()
    tao5 = cp.Variable()

    for i in range(turn):

        obj = cp.Minimize(-(fs * ts / phis + miu1 * (d1 + d2) +
                          fh * th / phih +  miu2 * (d5 + d6)) +
                           u* (Pbacs * t1 + Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) +  u  * (
                                  Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th))
        constraints = [
            t0 + t1 + t2 + t3 + t4 + t5 + t6 <= T,
            gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3 + kai1 * fs ** 3 * ts) - gsa * P0 * (
                    t0 + t1 + t3 + t5) <= B_s,
            gha * P0 * (tao1 + tao2) + (
                    Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5 + kai2 * fh ** 3 * th) - gha * P0 * (
                    t0 + t1 + t3 + t5) <= B_h,
            d1 <= - W * cp.rel_entr(t1 * sigma2, tao0 * gap * P0 * gsa * gsh + t1 * sigma2) / sigma2,
            d2 <= - W * cp.rel_entr(t2 * sigma2, tao3 * gsh + t2 * sigma2) / sigma2,
            d3 <= - W * cp.rel_entr(t3 * sigma2, tao1 * gap * P0 * gha * gha + t3 * sigma2) / sigma2,
            d4 <= - W * cp.rel_entr(t4 * sigma2, tao4 * gha + t4 * sigma2) / sigma2,
            d5 <= - W * cp.rel_entr(t5 * sigma2, tao2 * gap * P0 * gha * gha + t5 * sigma2) / sigma2,
            d6 <= - W * cp.rel_entr(t6 * sigma2, tao5 * gha + t6 * sigma2) / sigma2,
            d1 <= d3,
            d2 <= d4,
            fs * ts / phis + d1 + d2 <= Q_s,
            # fs * ts / phis + d1 + d2 >= 1,
            fh * th / phih + d5 + d6 <= Q_h,
            # fh * th / phih + d5 + d6 >= 1,
            t0 >= 0,
            t1 >= 0,
            t2 >= 0,
            t3 >= 0,
            t4 >= 0,
            t5 >= 0,
            t6 >= 0,
            ts >= 0,
            th >= 0,
            ts <= T,
            th <= T,
            tao0 >= 0,
            tao1 >= 0,
            tao2 >= 0,
            tao3 >= 0,
            tao4 >= 0,
            tao5 >= 0,
            d1 >= 0,
            d2 >= 0,
            d3 >= 0,
            d4 >= 0,
            d5 >= 0,
            d6 >= 0,
            tao0 <= t1,
            tao1 <= t3,
            tao2 <= t5,
            tao3 <= t2 * Pmax,
            tao4 <= t4 * Pmax,
            tao5 <= t6 * Pmax,
        ]

        prob = cp.Problem(obj, constraints)
        # prob.solve(verbose=True)
        prob.solve()
        a=w*t1.value*math.log(1+(tao0.value * gap * P0 * gsa * gsh) /( t1.value * sigma2))
        b=w*t4.value*math.log(1+(tao4.value*gha)/(t4.value*sigma2))
        print("a=",a,"b=",b)
        # 更新u
        # print("value=", prob.value)
        # print(t0.value, t1.value, t2.value, tao1.value, tao2.value)

        # ds_off = t1.value * W * math.log(1 + tao1.value * gsh / t1.value / sigma2)
        # dh_off_s = t2.value * W * math.log(1 + tao2.value * gha / t2.value / sigma2)
        # dh_off_h = t3.value * W * math.log(1 + tao3.value * gha / t3.value / sigma2)
        ds_loc = fs * ts.value / phis
        dh_loc = fh * th.value / phih

        # a=B_h- gha * P0 * (tao1 + tao2) - (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5)
        #  +gsa * P0 * (t0 + t1 + t3 + t5)
        # print("a=",a.value)

        d_tot = ds_loc + miu1 * (d1.value + d2.value) + dh_loc + miu2 * (d5.value + d6.value)
        es_loc = kai1 * fs ** 3 * ts.value
        es_off = Pbacs * t1.value + Phtts * t2.value + tao3.value
        eh_loc = kai2 * fh ** 3 * th.value
        eh_off1 = Pbach * t3.value + Phtth * t4.value + tao4.value
        eh_off2 = Pbach * t5.value + Phtth * t6.value + tao5.value
        e_s = es_loc + es_off
        e_h = eh_loc + eh_off1 + eh_off2
        es_hv = (t0.value + t1.value + t3.value + t5.value - tao0.value) * P0 * gsa
        eh_hv = (t0.value + t1.value + t3.value + t5.value - tao1.value - tao2.value) * P0 * gha
        e_tot = e_s + e_h

        # 更新任务队列和电池队列
        Q_s = Q_s - d1.value - d2.value - ds_loc + A_s[i]
        Q_h = Q_h - d5.value - d6.value - dh_loc + A_h[i]
        B_s = B_s- gsa * P0 * tao0.value - (Pbacs * t1.value + Phtts * t2.value + tao3.value+kai1 * fs ** 3 * ts.value) + gsa * P0 * (t0.value + t1.value + t3.value + t5.value)
        if B_s <= Bmax:
           B_s=B_s
        else:
           B_s=Bmax
        B_h = B_h-gha * P0 * (tao1.value + tao2.value) - (Pbach * t3.value + Phtth * t4.value + tao4.value + Pbach * t5.value + Phtth * t6.value + tao5.value+ kai2 * fh ** 3 * th.value)+gha * P0 * (t0.value + t1.value + t3.value + t5.value)
        if B_h <= Bmax:
            B_h =B_h
        else:
           B_h=Bmax
        # gsa * P0 * tao0 + (Pbacs * t1 + Phtts * t2 + tao3) - gsa * P0 * (t0 + t1 + t3 + t5) <= B_s,
        # gha * P0 * (tao1 + tao2) + (Pbach * t3 + Phtth * t4 + tao4 + Pbach * t5 + Phtth * t6 + tao5) - gsa * P0 * (
        #         t0 + t1 + t3 + t5) <= B_h,
        Beta1 = tao0.value / t1.value
        Beta2 = tao1.value / t3.value
        Beta3 = tao2.value / t5.value
        Ps = tao3.value / t2.value
        Ph1 = tao4.value / t4.value
        Ph2 = tao5.value / t6.value
        # 记录队列及目标函数
        # Q_pic.append(Q)
        # B1_pic.append(B1)
        # B2_pic.append(B2)
        # u_pic.append(u)
        tao0_list.append(tao0.value)
        if i == 0:
            bmean.append(d_tot)
            emean.append(e_tot)
        else:
            bmean.append(bmean[i - 1] * (i) / (i + 1) + d_tot / (i + 1))
            emean.append(emean[i - 1] * (i) / (i + 1) + e_tot / (i + 1))

        u =bmean[i] / emean[i]

        if i == 0:
            U.append(d_tot / e_tot)
            Umean.append(u)
            Qmean_s.append(Q_s)
            Qmean_h.append(Q_h)
            Bmean_h.append(B_h)
            Bmean_s.append(B_s)
        else:
            U.append(u)
            Umean.append(Umean[i - 1] * (i) / (i + 1) + u / (i + 1))
            Qmean_h.append(Qmean_h[i - 1] * (i) / (i + 1) + Q_h / (i + 1))
            Qmean_s.append(Qmean_s[i - 1] * (i) / (i + 1) + Q_s / (i + 1))
            Bmean_h.append(Bmean_h[i - 1] * (i) / (i + 1) + B_h / (i + 1))
            Bmean_s.append(Bmean_s[i - 1] * (i) / (i + 1) + B_s / (i + 1))

        if i == 0:
            Qver1.append((Q_s + Q_h)/2)
            Qver.append((Q_s + Q_h)/2)
            t0mean.append(t0.value)
            t1mean.append(t1.value)
            t2mean.append(t2.value)
            t3mean.append(t3.value)
            t4mean.append(t4.value)
            t5mean.append(t5.value)
            t6mean.append(t6.value)
        else:
            Qver1.append((Q_s + Q_h) / 2)
            Qver.append((Qmean_s[i] + Qmean_h[i]) / 2)
            t0mean.append(t0mean[i - 1] * (i) / (i + 1) + t0.value / (i + 1))
            t1mean.append(t1mean[i - 1] * (i) / (i + 1) + t1.value / (i + 1))
            t2mean.append(t2mean[i - 1] * (i) / (i + 1) + t2.value / (i + 1))
            t3mean.append(t3mean[i - 1] * (i) / (i + 1) + t3.value / (i + 1))
            t4mean.append(t4mean[i - 1] * (i) / (i + 1) + t4.value / (i + 1))
            t5mean.append(t5mean[i - 1] * (i) / (i + 1) + t5.value / (i + 1))
            t6mean.append(t6mean[i - 1] * (i) / (i + 1) + t6.value / (i + 1))

        # d2_pic.append(d2)
        # d4_pic.append(d4)
        # t0_pic.append(t0.value)
        # t1_pic.append(t1.value)
        # t2_pic.append(t2.value)
        print('slot=', i)
        print("us_loc=", ds_loc / es_loc, "uh_loc=", dh_loc / eh_loc, "us_off=", (d1.value + d2.value) / e_s, "uh_off=",
              (d3.value + d4.value + d5.value + d6.value) / e_h, )
        print("t0=", t0.value, ", t1=", t1.value, ", t2=", t2.value, "t3=", t3.value, "t4=", t4.value, "t5=", t5.value,
              "t6=", t6.value, "ts=", ts.value, "th=", th.value)
        print("ds_loc=", ds_loc, ", d1=", d1.value, ", d2= ", d2.value, ",dh_loc=", dh_loc, ", d3=", d3.value, ", d4=",
              d4.value, ",d5=", d5.value, ",d6=", d6.value)
        print("e_tot=", e_tot, "e_s=", e_s, "es_loc=", es_loc, "es_off=", es_off, ", es_hv=", es_hv, ", e_h=", e_h,
              "eh_loc=", eh_loc,"eh_off1=", eh_off1,"eh_off2=", eh_off2, ", eh_hv=", eh_hv)
        print("d_tot=", d_tot, "Q_s=", Q_s, "Q_h=", Q_h, ", B_s=", B_s, ", B_h=", B_h)
        print("bmean=", bmean[i], "emean=", emean[i], "Qver=",Qver[i],"Qmean_s=", Qmean_s[i], "Qmean_h=", Qmean_h[i], "Bmean_s=",
              Bmean_s[i], "Bmean_h=",Bmean_h[i])
        print("tao4=",tao4.value,)
        print("U=",U[i],'Umean=',Umean[i])
        print('Beta1=',Beta1,"Beta2=",Beta2,"Beta3=",Beta3)
        print("t0mean=",t0mean[i],"t1mean=",t1mean[i],"t2mean=",t2mean[i],"t3mean=",t3mean[i],"t4mean=",t4mean[i],"t5mean=",t5mean[i],"t6mean=",t6mean[i])
        print("--------------------------------------------------------------------------------------")

    Beta1 = tao0.value / t1.value
    Beta2 = tao1.value / t3.value
    Beta3 = tao2.value / t5.value
    Ps = tao3.value / t2.value
    Ph1 = tao4.value / t4.value
    Ph2 = tao5.value / t6.value

    data = [U,Umean,Qver,Qver1, Qmean_s, Qmean_h, t0mean,t1mean, t2mean, t3mean,t4mean, t5mean,t6mean]

    return data
if __name__ == '__main__':
  nolyp()
