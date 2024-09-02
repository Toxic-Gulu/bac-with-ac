import cvxpy as cp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random
import csv
import os
from new_func.myfunc import *
from new_func.algorithms import *


def do_proposed(turn=5000):
    data = proposed(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'proposed')

def do_UC_BAC(turn=5000):
    data = UC_BAC(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'UC_BAC')

def do_UC_AC(turn=5000):
    data = UC_AC(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'UC_AC')

def do_nocoop(turn=5000):
    data = nocoop(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'nocoop')

def do_nolyp(turn=5000):
    data = nolyp(turn=turn)
    data.append(np.arange(turn) + 1)
    savedata(data, 'nolyp')

def do_proposed_difga(turn=2000):
    U=[]
    Umean = []
    Qmean_s = []
    Qmean_h = []
    t0mean = []
    t1mean=[]
    t2mean = []
    t3mean=[]
    t4mean = []
    t5mean=[]
    t6mean = []
    for ga in np.arange(-22, -11, 1):
        data = proposed(turn=turn,ga=ga)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
        t0mean.append(data[5][-1])
        t1mean.append(data[6][-1])
        t2mean.append(data[7][-1])
        t3mean.append(data[8][-1])
        t4mean.append(data[9][-1])
        t5mean.append(data[10][-1])
        t6mean.append(data[11][-1])
    data = [U,Umean, Qmean_s, Qmean_h,t0mean,t1mean,t2mean,t3mean,t4mean,t5mean,t6mean, np.arange(-22, -11, 1)]
    savedata(data, 'difga_proposed')

def do_UC_BAC_difga(turn=2000):
    U=[]
    Umean = []
    Qmean_s = []
    Qmean_h = []
    for ga in np.arange(-22, -11, 1):
        data = UC_BAC(turn=turn,ga=ga)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U,Umean, Qmean_s, Qmean_h, np.arange(-22, -11, 1)]
    savedata(data, 'difga_UC_BAC')

def do_UC_AC_difga(turn=2000):
    U=[]
    Umean = []
    Qmean_s = []
    Qmean_h = []
    for ga in np.arange(-22, -11, 1):
        data = UC_AC(turn=turn,ga=ga)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U,Umean, Qmean_s, Qmean_h, np.arange(-22, -11, 1)]
    savedata(data, 'difga_UC_AC')

def do_nocoop_difga(turn=2000):
    U=[]
    Umean = []
    Qmean_s = []
    Qmean_h = []
    for ga in np.arange(-22, -11, 1):
        data = nocoop(turn=turn,ga=ga)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U,Umean, Qmean_s, Qmean_h, np.arange(-22, -11, 1)]
    savedata(data, 'difga_nocoop')
def do_proposed_difW(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for w in np.arange(1,1.5, 0.05):
        print(w)
        data = proposed(turn=turn, w=w)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean,Qver, Qmean_s, Qmean_h, np.arange(1,1.5, 0.05)]
    savedata(data, 'difW_proposed')

def do_UC_BAC_difW(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for w in np.arange(1,1.5, 0.05):
        print(w)
        data = UC_BAC(turn=turn, w=w)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean, Qver, Qmean_s, Qmean_h, np.arange(1,1.5, 0.05)]
    savedata(data, 'difW_UC_BAC')

def do_UC_AC_difW(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for w in np.arange(1,1.5, 0.05):
        print(w)
        data = UC_AC(turn=turn, w=w)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean, Qver, Qmean_s, Qmean_h, np.arange(1,1.5, 0.05)]
    savedata(data, 'difW_UC_AC')

def do_nocoop_difW(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for w in np.arange(1,1.5, 0.05):
        print(w)
        data =nocoop(turn=turn, w=w)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean, Qver, Qmean_s, Qmean_h, np.arange(1,1.5, 0.05)]
    savedata(data, 'difW_nocoop')

def do_proposed_difmiu(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for miu in np.arange(0, 3.3, 0.3):
        data = proposed(turn=turn, miu=miu)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean, Qver, Qmean_s, Qmean_h, np.arange(0, 3.3, 0.3)]
    savedata(data, 'difmiu_proposed')

def do_UC_BAC_difmiu(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for miu in np.arange(0, 3.3, 0.3):
        data = UC_BAC(turn=turn, miu=miu)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean, Qver, Qmean_s, Qmean_h, np.arange(0, 3.3, 0.3)]
    savedata(data, 'difmiu_UC_BAC')

def do_UC_AC_difmiu(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for miu in np.arange(0, 3.3, 0.3):
        data = UC_AC(turn=turn, miu=miu)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean, Qver, Qmean_s, Qmean_h, np.arange(0, 3.3, 0.3)]
    savedata(data, 'difmiu_UC_AC')

def do_nocoop_difmiu(turn=2000):
    U = []
    Umean = []
    Qver = []
    Qmean_s = []
    Qmean_h = []
    for miu in np.arange(0, 3.3, 0.3):
        data = nocoop(turn=turn, miu=miu)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qmean_s.append(data[3][-1])
        Qmean_h.append(data[4][-1])
    data = [U, Umean, Qver, Qmean_s, Qmean_h, np.arange(0, 3.3, 0.3)]
    savedata(data, 'difmiu_nocoop')

def do_proposed_difV(turn=2000,ga=-16):#每个时隙dif_v的具体情况
    U = []
    Umean= []
    Qver = []
    Qver1=[]
    for v in np.arange(10,110,10):
        data = proposed(turn=turn, v=v,ga=ga)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
        Qver1.append(data[3][-1])
    data = [U, Umean, Qver,Qver1,np.arange(10,100,10)]
    savedata(data, 'difV_EE')



def do_proposed_difD(turn=2000):
    U = []
    Umean = []
    Qver = []
    for d in np.arange(80, 180, 10):
        data = proposed(turn=turn, d=d)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])
    data = [U, Umean,Qver, np.arange(80, 180, 10)]
    savedata(data, 'difD_EE')


def do_proposed_difA(turn=2000):
    U = []
    Umean = []
    Qver = []
    for a_h in np.arange(1.7, 2.8, 0.1):
        data = proposed(turn=turn, a_h=a_h)
        U.append(data[0][-1])
        Umean.append(data[1][-1])
        Qver.append(data[2][-1])

    data = [U, Umean,Qver, np.arange(1.7, 2.8, 0.1)]
    savedata(data, 'difA_EE')

