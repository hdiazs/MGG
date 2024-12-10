import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad


def DCPH(T, DA, DB, DC, DD):
    return DA + DB * T + DC * T**2 + DD * T**-2

def DCPS(T,DA,DB,DC,DD):
    return (DA + DB * T + DC * T**2 + DD * T**-2)/T

def IDCPH(T0,T1, DA, DB, DC, DD):
    resultado, error = quad(DCPH,T0,T1, args=(DA,DB,DC,DD))
    return resultado

def IDCPS(T0,T1, DA, DB, DC, DD):
    resultado, error = quad(DCPS,T0,T1, args=(DA,DB,DC,DD))
    return resultado

# Carga de datos

coef = pd.read_csv('../CSV/CoefReac.csv')
thermoprops = pd.read_csv('../CSV/thermodynamic_properties.csv')

R = np.array(8.314) # J/(mol K) Constante de los gases ideales
T = np.arange(1000,1500,50).reshape(-1,1) # [K]
T0 = np.array(298.15) # K, Temperatura de referencia

A = np.array(thermoprops["A"]).reshape(-1,1)
B = np.array(thermoprops["B"]).reshape(-1,1)
C = np.array(thermoprops["C"]).reshape(-1,1)
D = np.array(thermoprops["D"]).reshape(-1,1)

GF = np.array(thermoprops["GF"]).reshape(-1,1)
HF = np.array(thermoprops["HF"]).reshape(-1,1)

reac = np.transpose(np.array([coef["R_1"], coef["R_2"], coef["R_3"], coef["R_4"], coef["R_5"], coef["R_6"], coef["R_7"], coef["R_8"]]))

DA = np.sum(A * reac, axis=0)
DB = np.sum(B * reac, axis=0)
DC = np.sum(C * reac, axis=0)
DD = np.sum(D * reac, axis=0)
DGF = np.sum(GF * reac, axis=0)
DHF = np.sum(HF * reac, axis=0)

n = len(T)
m = np.size(reac,1)

DG_RT_R = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        IDCPH_R = IDCPH(T0, T[i].item(), DA[j].item(), DB[j].item(), DC[j].item(), DD[j].item())
        IDCPS_R = IDCPS(T0, T[i].item(), DA[j].item(), DB[j].item(), DC[j].item(), DD[j].item())
        DG_RT_R[i,j] = (DGF[j].item() - DHF[j].item()) / (R * T0) + DHF[j].item() / (R * T[i].item())+ 1 / T[i].item() * IDCPH_R - IDCPS_R