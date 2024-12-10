import numpy as np
import pandas as pd

def DCPH(T, DA, DB, DC, DD):
    return DA + DB * T + DC * T**2 + DD * T**-2

def DCPS(T,DA,DB,DC,DD):
    return (DA + DB * T + DC * T**2 + DD * T**-2)/T

# def IDCPH(T0,T1, DA, DB, DC, DD):
#     resultado, error = quad(DCPH,T0,T1, args=(DA,DB,DC,DD))
#     return resultado

# def IDCPS(T0,T1, DA, DB, DC, DD):
#     resultado, error = quad(DCPS,T0,T1, args=(DA,DB,DC,DD))
#     return resultado

# Carga de datos

coef = pd.read_csv('../CSV/CoefReac.csv')
thermoprops = pd.read_csv('../CSV/thermodynamic_properties.csv')

R = np.array(8.314) # J/(mol K) Constante de los gases ideales
T = np.array(1000) # K, Temperatura del sistema reaccionante
T0 = np.array(298.15) # K, Temperatura de referencia

A = np.array(thermoprops["A"])
B = np.array(thermoprops["B"])
C = np.array(thermoprops["C"])
D = np.array(thermoprops["D"])

GF = np.array(thermoprops["GF"])
HF = np.array(thermoprops["HF"])


reac = np.array([coef["R_1"], coef["R_2"], coef["R_3"], coef["R_4"], coef["R_5"], coef["R_6"], coef["R_7"], coef["R_8"]])

print(GF)