import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

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



# Propiedades termodinámicas 

R = np.array(8.314) # J/(mol K) Constante de los gases ideales
# T = np.arange(1000,1120,10) # [K]
T = np.array(1000) # [K]
T = T.reshape(-1,1)
T0 = np.array(298.15) # K, Temperatura de referencia
SB = np.array(0.287) # Relación de biomasa/vapor

A = np.array(thermoprops["A"]).reshape(-1,1)
B = np.array(thermoprops["B"]).reshape(-1,1)
C = np.array(thermoprops["C"]).reshape(-1,1)
D = np.array(thermoprops["D"]).reshape(-1,1)

GF = np.array(thermoprops["GF"]).reshape(-1,1)
HF = np.array(thermoprops["HF"]).reshape(-1,1)

# Especies en el sistema
reac = np.transpose(np.array([coef['CH4'],coef["H2O"],coef['CO'],coef['CO2'], coef['H2']]))


DA = np.sum(A * reac, axis=0)
DB = np.sum(B * reac, axis=0)
DC = np.sum(C * reac, axis=0)
DD = np.sum(D * reac, axis=0)
DGF = np.sum(GF * reac, axis=0)
DHF = np.sum(HF * reac, axis=0)


n = np.size(T,0)
m = np.size(reac,1)

# Determinación de la energía libre de Gibbs para las reacciones de gasificación a cada temperatura dada

DGf_RT = np.zeros((n,m))

for i in range(n):
    for j in range(m):
        IDCPH_R = IDCPH(T0, T[i].item(), DA[j].item(), DB[j].item(), DC[j].item(), DD[j].item())
        IDCPS_R = IDCPS(T0, T[i].item(), DA[j].item(), DB[j].item(), DC[j].item(), DD[j].item())
        DGf_RT[i,j] = (DGF[j].item() - DHF[j].item()) / (R * T0) + DHF[j].item() / (R * T[i].item())+ 1 / T[i].item() * IDCPH_R - IDCPS_R


# Modelo de equilibrio

# DGf_i / R * T + log((y_i phi_i P) / Pº) + suma(lag_k*a_ik) = 0

# Donde:

#  El primer termino es el cambio estandar de la Energía libre de Gibbs
#  El segundo termino del logaritmo es la influencia de la fugacidad en el potencial químico
#  El tercer termino son las restricciones del sistema con los multiplidaores de lagrange.

# Siendo reacciones que se llevan a cabo por arriba de los 1000 K, y a presiones atmosféricas se puede considerar como gases ideales


Ak = np.array([4.55, 18.7444, 14.564]) # Balance de masas atómicas en el sistema antes de la reacción C H O

# Definir el sistema de ecuaciones
def sistema(vars, DG, T, Ak):
    y1, y2, y3, y4, y5, l1, l2, l3, x = vars 
    # Modelo
    eq1 = DG[0] + np.log(y1) + l1 / (R * T) + 4 * l2 / (R * T)
    eq2 = DG[1] + np.log(y2) + 2 * l2 / (R * T) + l3 / (R * T)
    eq3 = DG[2] + np.log(y3) + l1 / (R * T) + l3 / (R * T)
    eq4 = DG[3] + np.log(y4) + l1 / (R * T) + 2 * l3 / (R * T)
    eq5 = DG[4] + np.log(y5) + 2 * l2 / (R * T)
    
    # Restricciones
    eq6 = y1 + y3 + y4 - Ak[0] / x
    eq7 = 4 * y1 + 2 * y2 + 2 * y5 - Ak[1] / x
    eq8 = y2 + y3 + 2 * y4 - Ak[2] / x
    eq9 = y1 + y2 + y3 + y4 + y5 - 1
    return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9]


# Arreglo para guardar los datos de la solución del sistema

y = np.zeros((n,9)) # Aquí depende del numero de incognitas en el sistema

for i in range(n):
    DG = DGf_RT[i,:]
    Ti = T[i].item()

    # Valores iniciales
    valores_iniciales = [0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1]

    # Resolver el sistema
    soluciones = fsolve(sistema, valores_iniciales, args=(DG, Ti, Ak,))
    y[i] = soluciones
    valores_iniciales = soluciones



fmol = ["y_CH4","y_H2O","y_CO","y_CO2","y_H2"]
print(f"")
print("-"*37)
print(f"|{'Syngas composition':^35}|")
print("-"*37)
print(f"|{'Temperature, K':^15}|{'Steam/Biomass Ratio':^19}|")
print("-"*37)
print(f"|{Ti:^15.2f}|{SB:^19.2f}|")
print("-"*37)
print(f"|{'Especies':^15}|{'Fracción mol':^19}|")
print("-"*37)

for i in range(5):
    print(f"|{fmol[i]:<15}|{soluciones[i]:>19.4f}|")
    print("-"*37)


# plt.plot(T,y[ : , 0 ], c='black', label='CH4')
# plt.plot(T,y[ : , 1 ], c='blue', label='H2O')
# plt.plot(T,y[ : , 2 ], c='red', label='CO')
# plt.plot(T,y[ : , 3 ], c='green', label='CO2')
# plt.plot(T,y[ : , 4 ], c='magenta', label='H2')
# plt.legend()
# plt.show()
