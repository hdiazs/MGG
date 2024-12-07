import numpy as np
import pandas as pd



# Carga de datos

coef = pd.read_csv('../CSV/CoefReac.csv')

R = np.array(8.314) # J/(mol K) Constante de los gases ideales
T = np.array(1000) # K, Temperatura del sistema reaccionante


G = np.array([19720, -192420, -200240, -395790, 0])

G_RT = G / (R * T)


reac = np.array([coef["R_1"], coef["R_2"], coef["R_3"], coef["R_4"], coef["R_5"], coef["R_6"], coef["R_7"], coef["R_8"]])

print(reac)