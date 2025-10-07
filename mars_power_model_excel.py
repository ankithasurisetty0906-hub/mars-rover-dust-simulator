import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONSTANTS
# -----------------------------
G0 = 590           # W/m², Mars solar constant
eta_ref = 0.20     # Reference efficiency
T_ref = 25         # °C
alpha_eta = -0.004 # Temperature coefficient
kappa_s = 0.01     # Dust extinction coefficient m²/g
A = 1.0            # Panel area m²

# -----------------------------
# FUNCTIONS
# -----------------------------
def temperature_correction(eta_ref, alpha_eta, T_cell, T_ref=T_ref):
    return eta_ref * (1 + alpha_eta * (T_cell - T_ref))

def soiling_transmittance(kappa_s, m_d):
    return np.exp(-kappa_s * m_d)

def combined_efficiency(eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d):
    return temperature_correction(eta_ref, alpha_eta, T_cell, T_ref) * soiling_transmittance(kappa_s, m_d)

def instantaneous_power(A, G0, tau, theta_z, theta_i, eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d):
    mu_z = np.cos(np.radians(theta_z))
    mu_i = np.cos(np.radians(theta_i))
    attenuation = np.exp(-tau / mu_z)
    eta_eff = combined_efficiency(eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d)
    P = A * G0 * mu_i * attenuation * eta_eff
    return P

# -----------------------------
# READ DATA FROM EXCEL
# -----------------------------
data = pd.read_excel("mars_dust_data.xlsx")

# Compute power for each row
powers = []
for idx, row in data.iterrows():
    P = instantaneous_power(
        A, G0,
        tau=row['τ (optical depth)'],
        theta_z=row['θ_z (deg)'],
        theta_i=row['θ_z (deg)'],  # assume panel tilt = solar zenith for simplicity
        eta_ref=eta_ref,
        alpha_eta=alpha_eta,
        T_cell=row['T_cell (°C)'],
        T_ref=T_ref,
        kappa_s=kappa_s,
        m_d=row['m_d (g/m²)']
    )
    powers.append(P)

data['Power (W)'] = powers

# -----------------------------
# PLOT RESULTS
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(data['Time (h)'], data['Power (W)'], marker='o', color='orange', label='PV Power (W)')
plt.xlabel('Time (h)')
plt.ylabel('Power Output (W)')
plt.title('Mars Solar Panel Power During Dust Storm')
plt.grid(True)
plt.legend()
plt.show()