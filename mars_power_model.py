import numpy as np

# -----------------------------
# CONSTANTS
# -----------------------------
G0 = 590             # Solar constant at Mars [W/m²]
T_ref = 25           # Reference temperature [°C]
k_B = 1.380649e-23   # Boltzmann constant [J/K]
q_e = 1.602176634e-19  # Electron charge [C]

# -----------------------------
# CORE FUNCTIONS
# -----------------------------

def mars_irradiance(G0, tau, theta_z_deg):
    """Compute surface irradiance using Beer–Lambert attenuation."""
    mu = np.cos(np.radians(theta_z_deg))
    G = G0 * mu * np.exp(-tau / mu)
    return G, mu

def temperature_correction(eta_ref, alpha_eta, T_cell, T_ref=T_ref):
    """Temperature dependence of efficiency."""
    return eta_ref * (1 + alpha_eta * (T_cell - T_ref))

def soiling_transmittance(kappa_s, m_d):
    """Compute dust transmittance and effective efficiency factor."""
    T_soil = np.exp(-kappa_s * m_d)
    return T_soil

def combined_efficiency(eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d):
    """Combine temperature and dust soiling effects."""
    eta_temp = temperature_correction(eta_ref, alpha_eta, T_cell, T_ref)
    T_soil = soiling_transmittance(kappa_s, m_d)
    return eta_temp * T_soil

def instantaneous_power(A, G0, tau, theta_z, theta_i, eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d):
    """Full combined power model (static or time-varying)."""
    mu_z = np.cos(np.radians(theta_z))
    mu_i = np.cos(np.radians(theta_i))
    attenuation = np.exp(-tau / mu_z)
    eta_eff = combined_efficiency(eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d)
    P = A * G0 * mu_i * attenuation * eta_eff
    return P

# -----------------------------
# EXAMPLE CALCULATION (your given case)
# -----------------------------
if __name__ == "__main__":
    # Given parameters
    G0 = 590
    theta_z = 30       # degrees
    theta_i = 30       # degrees
    tau = 2.0
    A = 1.0            # m²
    eta_ref = 0.20
    T_cell = 0         # °C
    T_ref = 25         # °C
    alpha_eta = -0.004
    m_d = 50           # g/m²
    kappa_s = 0.01     # m²/g

    # Step-by-step
    G, mu = mars_irradiance(G0, tau, theta_z)
    eta_eff = combined_efficiency(eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d)
    P = instantaneous_power(A, G0, tau, theta_z, theta_i, eta_ref, alpha_eta, T_cell, T_ref, kappa_s, m_d)

    print("===== Mars Dust Storm Power Calculation =====")
    print(f"Solar constant (G0): {G0} W/m²")
    print(f"Solar zenith angle: {theta_z}° (μ={mu:.3f})")
    print(f"Optical depth (τ): {tau}")
    print(f"Irradiance at surface (G): {G:.2f} W/m²")
    print(f"Effective efficiency (η): {eta_eff*100:.2f}%")
    print(f"Power output (P): {P:.3f} W")