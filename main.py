# main.py
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# USER INPUTS
# -----------------------------
solar_constant = 590       # W/m^2 (Mars average)
tau = float(input("Enter dust opacity (0.1 to 5): "))
panel_area = float(input("Enter solar panel area (m^2): "))
efficiency = float(input("Enter panel efficiency (%): ")) / 100
storm_duration = float(input("Enter storm duration (hours): "))

# -----------------------------
# SIMULATION
# -----------------------------
time = np.linspace(0, storm_duration, 200)  # time steps
# simulate how opacity rises and falls during storm
tau_profile = tau * np.exp(-((time - storm_duration/2)**2) / (storm_duration/4)**2)

# Solar irradiance decreases exponentially with tau
irradiance = solar_constant * np.exp(-tau_profile)

# Power output from solar panels
power = efficiency * panel_area * irradiance

# Compute total energy (Wh)
energy_total = np.trapz(power, time)
energy_clear = efficiency * panel_area * solar_constant * storm_duration
energy_loss = energy_clear - energy_total

# -----------------------------
# OUTPUT
# -----------------------------
print(f"\nAverage Power during storm: {np.mean(power):.2f} W")
print(f"Total Energy Generated: {energy_total:.2f} Wh")
print(f"Energy Loss due to storm: {energy_loss:.2f} Wh")

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(time, tau_profile, color='brown')
plt.title("Dust Opacity Over Time")
plt.ylabel("Optical Depth (τ)")

plt.subplot(2,1,2)
plt.plot(time, power, color='orange')
plt.title("Solar Power Output During Storm")
plt.xlabel("Time (hours)")
plt.ylabel("Power (W)")
plt.tight_layout()
plt.show()