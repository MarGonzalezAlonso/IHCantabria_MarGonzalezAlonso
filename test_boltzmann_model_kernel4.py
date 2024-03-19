import numpy as np
import math
import matplotlib.pyplot as plt

## Define material parameters
# Static strain-stress curve
static_curve_strain = np.array([0, 1])
static_curve_stress = np.array([0, 1e7])
# Memory kernel coefficient
kernel_coef = np.array([1, -4])
# Cubic viscoelastic coefficients
visc_loading_coef = np.array([-7e6, 0, 0])
visc_unloading_coef = np.array([-5e6, 0, 0])

## Define strain timeseries
time_final = 100
time_step = 0.01
time = np.arange(0.001, time_final, time_step)
strain_reference = 0.2
strain_amplitude = 0.1
strain_period = 5
strain = strain_reference + strain_amplitude * np.sin(2 * np.pi / strain_period * time)
strain_rate = np.array([])  # ó np.zeros_like(strain) # no analítico
num_points = len(strain)


## Stress computation
# Initiallize stress
stress = np.zeros_like(strain)
tau = np.array([])
visc_resp = np.array([])
# Loop over all time steps

for i in range(num_points):
    # Compute elastic stress interpolating
    elastic_stress = np.interp(strain[i], static_curve_strain, static_curve_stress)
    tmp_time = time[i]
    tmp_strain = strain[i]
    # Compute strain rate
    d_strain = 0
    tmp_strain_rate = 0
    if i > 0:
        d_strain = strain[i] - strain[i - 1]
        d_tau = time_step
        # temporal strain rate scalar
        tmp_strain_rate = d_strain / d_tau
        strain_rate = np.append(strain_rate, tmp_strain_rate)
    if tmp_strain_rate >= 0:
        strain_poly_coef = visc_loading_coef
    else:
        strain_poly_coef = visc_unloading_coef
    strain_poly = 0.0
    for j in range(3):
        strain_poly = strain_poly + (tmp_strain**j) * strain_poly_coef[j] * (j + 1)
    # Compute viscous stress
    tau = np.append(tau, tmp_time)
    visc_resp = np.append(visc_resp, tmp_strain_rate * strain_poly)
    kernel = 0
    if i > 0:
        # kernel = kernel_coef[0] * tau * np.exp(kernel_coef[1] * (tau)) / (np.sqrt(tau))
        kernel = (
            kernel_coef[0] * np.exp(kernel_coef[1] * (tmp_time - tau)) / (np.sqrt(tau))
        )
    integrand = kernel * visc_resp  # / time_step  # remove time step!!!!!!!!!!!!!!!!!!!
    viscous_stress = np.trapz(integrand, tau)
    if tmp_time == np.round(time_final / 2):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].set_title(f"Time = {tmp_time}s ; Viscous stress = {viscous_stress} N")
        axs[0].plot(tau, visc_resp, "k-")
        axs[1].set_title(f"Time = {tmp_time}s ; Kernel")
        axs[1].plot(tau, kernel, "k-")
        axs[2].set_title(f"Time = {tmp_time}s ; Integrand")
        axs[2].plot(tau, integrand, "k-")
        plt.tight_layout()

    # Add both stresses
    stress[i] = elastic_stress + viscous_stress


# Creating the subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Plot for strain-time
axs[0].plot(time, strain, "k-")
axs[0].set_title("Strain vs Time")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Strain")

# Plot for stress-time
axs[1].plot(time, stress, "k-")
axs[1].set_title("Stress vs Time")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Stress")

# Plot for stress-strain
axs[2].plot(strain, stress, "k-")
axs[2].plot(static_curve_strain, static_curve_stress, "r--")
axs[2].set_title("Stress vs Strain")
axs[2].set_xlabel("Strain")
axs[2].set_ylabel("Stress")
axs[2].set_xlim(
    (
        strain_reference - strain_amplitude * 1.1,
        strain_reference + strain_amplitude * 1.1,
    )
)
# axs[2].set_ylim(
#    (np.min(stress) - 0.05 * np.mean(stress), np.max(stress) + 0.05 * np.mean(stress))


# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
