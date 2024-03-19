import numpy as np
import matplotlib.pyplot as plt

## Define material parameters
# Static strain-stress curve
static_curve_strain = np.array([0, 1])
static_curve_stress = np.array([0, 1e7])
# Elastic curve type
elastic_stress_type = "poly"  # "interp" or "poly"
# Memory kernel coefficient
kernel_coef = 0.1870
# Cubic viscoelastic coefficients
elastic_coef = np.array([178.9901, -479.8327, 344.3367])
visc_loading_coef = np.array([265.2555, -143.9302, 528.3796])
visc_unloading_coef = np.array([1403.7904, -1873.0947, 575.5831])


## Define strain timeseries
time_final = 3.5
time_step = 0.005
time = np.arange(0, time_final, time_step)
strain_reference = 0.2
strain_amplitude = 0.01
strain_period = 0.5
strain = strain_reference + strain_amplitude * np.exp(-5 / (time + 0.001)) * np.sin(
    2 * np.pi / strain_period * time
)
strain_rate = (
    -2
    * np.pi
    / strain_period
    * strain_amplitude
    * np.sin(2 * np.pi / strain_period * time)
)
num_points = len(strain)

## Modify strain curve
if elastic_stress_type == "poly":
    static_curve_strain = np.arange(
        strain_reference - strain_amplitude, strain_reference + strain_amplitude, 0.01
    )
    static_curve_stress = np.zeros_like(static_curve_strain)
    for j in range(3):
        static_curve_stress = (
            static_curve_stress + (static_curve_strain ** (j + 1)) * elastic_coef[j]
        )


## Stress computation
# Initiallize stress
stress = np.zeros_like(strain)
tau = np.array([])
visc_resp = np.array([])
kernel_rate = np.array([])
# Strain rate zero-crossing times
zc_times = np.array([])
# Strain rate zero-crossing viscoelastic loading responses
zc_visc_resp_load = np.array([])
# Strain rate zero-crossing viscoelastic unloading responses
zc_visc_resp_unload = np.array([])
for i in range(num_points):
    # Extract temporal strain values from vector
    tmp_time = time[i]
    tmp_strain = strain[i]
    tmp_strain_rate = strain_rate[i]

    # Compute elastic stress interpolating
    if elastic_stress_type == "interp":
        elastic_stress = np.interp(tmp_strain, static_curve_strain, static_curve_stress)
    elif elastic_stress_type == "poly":
        elastic_stress = 0.0
        for j in range(3):
            elastic_stress = elastic_stress + (tmp_strain ** (j + 1)) * elastic_coef[j]

    # Compute viscous stress
    if (i > 0) and (np.sign(strain_rate[i]) != np.sign(strain_rate[i - 1])):
        zc_time_tmp = (
            strain_rate[i - 1] * time_step / (tmp_strain_rate - strain_rate[i - 1])
            + time[i - 1]
        )
        zc_times = np.append(zc_times, zc_time_tmp)
        zc_strain_tmp = np.interp(
            np.array([zc_time_tmp]),
            np.array([time[i - 1], time[i]]),
            np.array([strain[i - 1], strain[i]]),
        )
        visc_resp_load_tmp = 0.0
        visc_resp_unload_tmp = 0.0
        for j in range(3):
            visc_resp_load_tmp = (
                visc_resp_load_tmp + (zc_strain_tmp ** (j + 1)) * visc_loading_coef[j]
            )
            visc_resp_unload_tmp = (
                visc_resp_unload_tmp
                + (zc_strain_tmp ** (j + 1)) * visc_unloading_coef[j]
            )
        zc_visc_resp_load = np.append(zc_visc_resp_load, visc_resp_load_tmp)
        zc_visc_resp_unload = np.append(zc_visc_resp_unload, visc_resp_unload_tmp)

    if tmp_strain_rate >= 0:
        strain_poly_coef = visc_loading_coef
    else:
        strain_poly_coef = visc_unloading_coef
    visc_resp_tmp = 0.0
    for j in range(3):
        visc_resp_tmp = visc_resp_tmp + (tmp_strain ** (j + 1)) * strain_poly_coef[j]

    if i == 0:
        visc_resp_0 = visc_resp_tmp
    tau = np.append(tau, tmp_time)
    visc_resp = np.append(visc_resp, visc_resp_tmp)
    kernel = np.exp(
        -kernel_coef * (tmp_time - tau)
    )  # TODO: replace by call to kernel function
    kernel_rate = -kernel_coef * kernel  # TODO: replace by call to kernel function

    # Integration by parts
    integrand = kernel_rate * visc_resp
    viscous_stress_integral = np.trapz(integrand, tau)
    initial_stress_integral = (
        np.exp(-kernel_coef * tmp_time) * visc_resp_0
    )  # TODO: replace by call to kernel function
    final_stress_integral = visc_resp_tmp  # TODO: replace by call to kernel function
    sum_stress_integral = 0.0
    for ii, t_k in enumerate(zc_times):
        sum_stress_integral = sum_stress_integral + np.exp(
            -kernel_coef * (tmp_time - t_k)
        ) * ((-1) ** (ii)) * (zc_visc_resp_load[ii] - zc_visc_resp_unload[ii])
    viscous_stress = (
        viscous_stress_integral
        + final_stress_integral
        - initial_stress_integral
        + sum_stress_integral
    )

    stress[i] = elastic_stress + viscous_stress

    if tmp_time == np.round(time_final / 2):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].set_title(
            f"Time = {tmp_time}s ; Viscous stress integral = {viscous_stress_integral} N"
        )
        axs[0].plot(tau, visc_resp, "k-")
        axs[1].set_title(f"Time = {tmp_time}s ; Kernel")
        axs[1].plot(tau, kernel, "k-")
        axs[2].set_title(f"Time = {tmp_time}s ; Integrand")
        axs[2].plot(tau, integrand, "k-")
        plt.tight_layout()

## Plotting the results
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
axs[2].set_ylim(
    (np.min(stress) - 0.05 * np.mean(stress), np.max(stress) + 0.05 * np.mean(stress))
)

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
