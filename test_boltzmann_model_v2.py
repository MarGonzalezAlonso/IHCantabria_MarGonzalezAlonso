import numpy as np
import matplotlib.pyplot as plt

## Define material parameters
# Static strain-stress curve
static_curve_strain = np.array([0, 1])
static_curve_stress = np.array([0, 1e7])
# Memory kernel coefficient
kernel_coef = 1
# Cubic viscoelastic coefficients
visc_loading_coef = np.array([1e7, 0, 0])
visc_unloading_coef = np.array([1e7, 0, 0])

## Define strain timeseries
time_final = 100
time_step = 0.01
time = np.arange(0, time_final, time_step)
strain_reference = 0.2
strain_amplitude = 0.1
strain_period = 5
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


## Stress computation
# Initiallize stress
stress = np.zeros_like(strain)
tau = np.array([])
visc_resp = np.array([])
times_strain_rate_zeros = np.array([])
loading_poly = 0
unloading_poly = 0
loading_poly_zeros = 0
unloading_poly_zeros = 0
for i in range(num_points):
    # Compute elastic stress interpolating
    elastic_stress = np.interp(strain[i], static_curve_strain, static_curve_stress)
    # Compute viscous stress
    tmp_time = time[i]
    tmp_strain = strain[i]
    tmp_strain_rate = strain_rate[i]
    if tmp_strain_rate >= 0:
        strain_poly_coef = visc_loading_coef
    else:
        strain_poly_coef = visc_unloading_coef
    strain_poly = 0.0
    strain_poly_rate = 0.0
    for j in range(3):
        strain_poly = strain_poly + (tmp_strain ** (j + 1)) * strain_poly_coef[j]
        strain_poly_rate = strain_poly_rate + (tmp_strain**j) * strain_poly_coef[
            j
        ] * (j + 1)
        loading_poly = loading_poly + tmp_strain ** (j + 1) * visc_loading_coef[j]
        unloading_poly = unloading_poly + tmp_strain ** (j + 1) * visc_unloading_coef[j]

    tau = np.append(tau, tmp_time)
    # Integration by parts

    # visc_resp = np.append(visc_resp, tmp_strain_rate * strain_poly_rate)

    kernel = np.exp(-kernel_coef * (tmp_time - tau))
    kernel_rate = kernel_coef * kernel
    visc_resp = np.append(visc_resp, strain_poly)
    # integrand = kernel * visc_resp  # / time_step  # remove time step!!!!!!!!!!!!!!!!!!!
    integrand = kernel_rate * visc_resp
    viscous_stress_integral = np.trapz(integrand, tau)

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

    if i == 0:
        # initial_integral_parts = kernel * strain_poly
        initial_integral_parts = np.exp(-kernel_coef * (time_final)) * strain_poly
    initial_integral_parts = initial_integral_parts

    final_integral_parts = 0
    if i == num_points:
        final_integral_parts = strain_poly

    # for j in range(3):
    #    loading_poly = loading_poly + tmp_strain ** (j + 1) * visc_loading_coef[j]
    #    unloading_poly = unloading_poly + tmp_strain ** (j + 1) * visc_unloading_coef[j]
    times_strain_rate_zeros = np.array([])
    # sum_stresses = np.array([])
    sum_stresses = 0
    count_zeros = 0
    # sum_stresses = np.append(sum_stresses, 0)
    if i >= 1 and i <= num_points - 2:
        if time[i - 1] - tmp_time < 0 or tmp_time - time[i - 1] < 0:
            # times_strain_rate_zeros[i - 1] = (
            #    -strain_rate[i - 1] * time_step / (tmp_strain_rate - strain_rate[i - 1])
            #    + tmp_time
            # )
            count_zeros += 1
            times_strain_rate_zeros = np.append(
                times_strain_rate_zeros,
                strain_rate[i - 1] * time_step / (tmp_strain_rate - strain_rate[i - 1])
                + time[i - 1],
            )
            # strain_zeros = np.interp(
            #    np.array(
            #        [time[i - 1], times_strain_rate_zeros[count_zeros - 1], tmp_time]
            #    ),
            #    np.array([time[i - 1], tmp_time]),
            #    np.array([strain[i - 1], tmp_strain]),
            # )
            strain_zeros = np.interp(
                times_strain_rate_zeros[count_zeros - 1],
                np.array([time[i - 1], tmp_time]),
                np.array([strain[i - 1], tmp_strain]),
            )
            for j in range(3):
                loading_poly_zeros = (
                    loading_poly_zeros + strain_zeros ** (j + 1) * visc_loading_coef[j]
                )
                unloading_poly_zeros = (
                    unloading_poly_zeros
                    + strain_zeros ** (j + 1) * visc_unloading_coef[j]
                )
            # sum_stresses[i] = sum_stresses[i - 1] + np.exp(
            #    -kernel_coef * (tmp_time - times_strain_rate_zeros)
            # ) * (-1) ** (i + 1) * (loading_poly_zeros - unloading_poly_zeros)
            # sum_stresses = np.append(
            #    sum_stresses,
            #    np.exp(-kernel_coef * (tmp_time - times_strain_rate_zeros))
            #    * (-1) ** (i + 1)
            #    * (loading_poly_zeros - unloading_poly_zeros),
            # )
            sum_stresses = (
                np.exp(
                    -kernel_coef * (tmp_time - times_strain_rate_zeros[count_zeros - 1])
                )
                * (-1) ** (count_zeros)
                * (
                    loading_poly_zeros  # [count_zeros - 1]
                    - unloading_poly_zeros  # [count_zeros - 1]
                )
            )
    # Hay que evaluar loading_poly y unloading_poly en los times_strain_rate_zeros
    # sum_stresses = sum_stresses

    if tmp_time == np.round(time_final * 5 / 6):
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        axs[0].set_title(
            f"Time = {times_strain_rate_zeros}s ; strain_zeros = {strain_zeros} N"
        )
        axs[0].plot(times_strain_rate_zeros, sum_stresses, "k-")
        axs[1].plot(tau, kernel, "k-")
        axs[2].plot(tau, integrand, "k-")
        plt.tight_layout()

    # Add all stresses
    # stress = np.append(
    #    stress,
    #    elastic_stress
    # + initial_integral_parts
    # + final_integral_parts
    # + sum_stresses
    # + viscous_stress_integral,
    # )
    stress[i] = (
        elastic_stress
        + viscous_stress_integral
        + initial_integral_parts
        + final_integral_parts
        + sum_stresses
    )

total_time = np.concatenate((time, times_strain_rate_zeros))
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
