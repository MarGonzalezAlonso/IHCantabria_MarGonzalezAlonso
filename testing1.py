import numpy as np
import matplotlib.pyplot as plt

## Define material parameters
# Static strain-stress curve
static_curve_strain = np.array([0, 1])
static_curve_stress = np.array([0, 1e7])
# Memory kernel coefficient
kernel_coef = 1
# Cubic viscoelastic coefficients
visc_loading_coef = np.array([-5e6, 0, 0])
visc_unloading_coef = np.array([-5e6, 0, 0])

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
strain_zeros = np.array([])
sum_stresses = np.array([])
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
    tau = np.append(tau, tmp_time)
    times_strain_rate_zeros = np.append(times_strain_rate_zeros, 0)
    # strain_zeros = np.append(strain_zeros, 0)
    sum_stresses = np.append(sum_stresses, 0)
    if i > 0 and i < num_points - 1:
        if time[i - 1] - tmp_time < 0 or tmp_time - time[i - 1] < 0:
            # times_strain_rate_zeros[i - 1] = (
            #    -strain_rate[i - 1] * time_step / (tmp_strain_rate - strain_rate[i - 1])
            #    + tmp_time
            # )
            times_strain_rate_zeros = np.append(
                times_strain_rate_zeros,
                -strain_rate[i - 1] * time_step / (tmp_strain_rate - strain_rate[i - 1])
                + tmp_time,
            )
            strain_zeros = np.interp(times_strain_rate_zeros, time, strain_rate)

        for j in range(3):
            loading_poly_zeros = strain_zeros ** (j + 1) * visc_loading_coef[j]
            unloading_poly_zeros = strain_zeros ** (j + 1) * visc_unloading_coef[j]

        # sum_stresses = np.append(sum_stresses, 0)
        sum_stresses = np.append(
            sum_stresses,
            sum_stresses[i - 1]
            + np.exp(-kernel_coef * (tmp_time - times_strain_rate_zeros))
            * (-1) ** (i + 1)
            * (loading_poly_zeros - unloading_poly_zeros),
        )


# print("times_strain_rate_zeros {}".format(times_strain_rate_zeros))
# strain_zeros
