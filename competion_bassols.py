# Predation model

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#parameters
#initial populations

R0 = 350000  #aprox half of population used in the competition model - ind
PB0 = 2000 #ind
O0 = 3000 #ind

y0 = [R0, PB0, O0]

#resource parameters
r_in = 0.1 #yr^-1
K = 1000000 #ind

# Polar Bears parameters
qmax_PB = 90 # seals / polar bear / year
b_PB  = 0.00001 #clearance rate
epsilon_PB = 0.001 # efficency
d_PB = 0.0667 #yr^-1

# orcas parameters
qmax_O = 730 #seals / orca / year
b_O = 0.00002 #clearance rate
epsilon_O = 0.0001 # efficency
d_O = 0.02 #yr^-1 

## type 2 functional response for consumption rate
def consumption_rate(R, qmax, b):

    return qmax * (b * R) / (b * R + qmax)

# competion model
def competition_model(t, y):
    R, PB, O = y

    q_PB = consumption_rate(R, qmax_PB, b_PB)
    q_O = consumption_rate(R, qmax_O, b_O)

    dRdt = r_in * (K - R) - q_PB * PB - q_O * O

    dPBdt = (epsilon_PB * q_PB - d_PB) * PB

    dOdt = (epsilon_O * q_O - d_O) * O

    return [dRdt, dPBdt, dOdt]

t_start = 0
t_end = 200 

t_eval = np.linspace(t_start, t_end, 201*12*4) # 201 years (0-100), monthly intervals and 4 points per month

solution = solve_ivp(
    competition_model,
    [t_start,t_end],
    y0,
    t_eval = t_eval)

R = solution.y[0]
PB = solution.y[1]
O = solution.y[2]

time = solution.t

#long term
print("\n--- Baseline numerical long-term state ---")

print("Final seal population =", R[-1])
print("Final polar bear population =", PB[-1])
print("Final orca population =", O[-1])

print("\nDifference between final seals and K =", R[-1] - K)

#plot all populations

plt.figure()

plt.plot(time, R, label="Seals")
plt.plot(time, PB, label="Polar bears")
plt.plot(time, O, label="Orcas")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Baseline competition model")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#Resoruce plot

plt.figure()
plt.plot(time, R)

plt.xlabel("Time (years)")
plt.ylabel("Seal population")
plt.title("Seal population dynamics")

plt.grid(alpha=0.3)
plt.show()

#consumers plot
plt.figure()

plt.plot(time, PB, label="Polar bears")
plt.plot(time, O, label="Orcas")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Consumer competition")

plt.legend()
plt.grid(alpha=0.3)

plt.show()

#R* analysis

def R_star(qmax, b, epsilon, d):

    return (1 / b) * (
        qmax * d
        / (qmax * epsilon - d))

Rstar_PB = R_star(
    qmax_PB,
    b_PB,
    epsilon_PB,
    d_PB)

Rstar_O = R_star(
    qmax_O,
    b_O,
    epsilon_O,
    d_O)

print("Polar bear R* =", Rstar_PB)
print("Orca R* =", Rstar_O)

print("\n--- Long-term analytical prediction ---")

print("Carrying/resource limit K =", K)
print("Polar bear R* =", Rstar_PB)
print("Orca R* =", Rstar_O)

#lower R* better competitor under reosurce limitaion

#inital growth rates at R0
q_PB_0 = consumption_rate(R0, qmax_PB, b_PB)
q_O_0 = consumption_rate(R0, qmax_O, b_O)

growth_PB_R0 = epsilon_PB * q_PB_0 - d_PB
growth_O_R0 = epsilon_O * q_O_0 - d_O

print("\n--- Growth rates at actual initial resource level ---")
print("Polar bear consumption at R0 =", q_PB_0)
print("Orca consumption at R0 =", q_O_0)

print("Polar bear growth rate at R0 =", growth_PB_R0)
print("Orca growth rate at R0 =", growth_O_R0)

#analyticla comparidon

R_values = np.linspace(0, K, 500)

q_PB_values = consumption_rate(R_values, qmax_PB, b_PB)
q_O_values = consumption_rate(R_values, qmax_O, b_O)

plt.figure(figsize=(10, 6))

plt.plot(R_values, q_PB_values, label="Polar bears")
plt.plot(R_values, q_O_values, label="Orcas")

plt.xlabel("Seal population")
plt.ylabel("Consumption rate")
plt.title("Type II functional responses")

plt.legend()
plt.grid(alpha=0.3)

plt.show()

#extension model improving efficency and clearance rate based on the rseults from the b aseline and add (assumption) some predation from orcas to polar 

#better acces to seals and better effincy of the consumed seals
# Improved clearance rates
b_PB_ext = 0.00003
b_O_ext = 0.00004

# Improved conversion efficiencies
epsilon_PB_ext = 0.005
epsilon_O_ext = 0.0005

#new competition

def improved_competition_model(t, y):
    R, PB, O = y

    q_PB = consumption_rate(R, qmax_PB, b_PB_ext)
    q_O = consumption_rate(R, qmax_O, b_O_ext)

    dRdt = (
        r_in * (K - R)
        - q_PB * PB
        - q_O * O)

    dPBdt = (
        epsilon_PB_ext * q_PB * PB
        - d_PB * PB)

    dOdt = (
        epsilon_O_ext * q_O * O
        - d_O * O)

    return [dRdt, dPBdt, dOdt]

# solving the model

solution_ext = solve_ivp(improved_competition_model, [t_start, t_end], y0, t_eval=t_eval)
R_ext = solution_ext.y[0]
PB_ext = solution_ext.y[1]
O_ext = solution_ext.y[2]

time_ext = solution_ext.t

print("\n--- Extension 1 numerical long-term state ---")

print("Final seals =", R_ext[-1])
print("Final polar bears =", PB_ext[-1])
print("Final orcas =", O_ext[-1])

#plton populations

plt.figure(figsize=(10, 6))

plt.plot(time, R, label="Seals - baseline")
plt.plot(time, PB, label="Polar bears - baseline")
plt.plot(time, O, label="Orcas - baseline")

plt.plot(time_ext, R_ext, "--", label="Seals - improved performance")
plt.plot(time_ext, PB_ext, "--", label="Polar bears - improved performance")
plt.plot(time_ext, O_ext, "--", label="Orcas - improved performance")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Baseline vs improved consumer performance")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#compare R*

Rstar_PB_ext = R_star(
    qmax_PB,
    b_PB_ext,
    epsilon_PB_ext,
    d_PB)

Rstar_O_ext = R_star(
    qmax_O,
    b_O_ext,
    epsilon_O_ext,
    d_O)

print("\nR* comparison")
print("Polar bears - baseline:", Rstar_PB)
print("Polar bears - extension:", Rstar_PB_ext)

print("Orcas - baseline:", Rstar_O)
print("Orcas - extension:", Rstar_O_ext)

print("\n--- Extension 1 long-term prediction ---")


# adding orcas predating polar bears
# Orca predation on polar bears
qmax_O_PB = 0.2
b_O_PB = 0.00001
epsilon_O_PB = 0.01

# Logistic growth in seals 
r = 0.1
K = 1000000

# EXTENSION: improved consumer performance + orca predation

def extended_model(t, y):

    R, PB, O = y

    # Consumption of seals
    q_PB = consumption_rate(R, qmax_PB, b_PB_ext)
    q_O = consumption_rate(R, qmax_O, b_O_ext)

    # Orca predation on polar bears
    q_O_PB = consumption_rate(PB, qmax_O_PB, b_O_PB)

    # Seal dynamics: logistic growth
    dRdt = (
        r * R * (1 - R / K)
        - q_PB * PB
        - q_O * O)

    dPBdt = (
        epsilon_PB_ext * q_PB * PB
        - d_PB * PB
        - q_O_PB * O)

    dOdt = (
        epsilon_O_ext * q_O * O
        - d_O * O
        + epsilon_O_PB * q_O_PB * O)

    return [dRdt, dPBdt, dOdt]

solution_final = solve_ivp(
    extended_model,
    [t_start, t_end],
    y0,
    t_eval=t_eval
)

R_final = solution_final.y[0]
PB_final = solution_final.y[1]
O_final = solution_final.y[2]

time_final = solution_final.t

#plot only extension

plt.figure(figsize=(10, 6))

plt.plot(time_final, R_final, label="Seals")
plt.plot(time_final, PB_final, label="Polar bears")
plt.plot(time_final, O_final, label="Orcas")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Extended model")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#comparing extension and baseline
plt.figure(figsize=(10, 6))

# Baseline
plt.plot(time, R, label="Seals - baseline")
plt.plot(time, PB, label="Polar bears - baseline")
plt.plot(time, O, label="Orcas - baseline")

# Extension
plt.plot(
    time_final, R_final, "--",
    label="Seals - extension")

plt.plot(
    time_final, PB_final, "--",
    label="Polar bears - extension")

plt.plot(
    time_final, O_final, "--",
    label="Orcas - extension")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Baseline vs extended model")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#predator comparison

plt.figure(figsize=(10, 6))

plt.plot(
    time, PB,
    label="Polar bears - baseline")

plt.plot(
    time, O,
    label="Orcas - baseline")

plt.plot(
    time_final, PB_final, "--",
    label="Polar bears - extension")

plt.plot(
    time_final, O_final, "--",
    label="Orcas - extension")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Consumer dynamics: baseline vs extension")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#seals plot

plt.figure(figsize=(10, 6))

plt.plot(
    time, R,
    label="Baseline")

plt.plot(
    time_final, R_final, "--",
    label="Extension")

plt.xlabel("Time (years)")
plt.ylabel("Seal population")
plt.title("Seal dynamics: baseline vs extension")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#plot rate orcas consumption on polar bear

q_O_PB_final = consumption_rate(
    PB_final,
    qmax_O_PB,
    b_O_PB)

plt.figure(figsize=(10, 6))

plt.plot(time_final, q_O_PB_final)

plt.xlabel("Time (years)")
plt.ylabel("Polar bears consumed per orca per year")
plt.title("Orca predation rate on polar bears")

plt.grid(alpha=0.3)

plt.show()