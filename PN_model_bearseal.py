# Predator (P) - Prey (N) model simulation using Lotka-Volterra equations
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

# Parameters
# Initial populations
P0 = 2275          # initial polar bear population
N0 = 733871        # initial ringed seal population

# Model parameters
r = 0.1            # intrinsic growth rate of seals (1/year)
b = 9.947e-5       # clearance/attack rate
epsilon = 9.14e-4  # reproductive efficiency
d = 0.0667         # polar bear mortality rate (1/year)

# Analytical equilibrium
N_eq = d / (epsilon * b)
P_eq = r / b

print("Equilibrium seal population:", N_eq)
print("Equilibrium polar bear population:", P_eq)

def lotka_volterra(t, y):
    
    N, P = y
    
    dNdt = r * N - b * P * N
    dPdt = epsilon * b * N * P - d * P
    
    return [dNdt, dPdt]

#simulatio period
t_start = 0
t_end = 200

t_eval = np.linspace(t_start, t_end, 101*12*4) # 101 years (0-100), monthly intervals and 4 points per month

solution = solve_ivp(lotka_volterra, [t_start, t_end], [N0, P0], t_eval=t_eval)

N = solution.y[0]
P = solution.y[1]

#plotting differential equations over time

plt.figure()

plt.plot(solution.t, N, label="Ringed seals")
plt.plot(solution.t, P, label="Polar bears")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Baseline Lotka–Volterra model")
plt.legend()
plt.grid()

plt.show()


#plotting numerical equation and equilibrium - AI help

fig, ax = plt.subplots(2, 1, sharex=True)

# Seal population
ax[0].plot(solution.t, N, label="Ringed seals")
ax[0].axhline(N_eq, linestyle="--", label="Seal equilibrium")

ax[0].set_ylabel("Seals")
ax[0].set_title("Baseline model: prey population")
ax[0].legend()
ax[0].grid()

# Polar bear population
ax[1].plot(solution.t, P, label="Polar bears")
ax[1].axhline(P_eq, linestyle="--", label="Polar bear equilibrium")

ax[1].set_xlabel("Time (years)")
ax[1].set_ylabel("Polar bears")
ax[1].set_title("Baseline model: predator population")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

#plotting phase plane (N vs P)

plt.figure()

plt.plot(N, P)

plt.scatter(N_eq, P_eq, label="Equilibrium")

plt.xlabel("Ringed seals (N)")
plt.ylabel("Polar bears (P)")
plt.title("Phase plot – Baseline Lotka–Volterra model")
plt.legend()
plt.grid()

plt.show()

# Extended model: logistic growth for prey (N) with carrying capacity K and  removal for both predator and prey

# Extended-model parameters
K = 1223118        # carrying capacity of ringed seals
H_P = 100            # polar bears removed by humans per year
H_N = 8000           # ringed seals removed by humans per year


# Differential equations for the extended model

def extended_model(t, y):
    
    N, P = y
    
    dNdt = r * N * (1 - N / K) - b * P * N - H_N
    dPdt = epsilon * b * N * P - d * P - H_P
    
    return [dNdt, dPdt]

# solving differential equations 

t_start = 0
t_end = 200

t_eval = np.linspace(t_start, t_end, 101*12*4) # 101 years (0-100), monthly intervals and 4 points per month

solution_ext = solve_ivp(extended_model, [t_start, t_end], [N0, P0], t_eval=t_eval)
N_ext = solution_ext.y[0]
P_ext = solution_ext.y[1]

#plotitng
plt.figure()

plt.plot(solution_ext.t, N_ext, label="Ringed seals")
plt.plot(solution_ext.t, P_ext, label="Polar bears")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Extended predator-prey model")
plt.legend()
plt.grid()

plt.show()

# Numerical equilibrium for the extended model
from scipy.optimize import root #debbugged bc it was not wokring

def eq_equations(z):
    
    N, P = z
    
    eq1 = r * N * (1 - N / K) - b * P * N - H_N
    eq2 = epsilon * b * N * P - d * P - H_P
    
    return [eq1, eq2]

equilibrium = root(
    eq_equations,
    [N0, P0])

N_eq_ext = equilibrium.x[0]
P_eq_ext = equilibrium.x[1]

print("Extended equilibrium seals:", N_eq_ext)
print("Extended equilibrium polar bears:", P_eq_ext)

#ploting equilibrium and populatiois

fig, ax = plt.subplots(2, 1, sharex=True)

# Seal population
ax[0].plot(solution_ext.t, N_ext, label="Ringed seals")
ax[0].axhline(
    N_eq_ext,
    linestyle="--",
    label="Seal equilibrium"
)

ax[0].set_ylabel("Seals")
ax[0].set_title("Extended model: prey population")
ax[0].legend()
ax[0].grid()

# Polar bear population
ax[1].plot(solution_ext.t, P_ext, label="Polar bears")
ax[1].axhline(
    P_eq_ext,
    linestyle="--",
    label="Polar bear equilibrium"
)

ax[1].set_xlabel("Time (years)")
ax[1].set_ylabel("Polar bears")
ax[1].set_title("Extended model: predator population")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

#phase plot

plt.figure()

plt.plot(N_ext, P_ext)

plt.scatter(
    N_eq_ext,
    P_eq_ext,
    label="Equilibrium"
)

plt.xlabel("Ringed seals (N)")
plt.ylabel("Polar bears (P)")
plt.title("Phase plot – Extended predator-prey model")
plt.legend()
plt.grid()

plt.show()

# extended version 2. Proportional removal of both predator and prey populations and threshold for the predator population

# Predator threshold
P_threshold = 500

# proportional removal rate
rr_hp = H_P / P0
rr_hn = H_N / N0

print("Ringed seal removal rate:", rr_hn)
print("Polar bear removal rate:", rr_hp)

# differential equations for the extended model 2

def extended_model2(t, y):
    
    N, P = y
    
    # Seal dynamics
    dNdt = (
        r * N * (1 - N / K)
        - b * P * N
        - rr_hn * N
    )
    
    # Polar bear dynamics
    if P > P_threshold:
        dPdt = (
            epsilon * b * N * P
            - d * P
            - rr_hp * P
        )
    else:
        dPdt = (
            epsilon * b * N * P
            - d * P
        )
    
    return [dNdt, dPdt]

# solving differential equations 

t_start = 0
t_end = 200

t_eval = np.linspace(t_start, t_end, 101*12*4) # 101 years (0-100), monthly intervals and 4 points per month

solution_ext2 = solve_ivp(extended_model2, [t_start, t_end], [N0, P0], t_eval=t_eval)
N_ext2 = solution_ext2.y[0]
P_ext2 = solution_ext2.y[1]


# each population separately
plt.figure()

plt.plot(solution_ext2.t, N_ext2, label="Ringed seals")
plt.plot(solution_ext2.t, P_ext2, label="Polar bears")

plt.xlabel("Time (years)")
plt.ylabel("Population")
plt.title("Extended predator-prey model")
plt.legend()
plt.grid()

plt.show()

fig, ax = plt.subplots(2, 1, sharex=True)

# Ringed seals
ax[0].plot(solution_ext2.t, N_ext2, label="Ringed seals")

ax[0].set_ylabel("Seals")
ax[0].set_title("Extended model: prey population")
ax[0].legend()
ax[0].grid()

# Polar bears
ax[1].plot(solution_ext2.t, P_ext2, label="Polar bears")

ax[1].axhline(
    P_threshold,
    linestyle="--",
    label="500-bear threshold"
)

ax[1].set_xlabel("Time (years)")
ax[1].set_ylabel("Polar bears")
ax[1].set_title("Extended model: predator population")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()

# phase plot extended model 2
plt.figure()

plt.plot(N_ext2, P_ext2)

plt.axhline(
    P_threshold,
    linestyle="--",
    label="500-bear threshold"
)

plt.xlabel("Ringed seals (N)")
plt.ylabel("Polar bears (P)")
plt.title("Phase plot – Extended predator-prey model")
plt.legend()
plt.grid()

plt.show()