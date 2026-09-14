import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Parameters

S0 = 2200
I0 = 200
R0 = 22

b = 0.0001 # contact rate yr-1 infected-1
a = 0.2 # recovery rate

# simulation time
t_0 = 0
t_end = 80

# differential equations

def SIR_model(t, y):
    S, I, R = y
    dSdt = -b * S * I
    dIdt = b * S * I - a * I
    dRdt = a * I
    return [dSdt, dIdt, dRdt]

#solving the function
# add time points for the simulation
t = np.linspace(t_0, t_end, 81*24) #define steps, 2 steps per month per year simulated

solution = solve_ivp(
    SIR_model,
    [t_0, t_end],
    [S0, I0, R0],
    t_eval=t
)

S = solution.y[0]
I = solution.y[1]
R = solution.y[2]

#plot SIR over time

plt.figure()

plt.plot(t, S, label="Susceptible (S)")
plt.plot(t, I, label="Infected (I)")
plt.plot(t, R, label="Recovered (R)")

plt.xlabel("Time [years]")
plt.ylabel("Number of individuals")
plt.title("SIR model – Pinna nobilis")
plt.legend()
plt.grid()

plt.show()

# plot S vs I - phase plane

plt.figure()

plt.plot(S, I)

plt.xlabel("Susceptible individuals (S)")
plt.ylabel("Infected individuals (I)")
plt.title("Phase plane – SIR model")

plt.grid()
plt.show()

## results - exercise 2
# baseline scenario

# Initial rate of increase
initial_rate = (b * S[0] - a) * I[0]

# Reproductive number
R_number = b * S[0] / a

# Maximum number of infected
max_infected = np.max(I)
time_max_infected = t[np.argmax(I)]

# Number of susceptible individuals at the end
susceptible_end = S[-1]

# Practical epidemic duration: I < 1
below_1 = np.where(I < 1)[0]

if len(below_1) > 0:
    epidemic_duration = t[below_1[0]]
else:
    epidemic_duration = "Not reached"


print("STANDARD SIR MODEL")
print(f"Initial rate of increase: {initial_rate:.2f} infected/year")
print(f"Reproductive number R0: {R_number:.2f}")
print(f"Maximum infected: {max_infected:.0f} individuals")
print(f"Time of maximum infection: {time_max_infected:.2f} years")
print(f"Susceptible individuals at the end: {susceptible_end:.0f}")
print(f"Epidemic duration (I < 1): {epidemic_duration} years")

# exercise 3: extended model
# Restoration programe during 25 years.
# at 25 years, outbreak of the virus due to heat wave. contact rate x5

#restoration parameters

ind = 2 #susceptible individuals introduced per year
t_rp = 25 #time of restoration program

# heatwave parameter
b_hw = 0.0005 # contact rate yr-1 infected-1 - b_hw=5b

def SIR_model_extended(t, y):
    S, I, R = y
    if t < t_rp:
        dSdt = -b * S * I + ind
        dIdt = b * S * I - a * I
        dRdt = a * I 
    else:
        dSdt = -b_hw * S * I
        dIdt = b_hw * S * I - a * I
        dRdt = a * I
    return [dSdt, dIdt, dRdt]

#solving the extended model
solution_ext = solve_ivp(
    SIR_model_extended,
    [t_0, t_end],
    [S0, I0, R0],
    t_eval=t,
    max_step=0.02
)

S_ext = solution_ext.y[0]
I_ext = solution_ext.y[1]
R_ext = solution_ext.y[2]

#ºplot SIR over time

plt.figure()

plt.plot(t, S_ext, label="Susceptible (S)")
plt.plot(t, I_ext, label="Infected (I)")
plt.plot(t, R_ext, label="Recovered (R)")

plt.axvline(
    25,
    linestyle="--",
    label="Heatwave"
)

plt.xlabel("Time [years]")
plt.ylabel("Number of individuals")
plt.title("Pinna nobilis population under restoration and heatwave")
plt.legend()
plt.grid()

plt.show()

## results exercise 2 for the extended model

# Reproductive number before the heatwave
S_25 = np.interp(t_rp, t, S_ext)

R_before = b * S_25 / a
R_after = b_hw * S_25 / a

# Maximum number of infected
max_infected_ext = np.max(I_ext)
time_max_infected_ext = t[np.argmax(I_ext)]

# Susceptible individuals at the end
susceptible_end_ext = S_ext[-1]

# Practical epidemic duration
below_1_ext = np.where(I_ext < 1)[0]

if len(below_1_ext) > 0:
    epidemic_duration_ext = t[below_1_ext[0]]
else:
    epidemic_duration_ext = "Not reached"


print("EXTENDED SIR MODEL")
print(f"Susceptible individuals at year 25: {S_25:.0f}")
print(f"Epidemic threshold R0 before heatwave: {R_before:.2f}")
print(f"Epidemic threshold R0 after heatwave: {R_after:.2f}")
print(f"Maximum infected: {max_infected_ext:.0f} individuals")
print(f"Time of maximum infection: {time_max_infected_ext:.2f} years")
print(f"Susceptible individuals at the end: {susceptible_end_ext:.0f}")
print(f"Epidemic duration (I < 1): {epidemic_duration_ext} years")


