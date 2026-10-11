#somatic growth
#switch function test

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
 
def H(x, omega):
    return 0.5 * (1.0 + np.tanh(x / omega))
 
x = np.linspace(-5, 5, 1000)
 
plt.figure(figsize=(8, 5))

for omega in [0.1, 0.5, 1.0, 2.0]:
    plt.plot(x, H(x, omega), label=rf"$\omega$ = {omega}")
 
plt.step(x, np.heaviside(x, 0.5), "k--", where="mid", lw=1, label="Sharp")
 
plt.axhline(0.5, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.xlabel("x")
plt.ylabel("H(x)")
plt.title(r"$H(x) = \frac{1}{2}\left(1 + \tanh(x/\omega)\right)$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#parameters for somatic growth

m_inf = 17576       # Asymptotic body mass (g)
m_mature = 3133     # Body mass at sexual maturity (g)
m0 = 0.001          # Initial body mass (g)

A = 6               # Growth constant (g^0.25/year)

# Calculate the maintenance coefficient

B = A * m_inf**(-1/4)

print("B =", B)


# Von Bertalanffy growth model

def von_bertalanffy(t, m):
    dmdt = A * m**(3/4) - B * m
    return dmdt

t_start = 0
t_end = 100

# Solve the differential equation

solution = solve_ivp(
    von_bertalanffy,
    (t_start, t_end),
    [m0],
    dense_output=True)

# Create an array of ages
t = np.linspace(t_start, t_end, 500)

# body mass at each age
m = solution.sol(t)[0]

#plot growth curve

plt.figure(figsize=(8, 5))

plt.plot(t, m, label="Von Bertalanffy growth")

plt.axhline(
    m_inf,
    linestyle="--",
    label="Asymptotic mass")

plt.xlabel("Age (years)")
plt.ylabel("Body mass (g)")
plt.title("Von Bertalanffy growth of the dusky grouper")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Biphasic growth model

def biphasic_growth(t, m):
    if m[0] < m_mature:
        H = 0
    else:
        H = 1

    dmdt = A * m[0]**(3/4) - H * B * m[0]
    return [dmdt]

# Solve the biphasic growth model

solution_bi = solve_ivp(
    biphasic_growth,
    (t_start, t_end),
    [m0],
    dense_output=True,
    max_step=0.1)

m_bi = solution_bi.sol(t)[0]

#plotting both models together to compare

plt.figure(figsize=(8, 5))

plt.plot(t, m, label="Von Bertalanffy growth")
plt.plot(t, m_bi, label="Biphasic growth")

plt.axhline(
    m_inf,
    linestyle="--",
    label="Asymptotic mass")
plt.axhline(
    m_mature,
    linestyle=":",
    label="Mass at maturity")

plt.xlabel("Age (years)")
plt.ylabel("Body mass (g)")
plt.title("Comparison of growth models")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#extensioon - feeding levels and metbolic costs

f_c = 0.2           # Critical feeding level
f_normal = 0.7      # Reference feeding level

# Calibrate epsilon*h from the baseline relationship:
# A = epsilon*h*(f - f_c)

epsilon_h = A / (f_normal - f_c)

# Keep B fixed to isolate the effects of feeding
B = A * m_inf**(-1/4)

# Illustrative feeding scenarios
feeding_levels = {
    "Low food (f=0.6)": 0.6,
    "Normal food (f=0.7)": 0.7,
    "High food (f=0.8)": 0.8}

# Assumed metabolic feeding-cost coefficient
D = 1.5

# Simulation time (years)
t_start = 0
t_end = 100
t = np.linspace(t_start, t_end, 1000)

# Calculate A for each feeding level

A_values = {}

for scenario, f in feeding_levels.items():
    A_values[scenario] = epsilon_h * (f - f_c)

print("Baseline B =", round(B, 4), "per year")
print("\nFeeding scenarios:")

for scenario, f in feeding_levels.items():
    A = A_values[scenario]
    A_net = A - D * f

    print(
        scenario,
        "| A =", round(A, 3),
        "| A with feeding cost =", round(A_net, 3))


# Define both growth models

def growth_model(t, m, A_eff, model_type):
    mass = max(m[0], 0)

    # Von Bertalanffy: maintenance term always active
    if model_type == "Von Bertalanffy":
        H = 1

    # Biphasic: maintenance term active after maturity
    elif model_type == "Biphasic":
        H = 0 if mass < m_mature else 1

    dmdt = A_eff * mass**(3/4) - H * B * mass

    return [dmdt]


# Solve 

results = {}

for cost_condition in ["Without feeding cost",
                       "With feeding cost"]:

    results[cost_condition] = {}

    for scenario, f in feeding_levels.items():

        A = A_values[scenario]

        # Subtract the metabolic cost when included
        if cost_condition == "With feeding cost":
            A_eff = A - D * f
        else:
            A_eff = A

        for model_type in ["Von Bertalanffy", "Biphasic"]:

            solution = solve_ivp(
                lambda time, mass: growth_model(
                    time, mass, A_eff, model_type
                ),
                (t_start, t_end),
                [m0],
                t_eval=t,
                max_step=0.1)

            results[cost_condition][
                (scenario, model_type)
            ] = solution.y[0]



# Plot the comparisons

colors = {
    "Low food (f=0.6)": "tab:blue",
    "Normal food (f=0.7)": "tab:orange",
    "High food (f=0.8)": "tab:green"}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, cost_condition in zip(
    axes,
    ["Without feeding cost", "With feeding cost"]):

    for scenario in feeding_levels:

        # Solid line: von Bertalanffy
        ax.plot(
            t,
            results[cost_condition][
                (scenario, "Von Bertalanffy")
            ],
            color=colors[scenario],
            linestyle="-",
            label=f"{scenario} - VB")

        # Dashed line: biphasic
        ax.plot(
            t,
            results[cost_condition][
                (scenario, "Biphasic")
            ],
            color=colors[scenario],
            linestyle="--",
            label=f"{scenario} - Biphasic")

    ax.set_title(cost_condition)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Body mass (g)")
    ax.grid(True)
    ax.legend(fontsize=8)

plt.suptitle("Effects of Feeding Level and Feeding Costs on Grouper Growth")
plt.tight_layout()
plt.show()

#m_inf for the different scenarios

print("\nAsymptotic mass under all conditions")
print("-" * 65)
print(f"{'Food scenario':<22} {'Without cost (g)':>20} {'With cost (g)':>18}")

for scenario, f in feeding_levels.items():
    A = A_values[scenario]
    A_net = A - D * f

    m_inf_no_cost = (A / B)**4
    m_inf_with_cost = (A_net / B)**4 if A_net > 0 else 0

    print(f"{scenario:<22} {m_inf_no_cost:>20.1f} {m_inf_with_cost:>18.1f}")
