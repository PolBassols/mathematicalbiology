
# Stage-structured population model: gilthead seabream

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# parameters and baseline projection matrix

s_J = 0.4       # Annual juvenile survival
m_J = 0.80      # Probability of juvenile maturation into males
s_M = 0.6       # Annual male survival
s_F = 0.6       # Annual female survival
c = 0.5         # Probability of male-to-female sex change
f = 3.0         # New recruits per female (assumption)

# stages juveniles, males, females
J0, M0, F0 = 500, 300, 200
population_initial = np.array([J0, M0, F0], dtype=float)

A = np.array([
    [s_J * (1 - m_J), 0, f],
    [s_J * m_J, s_M * (1 - c), 0],
    [0, s_M * c, s_F]
])

print("Projection matrix:")
print(A)

# baseline simuliation and eigenvalue analysis
years = 100
time = np.arange(years + 1)

population = np.zeros((years + 1, 3))
population[0] = population_initial

for t in range(years):
    population[t + 1] = A @ population[t]

total_population = population.sum(axis=1)

# Dominant eigenvalue and stable stage distribution
eigenvalues, eigenvectors = np.linalg.eig(A)
idx = np.argmax(eigenvalues.real)

lambda_dom = eigenvalues[idx].real
stable_structure = eigenvectors[:, idx].real

if stable_structure.sum() < 0:
    stable_structure = -stable_structure

stable_structure /= stable_structure.sum()
r = np.log(lambda_dom)

print(f"\nDominant eigenvalue: {lambda_dom:.4f}")
print(f"Long-term growth rate: {r:.4f} per year")
print(f"Population after {years} years: {total_population[-1]:.2f}")

print("\nStable stage distribution:")
print(f"Juveniles: {stable_structure[0]:.1%}")
print(f"Males:     {stable_structure[1]:.1%}")
print(f"Females:   {stable_structure[2]:.1%}")

# Plot baseline population dynamics
plt.figure(figsize=(8, 5))
plt.plot(time, population[:, 0], label="Juveniles")
plt.plot(time, population[:, 1], label="Males")
plt.plot(time, population[:, 2], label="Females")
plt.plot(time, total_population, "--", label="Total")

plt.xlabel("Year")
plt.ylabel("Number of fish")
plt.title("Baseline population dynamics")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Compare theoretical and simulated growth rates
start_year = 10

slope, intercept = np.polyfit(
    time[start_year:],
    np.log(total_population[start_year:]),
    1)

print(f"Theoretical growth rate: {r:.4f}")
print(f"Simulated growth rate:   {slope:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(time, np.log(total_population), label="Simulated log abundance")
plt.plot(
    time[start_year:],
    intercept + slope * time[start_year:],
    "--",
    label="Linear fit"
)

plt.xlabel("Year")
plt.ylabel("Log total abundance")
plt.title("Theoretical vs. simulated growth")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# density-dependent reproduction (Beverton-Holt model)

f0 = 3.0
a = 0.001

population_dd = np.zeros((years + 1, 3))
population_dd[0] = population_initial

fertility = np.zeros(years + 1)

for t in range(years):
    J, M, F = population_dd[t]

    # Spawning adults
    N_adults = M + F

    # Beverton-Holt fertility
    fertility[t] = f0 / (1 + a * N_adults)

    # Update population stages
    population_dd[t + 1, 0] = (
        s_J * (1 - m_J) * J + fertility[t] * F)

    population_dd[t + 1, 1] = (
        s_J * m_J * J + s_M * (1 - c) * M)

    population_dd[t + 1, 2] = (
        s_M * c * M + s_F * F)

N_adults_final = population_dd[-1, 1] + population_dd[-1, 2]
fertility[-1] = f0 / (1 + a * N_adults_final)

total_dd = population_dd.sum(axis=1)

print("\nDensity-dependent model:")
print(f"Population after {years} years: {total_dd[-1]:.2f}")
print(f"Initial fertility: {fertility[0]:.4f}")
print(f"Final fertility: {fertility[-1]:.4f}")

# Compare the two models
plt.figure(figsize=(8, 5))
plt.plot(time, total_population, label="Linear model")
plt.plot(time, total_dd, label="Beverton-Holt model")

plt.xlabel("Year")
plt.ylabel("Total population")
plt.title("Effect of density-dependent reproduction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# equilibirum and emergent carrying capacity
def equilibrium_equations(state):
    J, M, F = state

    fertility_now = f0 / (1 + a * (M + F))

    next_J = s_J * (1 - m_J) * J + fertility_now * F
    next_M = s_J * m_J * J + s_M * (1 - c) * M
    next_F = s_M * c * M + s_F * F

    return [
        next_J - J,
        next_M - M,
        next_F - F
    ]

result = root(equilibrium_equations, [100, 50, 40])

print("\nEquilibrium calculation:")
print("Solver converged:", result.success)
print("Equilibrium stages:", result.x)
print("Total equilibrium population:", result.x.sum())
print("Residuals:", equilibrium_equations(result.x))


# extension - fishing scenario 5% of males and 10%females removed annually

q_M = 0.05  # Annual fraction of males removed
q_F = 0.10  # Annual fraction of females removed
s_J_fished = 0.5# Higher juvenile survival in the fishing scenario


population_fished = np.zeros((years + 1, 3))
population_fished[0] = population_initial

for t in range(years):
    J, M, F = population_fished[t]

    # Juveniles: no fishing
    next_J = s_J_fished * (1 - m_J) * J + f * F

    # Males: natural transitions, then 5% removed
    next_M = (
        s_J_fished * m_J * J
        + s_M * (1 - c) * M
    ) * (1 - q_M)

    # Females: natural transitions, then 10% removed
    next_F = (
        s_M * c * M
        + s_F * F
    ) * (1 - q_F)

    population_fished[t + 1] = [next_J, next_M, next_F]

total_fished = population_fished.sum(axis=1)

print("\nFishing scenario:")
print(f"Annual male removal:   {q_M:.0%}")
print(f"Annual female removal: {q_F:.0%}")
print(f"Initial population: {total_fished[0]:.0f}")
print(f"Population after {years} years: {total_fished[-1]:.2f}")

# Compare population trajectories
plt.figure(figsize=(8, 5))
plt.plot(time, total_population, label="No fishing")
plt.plot(time, total_fished, label="Fishing: 5% males, 10% females")

plt.xlabel("Year")
plt.ylabel("Total population")
plt.title("Effect of sex-specific fishing mortality")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#check how fishing affect juveniles

plt.figure(figsize=(8, 5))

plt.plot(time, population[:, 0], label="Without fishing")
plt.plot(time, population_fished[:, 0], label="With fishing")

plt.xlabel("Year")
plt.ylabel("Number of juveniles")
plt.title("Effect of fishing on the juvenile population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()