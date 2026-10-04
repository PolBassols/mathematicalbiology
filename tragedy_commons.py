# Tragedy of the commons in fisheries management

#COMMENTS FROM CLASS
#run to the full equilibrium
#transient - first part where it changes
#interest in the steady state
#


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#question 1

#speceis : aristeus antennatus
#parameters
N0 = 2194 #initial biommas in tonnes

K = 3656 # carrying capacity in tonnes

r = 0.226 # intrinsic growth rate in 1/year

#question 2

E = 0.10        # Constant fishing effort 1/year

# population model

def population_model(t, N):
    dNdt = r * N * (1 - N / K) - E * N
    return dNdt

#solution of the differential equation

t_span = (0, 100)  # time span in years
t_eval = np.linspace(t_span[0], t_span[1], 100)

N0_constant_effort = K

solution = solve_ivp(
    population_model,
    t_span,
    [N0_constant_effort],
    t_eval=t_eval
)
N = solution.y[0]

#plotting the results
plt.figure(figsize=(8, 5))

plt.plot(t_eval, N, label="Shrimp biomass")

plt.axhline(K, linestyle="--", label="Carrying capacity")

plt.xlabel("Time (years)")
plt.ylabel("Biomass (tonnes)")
plt.title("Red shrimp biomass under constant fishing effort")

plt.legend()
plt.grid(True)

plt.show()

#question 3: yield vary in time

Y = E * N  # yield in tonnes/year

print(f"Initial biomass: {N[0]:.2f} tonnes")
print(f"Initial yield: {Y[0]:.2f} tonnes/year")

print(f"Final biomass: {N[-1]:.2f} tonnes")
print(f"Final yield: {Y[-1]:.2f} tonnes/year")

#question 4 - numerical vs analytical equilibrium
# analytical equilibrium biomass
N_eq = K * (1 - E / r)    
Y_eq = E * N_eq  # analytical equilibrium yield

print(f"Analytical equilibrium biomass: {N_eq:.2f} tonnes")
print(f"Analytical equilibrium yield: {Y_eq:.2f} tonnes/year")

# compare with numerical equilibrium
N_numerical_eq = N[-1]
Y_numerical_eq = Y[-1]

print(f"Numerical equilibrium biomass: {N_numerical_eq:.2f} tonnes")
print(f"Numerical equilibrium yield: {Y_numerical_eq:.2f} tonnes/year")

print(f"Difference in biomass: {abs(N_eq - N_numerical_eq):.2f} tonnes")
print(f"Difference in yield: {abs(Y_eq - Y_numerical_eq):.2f} tonnes/year")


#plot
plt.figure(figsize=(8, 5))

plt.plot(t_eval, Y, label="Yield")

plt.xlabel("Time (years)")
plt.ylabel("Yield (tonnes/year)")
plt.title("Red shrimp yield under constant fishing effort")

plt.axhline(Y_eq, linestyle="--", label="Analytical equilibrium yield")

plt.legend()
plt.grid(True)

plt.show()

#question 5 - freely exploitable poop

#parameters for the open access model

# Fishing agents
A0 = 22            # Initial number of vessels

# Exploitation
b = 0.003          # Clearance rate, vessel^-1 yr^-1
p = 23000          # Price, €/tonne
c = 120000         # Cost, €/vessel/year
k = 1              # Entry rate, vessels/year

#equation
def open_access_model(t, y):

    N, A = y

    # consider for negative effort
    if A <= 0 and (p * b * N / c - 1) < 0:
        dAdt = 0
    else:
        dAdt = k * (p * b * N / c - 1)

    dNdt = r * N * (1 - N / K) - b * A * N

    return [dNdt, dAdt]

t_span = (0, 100)  # time span in years

solution = solve_ivp(
    open_access_model,
    t_span,
    [N0, A0],
    dense_output=True
)

t = np.linspace(0, 100, 1000)

N = solution.sol(t)[0]
A = solution.sol(t)[1]

#plot population and number of vessels

plt.figure(figsize=(8, 5))

plt.plot(t, N)

plt.xlabel("Time (years)")
plt.ylabel("Biomass (tonnes)")
plt.title("Red shrimp biomass under open access")
plt.grid(True)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(t, A)

plt.xlabel("Time (years)")
plt.ylabel("Number of vessels")
plt.title("Fishing vessels under open access")
plt.grid(True)

plt.show()

#question 6 - analytical equilibrium for open access

# Equilibrium biomass under open access
N_eq = c / (p * b)

# Equilibrium effort
E_eq = r * (1 - N_eq / K)

# Equilibrium number of agents
A_eq = E_eq / b

# Equilibrium yield
Y_eq = E_eq * N_eq

# MSY
N_MSY = K / 2
E_MSY = r / 2
Y_MSY = r * K / 4

print("\n===== OPEN-ACCESS EQUILIBRIUM =====")
print(f"Equilibrium biomass = {N_eq:.2f} tonnes")
print(f"Equilibrium effort  = {E_eq:.4f} yr^-1")
print(f"Equilibrium vessels = {A_eq:.2f}")
print(f"Equilibrium yield   = {Y_eq:.2f} tonnes/year")

print("\n===== MSY =====")
print(f"MSY biomass = {N_MSY:.2f} tonnes")
print(f"MSY effort  = {E_MSY:.4f} yr^-1")
print(f"MSY yield   = {Y_MSY:.2f} tonnes/year")

#question 7 - numerical solution vs analytical solution
#effort
E = b * A

plt.figure(figsize=(8, 5))

plt.plot(t, E)

plt.xlabel("Time (years)")
plt.ylabel("Fishing effort (1/year)")
plt.title("Fishing effort under open access")
plt.grid(True)

plt.show()

#yield
Y = b * A * N

plt.figure(figsize=(8, 5))

plt.plot(t, Y)

plt.xlabel("Time (years)")
plt.ylabel("Yield (tonnes/year)")
plt.title("Yield under open access")
plt.grid(True)

plt.show()

#Revenue
R = p * Y

plt.figure(figsize=(8, 5))

plt.plot(t, R)

plt.xlabel("Time (years)")
plt.ylabel("Revenue (€/year)")
plt.title("Revenue under open access")
plt.grid(True)

plt.show()

#profit
profit = p * b * N - c

plt.figure(figsize=(8, 5))

plt.plot(t, profit)

plt.xlabel("Time (years)")
plt.ylabel("Profit per vessel (€/year)")
plt.title("Profit per vessel under open access")
plt.grid(True)

plt.show()

#r4esults

N_num = N[-1]
A_num = A[-1]
E_num = E[-1]
Y_num = Y[-1]
revenue_num = R[-1]

print("\n===== NUMERICAL EQUILIBRIUM =====")

print(f"Biomass  = {N_num:.2f} tonnes")
print(f"Vessels  = {A_num:.2f}")
print(f"Effort   = {E_num:.4f} yr^-1")
print(f"Yield    = {Y_num:.2f} tonnes/year")
print(f"Revenue  = €{revenue_num:,.2f}/year")

print("\n===== ANALYTICAL EQUILIBRIUM =====")

print(f"Biomass  = {N_eq:.2f} tonnes")
print(f"Vessels  = {A_eq:.2f}")
print(f"Effort   = {E_eq:.4f} yr^-1")
print(f"Yield    = {Y_eq:.2f} tonnes/year")
print(f"Revenue  = €{p * Y_eq:,.2f}/year")

profit_num = p * b * N_num - c

print(f"\nEquilibrium profit per vessel = €{profit_num:.2f}/year")

#plotting  the resulting interactions
# biomass vs effort
fig, ax1 = plt.subplots(figsize=(8, 5))

line1, = ax1.plot(t, N, label="Biomass")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Biomass (tonnes)")

ax2 = ax1.twinx()

line2, = ax2.plot(t, E, linestyle="--", label="Fishing effort")
ax2.set_ylabel("Fishing effort (1/year)")

ax1.legend(
    [line1, line2],
    ["Biomass", "Fishing effort"],
    loc="center right")

plt.title("Interaction between biomass and fishing effort")
plt.grid(True)

plt.show()

#vessels vs profit
fig, ax1 = plt.subplots(figsize=(8, 5))

line1, = ax1.plot(t, A, label="Fishing vessels")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Number of vessels")

ax2 = ax1.twinx()

line2, = ax2.plot(t, profit, linestyle="--", label="Profit per vessel")
ax2.set_ylabel("Profit (€/year)")

ax1.legend(
    [line1, line2],
    ["Fishing vessels", "Profit per vessel"],
    loc="center right")

plt.title("Interaction between fishing vessels and profitability")
plt.grid(True)

plt.show()

#extension - fleet fixed at 16 vessels (reducing effort) and 60 closure days per year

#extension parameters
A_man = 16  # fixed number of vessels
closure_days = 60  # closure days per year
days_per_year = 365

def management_model(t, N):

    # Day of the year - t % 1 gives the fractional part of t, which represents the time within a year
    day_of_year = (t % 1) * days_per_year

    # Fishing is closed during the first 60 days of each year
    if day_of_year < closure_days:
        E = 0
    else:
        E = b * A_man

    dNdt = r * N * (1 - N / K) - E * N

    return dNdt

t_management = np.linspace(0, 100, 1000)

solution_management = solve_ivp(
    management_model,
    (0, 100),
    [N0],
    t_eval=t_management,
    max_step=0.05)

N_man = solution_management.y[0]

day_of_year = (t_management % 1) * 365

closed = day_of_year < closure_days

E_man = np.where(
    closed,
    0,
    b * A_man)

Y_man = E_man * N_man

plt.figure(figsize=(8, 5))

plt.plot(t, N, label="Open access")
plt.plot(t_management, N_man, label="Management")

plt.xlabel("Time (years)")
plt.ylabel("Biomass (tonnes)")
plt.title("Red shrimp biomass: open access vs management")

plt.legend(loc="best")
plt.grid(True)

plt.show()

#plot - compare scenarios effort and yield 

plt.figure(figsize=(8, 5))

plt.plot(t, E, label="Open access")
plt.plot(t_management, E_man, label="Management")

plt.xlabel("Time (years)")
plt.ylabel("Fishing effort (1/year)")
plt.title("Fishing effort: open access vs management")

plt.legend(loc="best")
plt.grid(True)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(t, Y, label="Open access")
plt.plot(t_management, Y_man, label="Management")

plt.xlabel("Time (years)")
plt.ylabel("Yield (tonnes/year)")
plt.title("Fishing yield: open access vs management")

plt.legend(loc="best")
plt.grid(True)

plt.show()

#final comparison
print("\n===== MANAGEMENT SCENARIO =====")

print(f"Final biomass = {N_man[-1]:.2f} tonnes")
print(f"Number of vessels = {A_man}")
print(f"Maximum fishing effort = {b * A_man:.4f} yr^-1")

print("\n===== OPEN ACCESS SCENARIO =====")

print(f"Final biomass = {N[-1]:.2f} tonnes")
print(f"Final number of vessels = {A[-1]:.2f}")
print(f"Final fishing effort = {E[-1]:.4f} yr^-1")

# Revenue and profit under management
p_man= 41000 #approx price per tonne currently in reality after managment
c_maintenance = 50000   # €/vessel/year - assumption
c_fishing = 70000       # €/vessel/year - assumption
# = 120000 €/vessel/year

fishing_fraction = (days_per_year - closure_days) / days_per_year

#revenue management
R_man = p_man * Y_man

# Total annual cost of the fleet, only when fishery is open - not considering negative costs
# Fishing cost only when the fishery is open
cost_management = np.where(
    E_man > 0,
    A_man * (
        c_maintenance +
        c_fishing / fishing_fraction),
    A_man * c_maintenance)

# Instantaneous economic result
profit_man = R_man - cost_management

plt.figure(figsize=(8, 5))

plt.plot(t_management, R_man, label="Revenue")
plt.plot(t_management, profit_man, label="Profit")

plt.xlabel("Time (years)")
plt.ylabel("€/year")
plt.title("Revenue and profit under management")

plt.legend()
plt.grid(True)

plt.show()

#plotting revenue comparison
# Revenue comparison

R_open = p * Y

plt.figure(figsize=(8, 5))

plt.plot(t, R_open, label="Open access")
plt.plot(t_management, R_man, label="Management")

plt.xlabel("Time (years)")
plt.ylabel("Revenue (€/year)")
plt.title("Revenue: open access vs management")

plt.legend()
plt.grid(True)

plt.show()

#plotting profit comparison

# Profit comparison

profit_open = p * Y - A * c

plt.figure(figsize=(8, 5))

plt.plot(t, profit_open, label="Open access")
plt.plot(t_management, profit_man, label="Management")

plt.xlabel("Time (years)")
plt.ylabel("Profit (€/year)")
plt.title("Profit: open access vs management")

plt.legend()
plt.grid(True)

plt.show()