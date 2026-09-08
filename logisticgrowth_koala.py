import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## PARAMETERS
N0 =  20000
# lifespan = 12
# average offspring = 7.2
b =  0.3 #approx offsping every year an a half - 2 years
d = 0.1 # 1/12 = aprox 1
d_2 = 0.3
d_5 = 0.7
r_max = 0.2 # max growth rate
K = 30000 # carrying capacity

# simulation time
t_0 = 0
t_end = 40

# logistic growth equations

def logistic_growth(t, N):
    dNdt = r_max * (1 - N / K) * N
    return dNdt

#solving the function
# add time points for the simulation
t = np.linspace(t_0, t_end, 41*12) #define steps, 41 years with 12 months each

# Numerical solution
solution = solve_ivp(
    logistic_growth,
    [t_0, t_end],
    [N0],
    t_eval=t)

N = solution.y[0]

## Plotting initial population and carrying capacity - no fires
plt.plot(t, N)

plt.xlabel("Time [years]")
plt.ylabel("Koala population")
plt.title("Logistic growth of koala population")

plt.grid()
plt.show()

## Analytical solution - no fires

N_analytical = (
    N0 * K * np.exp(r_max * t)
    / (K + N0 * (np.exp(r_max * t) - 1))
)

#plot analytical solution
plt.figure()

plt.plot(t, N, label="Numerical")
plt.plot(t, N_analytical, "--", label="Analytical")

plt.xlabel("Time [years]")
plt.ylabel("Koala population")
plt.title("Logistic growth - analytical vs numerical")

plt.legend()
plt.grid()
plt.show()

## Fire case event period definition and d_2 and d_5
# use modulos function (%) to define years and months when fires are occuring

def mortality_rate(t):

    year = int(t)
    month = (t - year) * 12

    # Fire season: January to April
    if month <= 4 and year > 0:

        # Large fire every 5 years
        if year % 5 == 0:
            return d_5

        # Small fire every 2 years
        elif year % 2 == 0:
            return d_2

    # Normal mortality outside fire periods
    return d

##ploting mortality rate over time
## Death rate evolution with seasonal fires

t_mortality = np.linspace(t_0, t_end, 41*12) # from year 0 to year 40 defining months

death_rate = [mortality_rate(time) for time in t_mortality] #to show stepwise function in further slides
#would be better represented with a hyperbolic function

plt.figure()

plt.plot(t_mortality, death_rate)

plt.xlabel("Time [years]")
plt.ylabel("Death rate")
plt.title("Death rate with seasonal fire events")

plt.grid()
plt.show()

## fire event equation

def logistic_growth_fire(t, N):

    mortality = mortality_rate(t)

    # Additional mortality caused by the fire
    fire_mortality = mortality - d

    dNdt = ( r_max * (1 - N / K) * N
        - fire_mortality * N
    )
    return dNdt

## numerical solution with fire events

solution_fire = solve_ivp(
    logistic_growth_fire,
    [t_0, t_end],
    [N0],
    max_step=0.02, #ai help - this is to ensure that the solver takes small enough steps to accurately capture the dynamics of the system, especially during periods of rapid change.
    # without max_step just the first and last fire events were captured 
    t_eval=t)

N_fire = solution_fire.y[0]

#plot fire event 

plt.plot(t, N_fire)
plt.xlabel("Time [years]")
plt.ylabel("Koala population")
plt.title("Koala population with seasonal fires")

plt.grid()
plt.show()

## analytical solution with fire events 
# - AI helped me coding this part, especially to define the months when fires are occuring

## Analytical solution for one period (year)

def analytical_fire(N_start, duration, mortality):

    R = r_max + d - mortality

    if np.isclose(R, 0):

        N_end = N_start / (
            1 + (r_max / K) * N_start * duration
        )

    else:

        N_end = (
            R * N_start * np.exp(R * duration)
            / (
                R
                + (r_max / K) * N_start
                * (np.exp(R * duration) - 1)
            )
        )

    return N_end

### Analytical solution for multiple periods (years)

N_analytical_fire = [N0]
t_analytical_fire = [0]

N_current = N0

for year in range(40):

    # January to April
    if year > 0 and year % 5 == 0:
        mortality = d_5

    elif year > 0 and year % 2 == 0:
        mortality = d_2

    else:
        mortality = d

    N_current = analytical_fire(
        N_current,
        4 / 12,
        mortality)

    N_analytical_fire.append(N_current)
    t_analytical_fire.append(year + 4 / 12)

    # May to December
    N_current = analytical_fire(
        N_current,
        8 / 12,
        d)

    N_analytical_fire.append(N_current)
    t_analytical_fire.append(year + 1)

#plotting analytical solution with fires - it open 40 windows bc it is in a loop

    plt.figure()

plt.plot(t, N_fire, label="Numerical")
plt.plot(
    t_analytical_fire,
    N_analytical_fire,
    "--",
    label="Analytical")

plt.xlabel("Time [years]")
plt.ylabel("Koala population")
plt.title("Koala population with seasonal fires - Analytical vs Numerical")

plt.legend()
plt.grid()
plt.show()
