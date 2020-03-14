import numpy as np

from scipy import integrate

from matplotlib import pyplot as plt
import seaborn as sns


N = 68e6
R0 = 2.5

beta = 1
gamma = beta / R0


def f(t, y):
    S, I, R = y[0], y[1], y[2]

    dSdt = -beta / N * I * S
    dIdt = beta / N * I * S - gamma * I
    dRdt = gamma * I

    return np.array([dSdt, dIdt, dRdt])


y0 = np.array([N, 700, 0])

sir = integrate.solve_ivp(f, (0, 60), y0, rtol=1e-12, atol=1e-12)

plt.figure()
plt.plot(sir.t, sir.y[0], label="Susceptible")
plt.plot(sir.t, sir.y[1], label="Infected")
plt.plot(sir.t, sir.y[2], label="Recovered")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Number of People")
plt.savefig("plots/SIR_simple.pdf")
