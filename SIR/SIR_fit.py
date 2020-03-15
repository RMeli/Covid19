import pandas as pd
import numpy as np

from scipy import optimize

from matplotlib import pyplot as plt

import os

from SIR import plot_sir

datapath = "../data/UK/"
fname = "DailyConfirmedCases.xlsx"

plotpath = "dataviz/plots"

dailycases = pd.read_excel(os.path.join(datapath, fname)).drop(
    columns=["DateVal", "CMODateCount"]
)

dailycases.rename(
    columns={"CumCases": "total"}, inplace=True,
)


def I_exp(t, I0, R0, gamma):
    """
    Infected (I); initial exponential growth for SIR model.

    Parameters
    ----------
    t: double
        Elapsed time
    R0: double
        Basic reproductive ratio
    gamma: double
    """

    return I0 * np.exp((R0 - 1) * gamma * t)


t = np.arange(0, len(dailycases["total"]))
y = dailycases["total"]

# From UK data, exponential growth starts 27 days after first cases
t0 = 27

popt, _ = optimize.curve_fit(I_exp, t[t0:], y[t0:], p0=[13, 2.5, 1])

print(f"I0 = {popt[0]:.2f}")
print(f"R0 = {popt[1]:.2f}")
print(f"γ = {popt[2]:.2f}")

tt = np.linspace(t0, t[-1], 1000)

fig = plt.figure()
plt.semilogy(t, y, "o")
plt.semilogy(
    tt,
    I_exp(tt, popt[0], popt[1], popt[2]),
    label=f"$R_0$ = {popt[1]:.2f}; $\\gamma$ = {popt[2]:.2f}",
)

plt.legend()
plt.xlabel("Time (Days)")
plt.ylabel("Total Cases (UK)")

plt.savefig("plots/SIR_fit.pdf")
plt.close(fig)

t, y = plot_sir(I0=popt[0], tint=(0, 100), R0=popt[1], gamma=popt[2], tag="fitted_UK")
print(f"I_max = {max(y[1]):.3f}")