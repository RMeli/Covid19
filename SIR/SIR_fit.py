import pandas as pd
import numpy as np

from scipy import optimize

from matplotlib import pyplot as plt

import os
import sys

from SIR import plot_sir

cdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.join(os.path.dirname(cdir), "data")
sys.path.append(pdir)

from load import load_uk, load_ita


population = {"ita": 60549600, "uk": 66435550}
exponentialgrowth = {"ita": (3, 20), "uk": (27, None)}


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


for loader in [load_uk, load_ita]:

    data, tag = loader()

    t = np.arange(0, len(data["total"]))
    y = data["total"]

    ti, tf = exponentialgrowth[tag]

    if ti is None:
        ti = 0
    if tf is None:
        tf = -1

    popt, _ = optimize.curve_fit(I_exp, t[ti:tf], y[ti:tf], p0=[13, 2.5, 1])

    print(f"I0 = {popt[0]:.2f}")
    print(f"R0 = {popt[1]:.2f}")
    print(f"γ = {popt[2]:.2f}")

    tt = np.linspace(t[ti], t[tf], 1000)

    fig = plt.figure()
    plt.semilogy(t, y, "o")
    plt.semilogy(
        tt,
        I_exp(tt, popt[0], popt[1], popt[2]),
        label=f"$R_0$ = {popt[1]:.2f}; $\\gamma$ = {popt[2]:.2f}",
    )

    plt.legend()
    plt.xlabel("Time (Days)")
    plt.ylabel(f"Total Cases ({tag.upper()})")

    plt.savefig(f"plots/SIR_fit_{tag}.pdf")
    plt.close(fig)

    t_sir, y_sir = plot_sir(
        I0=popt[0], tint=(0, 100), R0=popt[1], gamma=popt[2], tag=f"fitted"
    )

    t_max = t_sir[np.argmax(y_sir[1])]
    I_max = max(y_sir[1])
    print(f"I_max = {I_max:.3f}")

    N = population[tag]

    fig = plt.figure()

    plt.semilogy(t, y, "o")  # Data
    plt.semilogy(t_sir, y_sir[1] * N, label="SIR")  # Fit
    plt.plot(t_max, I_max * N, "o", label=f"I = {I_max * N / 1e6:.1f} M")  # Maximum

    plt.ylim([1, None])
    plt.legend()
    plt.xlabel("Time (Days)")
    plt.ylabel(f"Total Cases ({tag.upper()})")

    plt.savefig(f"plots/SIR_fit_infected_{tag}.pdf")
    plt.close(fig)
