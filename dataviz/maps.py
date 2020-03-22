import pandas as pd
import geopandas as gpd

from matplotlib import pyplot as plt

import os
import sys

cdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.join(os.path.dirname(cdir), "data")
sys.path.append(pdir)

from load import load_ita_geo

df = load_ita_geo()


def plotmap(df, tag):
    fig, ax = plt.subplots(1, 1)

    df.plot(
        ax=ax,
        column="total", # Data for choropleth map
        cmap="OrRd", # Color map
        legend=True,
        edgecolor="black",
        linewidth=0.1,
    )

    plt.axis("off")
    plt.title(f"Total Cases ({tag.upper()})")

    plt.savefig(f"plots/map_{tag}.pdf")

for loader in [load_ita_geo]:

    df, tag = loader()

    plotmap(df, tag)