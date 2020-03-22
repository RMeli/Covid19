from matplotlib import pyplot as plt
import seaborn as sns

import itertools
import os
import sys

cdir = os.path.dirname(os.path.realpath(__file__))
pdir = os.path.join(os.path.dirname(cdir), "data")
sys.path.append(pdir)

from load import load_uk, load_ita


def plotdailycases(daya, yname, tag, log=False, xskyp=2):
    plt.figure()

    sns.barplot(
        log=log, x="date", y=yname, data=data, palette="rocket_r",
    )

    plt.xlabel("")
    plt.ylabel(f"{yname.capitalize()} Cases ({tag.upper()})")

    locs, labels = plt.xticks(rotation=90)
    plt.xticks(locs[::xskyp], labels[::xskyp])

    plt.tight_layout()
    plt.savefig(
        os.path.join("plots", f"{yname}_cases_{tag}" f"{'_log' if log else ''}.pdf",)
    )


for loader, yname, log in itertools.product(
    (load_uk, load_ita), ("new", "total"), (True, False)
):
    data, tag = loader()
    plotdailycases(data, yname, tag, log)
