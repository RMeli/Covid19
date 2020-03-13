import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

import datetime
import itertools
import os

datapath = "data/UK/"
fname = "DailyConfirmedCases.xlsx"

plotpath = "dataviz/plots"

dailycases = pd.read_excel(os.path.join(datapath, fname))

dailycases.rename(
    columns={"DateVal": "date", "CMODateCount": "new", "CumCases": "total"},
    inplace=True,
)

dailycases["date"] = dailycases["date"].apply(pd.to_datetime).dt.date

print(dailycases.describe())


def plotdailycases(yname, log=False, xskyp=2, start_date=datetime.date(2020, 2, 22)):
    plt.figure()

    sns.barplot(
        log=log,
        x="date",
        y=yname,
        data=dailycases[dailycases.date >= start_date],
        palette="rocket_r",
    )

    plt.xlabel("")
    plt.ylabel(f"{yname.capitalize()} Cases (UK)")

    locs, labels = plt.xticks(rotation=90)
    plt.xticks(locs[::xskyp], labels[::xskyp])

    plt.tight_layout()
    plt.savefig(os.path.join(plotpath, f"{yname}_cases_uk{'_log' if log else ''}.pdf"))


for yname, log in itertools.product(("new", "total"), (True, False)):
    plotdailycases(yname, log)
