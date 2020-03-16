import pandas as pd

import os

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")


def load_ita(describe=False):

    fname = "dpc-covid19-ita-andamento-nazionale.csv"

    data = pd.read_csv(os.path.join(datadir, "Italy", fname))

    data.rename(
        columns={
            "data": "date",
            "nuovi_attualmente_positivi": "new",
            "totale_casi": "total",
            "tamponi": "tested",
        },
        inplace=True,
    )

    data.drop(
        columns=[
            "stato",
            "ricoverati_con_sintomi",
            "terapia_intensiva",
            "totale_ospedalizzati",
            "isolamento_domiciliare",
            "totale_attualmente_positivi",
            "dimessi_guariti",
            "deceduti",
        ],
        inplace=True,
    )

    # Transform column to date
    data["date"] = data["date"].apply(pd.to_datetime).dt.date

    if describe:
        print(data.describe())

    return data, "ita"


def load_uk(describe=False):

    fname = "DailyConfirmedCases.xlsx"

    data = pd.read_excel(os.path.join(datadir, "UK", fname))

    data.rename(
        columns={"DateVal": "date", "CMODateCount": "new", "CumCases": "total"},
        inplace=True,
    )

    data["date"] = data["date"].apply(pd.to_datetime).dt.date

    if describe:
        print(data.describe())

    return data, "uk"


if __name__ == "__main__":

    data_ita, _ = load_ita()
    print(data_ita.describe())

    data_uk, _ = load_uk()
    print(data_uk.describe())
