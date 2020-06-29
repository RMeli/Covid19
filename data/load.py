import pandas as pd
import geopandas as gpd

import os

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def load_ita(describe=False):

    fname = "dpc-covid19-ita-andamento-nazionale.csv"

    data = pd.read_csv(os.path.join(datadir, "Italy", fname))

    mapcols = {
        "data": "date",
        "nuovi_positivi": "new",
        "totale_casi": "total",
        "tamponi": "tested",
    }

    data.rename(
        columns=mapcols, inplace=True,
    )

    data = data[mapcols.values()]

    # Transform column to date
    data["date"] = data["date"].apply(pd.to_datetime).dt.date

    if describe:
        print(data.describe())

    return data, "ita"


def load_ita_geo():

    geoname = "limits_IT_regions.geojson"
    dataname = "dpc-covid19-ita-regioni-latest.csv"

    # Load regions
    regions = gpd.read_file(os.path.join(datadir, "Italy", geoname))

    regions_mapcols = {"reg_name": "name", "reg_istat_code_num": "id"}

    # Rename columns and keep renamed colums
    # Keep geometry column as well
    regions.rename(columns=regions_mapcols, inplace=True)
    regions = regions[list(regions_mapcols.values()) + ["geometry"]]

    # Load data
    cases = pd.read_csv(os.path.join(datadir, "Italy", dataname))

    cases_mapcols = {
        "codice_regione": "id",
        "denominazione_regione": "name",
        "totale_casi": "total",
    }

    cases.rename(
        columns=cases_mapcols, inplace=True,
    )
    cases = cases[cases_mapcols.values()]

    # Put together regions
    cases = cases.groupby("id").sum()

    df = regions.merge(cases, on="id")

    return df, "ita"


def load_uk(describe=False):

    fname = "cases.csv"

    data = pd.read_csv(os.path.join(datadir, "UK", fname))

    mapcols = {"Specimen date": "date", "Daily lab-confirmed cases": "new", "Cumulative lab-confirmed cases": "total"}

    data.rename(
        columns=mapcols, inplace=True,
    )

    data = data[mapcols.values()]

    data["date"] = data["date"].apply(pd.to_datetime).dt.date

    data = data.groupby(data["date"], as_index=False).sum()

    print(data)

    if describe:
        print(data.describe())

    return data, "uk"


if __name__ == "__main__":

    data_ita, _ = load_ita()
    print(data_ita.describe())

    data_uk, _ = load_uk()
    print(data_uk.describe())
