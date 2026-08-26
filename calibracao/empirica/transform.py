"""Transformações declaradas da Base Empírica."""

from __future__ import annotations

import pandas as pd


def compound_monthly_inflation_to_quarterly(ipca: pd.DataFrame) -> pd.DataFrame:
    """Compõe taxas mensais percentuais de IPCA em inflação trimestral."""
    frame = ipca.copy()
    frame["quarter"] = frame["date"].dt.to_period("Q")
    result = frame.groupby("quarter")["value"].agg(
        value=lambda values: ((1 + values / 100).prod() - 1) * 100,
        n_months="count",
    )
    result = result.loc[result["n_months"].eq(3)].reset_index()
    result["date"] = result["quarter"].dt.to_timestamp(how="end").dt.normalize()
    return result[["date", "value"]]


def annualize_quarterly_rate(quarterly_rate: pd.DataFrame) -> pd.DataFrame:
    """Converte uma taxa trimestral percentual em taxa anualizada percentual."""
    result = quarterly_rate.copy()
    result["value"] = ((1 + result["value"] / 100) ** 4 - 1) * 100
    return result


def compound_monthly_inflation_to_annual(ipca: pd.DataFrame) -> pd.DataFrame:
    """Compõe doze taxas mensais percentuais em inflação anual."""

    frame = ipca.copy()
    frame["year"] = frame["date"].dt.year
    result = frame.groupby("year")["value"].agg(
        value=lambda values: ((1.0 + values / 100.0).prod() - 1.0) * 100.0,
        n_months="count",
    )
    result = result.loc[result["n_months"].eq(12)].reset_index()
    result["date"] = pd.to_datetime(result["year"].astype(str) + "-12-31")
    return result[["date", "value"]]


def annual_average_quarterly_index(index_series: pd.DataFrame) -> pd.DataFrame:
    """Calcula o índice anual como média dos quatro índices trimestrais."""

    frame = index_series.copy()
    frame["year"] = frame["date"].dt.year
    result = frame.groupby("year")["value"].agg(value="mean", n_quarters="count")
    result = result.loc[result["n_quarters"].eq(4)].reset_index()
    result["date"] = pd.to_datetime(result["year"].astype(str) + "-12-31")
    return result[["date", "value"]]


def annual_growth(index_series: pd.DataFrame) -> pd.DataFrame:
    """Calcula crescimento percentual anual de um índice de volume."""

    result = index_series.copy().sort_values("date").reset_index(drop=True)
    result["value"] = result["value"].pct_change() * 100.0
    return result.dropna(subset=["value"]).reset_index(drop=True)


def quarter_over_quarter_growth(index_series: pd.DataFrame) -> pd.DataFrame:
    """Calcula a variação percentual contra o trimestre imediatamente anterior."""
    result = index_series.copy().sort_values("date").reset_index(drop=True)
    result["value"] = result["value"].pct_change() * 100
    return result.dropna(subset=["value"]).reset_index(drop=True)


def filter_period(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Restringe uma série a datas inclusivas, sem preencher valores ausentes."""
    result = frame.copy()
    if start:
        result = result.loc[result["date"] >= pd.Timestamp(start)]
    if end:
        result = result.loc[result["date"] <= pd.Timestamp(end)]
    return result.reset_index(drop=True)
