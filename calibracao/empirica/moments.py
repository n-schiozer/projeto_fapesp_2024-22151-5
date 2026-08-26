"""Cálculo de momentos empíricos, independente da lógica de simulação."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_moments(values: pd.Series) -> dict[str, float | int]:
    """Retorna momentos descritivos comparáveis, ignorando observações ausentes."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n_obs": int(clean.size),
        "mean": float(clean.mean()) if clean.size else float("nan"),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else float("nan"),
        "autocorrelation_lag1": (
            float(clean.autocorr(lag=1)) if clean.size > 2 else float("nan")
        ),
        "minimum": float(clean.min()) if clean.size else float("nan"),
        "maximum": float(clean.max()) if clean.size else float("nan"),
    }


def growth_rates(values: pd.Series) -> pd.Series:
    """Retorna variações relativas, preservando a ordem temporal."""
    clean = pd.to_numeric(values, errors="coerce")
    return clean.pct_change(fill_method=None)


def cross_correlation(
    first: pd.Series,
    second: pd.Series,
    *,
    lag_second: int = 0,
) -> float:
    """Correlação comum entre a primeira série e a segunda defasada."""
    aligned = pd.concat(
        [
            pd.to_numeric(first, errors="coerce"),
            pd.to_numeric(second, errors="coerce").shift(lag_second),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 3:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def build_calibration_moments(targets: pd.DataFrame) -> pd.DataFrame:
    """Constrói os 11 targets iniciais da calibração em frações."""

    data = targets.copy()
    data["date"] = pd.to_datetime(data["date"])

    def series(series_id: str, start: str, end: str) -> pd.Series:
        subset = data.loc[data["series_id"].eq(series_id)].copy()
        subset = subset.loc[subset["date"].between(start, end)].sort_values("date")
        values = pd.to_numeric(subset["value"], errors="coerce")
        if not subset.empty and subset["unit"].iloc[0] == "%":
            values = values / 100.0
        return pd.Series(values.to_numpy(), index=subset["date"], dtype=float)

    definitions = (
        ("inflacao", series("inflacao_anual", "2010-01-01", "2019-12-31"), "2010-2019"),
        ("crescimento_pib_real", series("crescimento_pib_real_anual", "2010-01-01", "2019-12-31"), "2010-2019"),
        ("taxa_desemprego", series("taxa_desemprego_anual", "2012-01-01", "2019-12-31"), "2012-2019"),
    )
    rows: list[dict[str, object]] = []
    for variable, values, period in definitions:
        moments = calculate_moments(values)
        scale = max(float(moments["std"]), 1e-4)
        for moment, source_key in (
            ("mean", "mean"),
            ("std", "std"),
            ("acf1", "autocorrelation_lag1"),
        ):
            rows.append(
                {
                    "moment_id": f"{variable}__{moment}",
                    "variable": variable,
                    "moment": moment,
                    "value": float(moments[source_key]),
                    "scale": 1.0 if moment == "acf1" else scale,
                    "n_obs": int(moments["n_obs"]),
                    "period": period,
                    "unit": "fraction",
                }
            )

    population = series(
        "crescimento_populacao_anual", "2010-01-01", "2019-12-31"
    )
    pop_moments = calculate_moments(population)
    rows.append(
        {
            "moment_id": "crescimento_populacao__mean",
            "variable": "crescimento_populacao",
            "moment": "mean",
            "value": float(pop_moments["mean"]),
            "scale": max(float(pop_moments["std"]), 1e-4),
            "n_obs": int(pop_moments["n_obs"]),
            "period": "2012-2019",
            "unit": "fraction",
        }
    )

    inflation = definitions[0][1]
    unemployment = definitions[2][1]
    annual = pd.concat(
        [inflation.rename("inflacao"), unemployment.rename("desemprego")],
        axis=1,
    ).sort_index()
    cross = cross_correlation(
        annual["inflacao"], annual["desemprego"], lag_second=1
    )
    rows.append(
        {
            "moment_id": "inflacao_desemprego_defasado__corr",
            "variable": "inflacao_desemprego_defasado",
            "moment": "corr_pi_t_u_t_1",
            "value": cross,
            "scale": 1.0,
            "n_obs": int(annual.dropna().shape[0] - 1),
            "period": "2013-2019",
            "unit": "correlation",
        }
    )
    result = pd.DataFrame(rows)
    if not np.isfinite(result["value"]).all():
        raise ValueError("A base empírica produziu momentos não finitos.")
    return result


def build_moments_table(
    targets: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Calcula momentos apenas para séries liberadas como ready no catálogo."""
    ready = mapping.loc[mapping["status"].eq("ready")].copy()
    rows: list[dict[str, object]] = []
    for item in ready.to_dict(orient="records"):
        subset = targets.loc[targets["series_id"].eq(item["empirical_series_id"])]
        if subset.empty:
            continue
        moments = calculate_moments(subset["value"])
        for measure, value in moments.items():
            rows.append(
                {
                    "moment_id": item["moment_id"],
                    "series_id": item["empirical_series_id"],
                    "model_variable": item["model_variable"],
                    "use": item["use"],
                    "status": item["status"],
                    "moment": measure,
                    "value": value,
                    "frequency": subset["frequency"].iloc[0],
                    "unit": subset["unit"].iloc[0],
                }
            )
    return pd.DataFrame(rows)
