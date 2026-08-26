"""Download reprodutível das Contas Nacionais Trimestrais do IBGE/SIDRA."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


GDP_TABLE = "1621"
GDP_VARIABLE = "584"
GDP_SECTOR_CLASSIFICATION = "11255"
GDP_MARKET_PRICE_CATEGORY = "90707"
GOVERNMENT_CONSUMPTION_CATEGORY = "93405"
GFCF_CATEGORY = "93406"
EXPORTS_CATEGORY = "93407"
GDP_URL = (
    "https://apisidra.ibge.gov.br/values/"
    f"t/{GDP_TABLE}/n1/all/v/{GDP_VARIABLE}/p/all/"
    f"c{GDP_SECTOR_CLASSIFICATION}/{GDP_MARKET_PRICE_CATEGORY}/d/v{GDP_VARIABLE}%202"
)
POPULATION_TABLE = "6579"
POPULATION_VARIABLE = "9324"
POPULATION_URL = (
    "https://apisidra.ibge.gov.br/values/"
    f"t/{POPULATION_TABLE}/n1/all/v/{POPULATION_VARIABLE}/p/all"
)


def quarterly_volume_url(category: str) -> str:
    """Monta a URL do índice trimestral dessazonalizado de volume."""
    return (
        "https://apisidra.ibge.gov.br/values/"
        f"t/{GDP_TABLE}/n1/all/v/{GDP_VARIABLE}/p/all/"
        f"c{GDP_SECTOR_CLASSIFICATION}/{category}/d/v{GDP_VARIABLE}%202"
    )


GOVERNMENT_CONSUMPTION_URL = quarterly_volume_url(
    GOVERNMENT_CONSUMPTION_CATEGORY
)
GFCF_URL = quarterly_volume_url(GFCF_CATEGORY)
EXPORTS_URL = quarterly_volume_url(EXPORTS_CATEGORY)


def _download_json(url: str) -> tuple[bytes, object]:
    request = Request(url, headers={"User-Agent": "io-abm-sfc-empirical/1.0"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    return raw, json.loads(raw.decode("utf-8"))


def download_quarterly_volume(
    raw_path: Path,
    *,
    category: str,
    download: bool = True,
) -> pd.DataFrame:
    """Obtém um índice trimestral de volume com ajuste sazonal."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if download:
        raw, payload = _download_json(quarterly_volume_url(category))
        raw_path.write_bytes(raw)
    else:
        payload = json.loads(raw_path.read_bytes().decode("utf-8"))

    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Resposta SIDRA vazia ou em formato inesperado.")

    frame = pd.DataFrame(payload[1:])
    required = {"D3C", "V"}
    if not required.issubset(frame.columns):
        raise ValueError("Resposta SIDRA não contém os campos D3C e V.")

    quarter = frame["D3C"].astype(str)
    frame["date"] = pd.PeriodIndex(
        quarter.str[:4] + "Q" + quarter.str[-1], freq="Q"
    ).to_timestamp(how="end").normalize()
    frame["value"] = frame["V"].astype(str).str.replace(",", ".", regex=False).astype(float)
    return frame[["date", "value"]].sort_values("date").reset_index(drop=True)


def download_real_gdp(raw_path: Path, download: bool = True) -> pd.DataFrame:
    """Obtém o índice trimestral do PIB real com ajuste sazonal."""
    return download_quarterly_volume(
        raw_path,
        category=GDP_MARKET_PRICE_CATEGORY,
        download=download,
    )


def download_population(raw_path: Path, download: bool = True) -> pd.DataFrame:
    """Obtém a população residente estimada anual do Brasil."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if download:
        raw, payload = _download_json(POPULATION_URL)
        raw_path.write_bytes(raw)
    else:
        payload = json.loads(raw_path.read_bytes().decode("utf-8"))
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Resposta SIDRA de população vazia ou inesperada.")
    frame = pd.DataFrame(payload[1:])
    required = {"D3C", "V"}
    if not required.issubset(frame.columns):
        raise ValueError("Resposta de população não contém D3C e V.")
    frame["date"] = pd.to_datetime(frame["D3C"].astype(str) + "-12-31")
    frame["value"] = (
        frame["V"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    return frame[["date", "value"]].sort_values("date").reset_index(drop=True)
