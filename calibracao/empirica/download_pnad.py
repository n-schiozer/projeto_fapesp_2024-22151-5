"""Download reprodutível da taxa anual de desocupação da PNAD Contínua."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


UNEMPLOYMENT_TABLE = "4562"
UNEMPLOYMENT_VARIABLE = "4099"
UNEMPLOYMENT_URL = (
    "https://apisidra.ibge.gov.br/values/"
    f"t/{UNEMPLOYMENT_TABLE}/n1/all/v/{UNEMPLOYMENT_VARIABLE}/p/all"
)


def _download_json(url: str) -> tuple[bytes, object]:
    request = Request(url, headers={"User-Agent": "io-abm-sfc-empirical/1.0"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    return raw, json.loads(raw.decode("utf-8"))


def download_annual_unemployment(
    raw_path: Path,
    download: bool = True,
) -> pd.DataFrame:
    """Obtém a taxa anual de desocupação do Brasil, em percentual."""

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if download:
        raw, payload = _download_json(UNEMPLOYMENT_URL)
        raw_path.write_bytes(raw)
    else:
        payload = json.loads(raw_path.read_bytes().decode("utf-8"))

    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Resposta SIDRA da PNAD vazia ou em formato inesperado.")

    frame = pd.DataFrame(payload[1:])
    if not {"D3C", "V"}.issubset(frame.columns):
        raise ValueError("Resposta SIDRA da PNAD não contém os campos D3C e V.")

    frame["date"] = pd.to_datetime(
        frame["D3C"].astype(str) + "-12-31",
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(
        frame["V"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return (
        frame[["date", "value"]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
