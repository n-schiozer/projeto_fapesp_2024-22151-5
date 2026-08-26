"""Download reprodutível de séries do SGS/BCB."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


IPCA_SGS_CODE = 433
IPCA_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs."
    f"{IPCA_SGS_CODE}/dados?formato=json"
)


def _download_json(url: str) -> tuple[bytes, object]:
    request = Request(url, headers={"User-Agent": "io-abm-sfc-empirical/1.0"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    return raw, json.loads(raw.decode("utf-8"))


def download_ipca(raw_path: Path, download: bool = True) -> pd.DataFrame:
    """Obtém IPCA mensal (variação percentual) da série SGS 433."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if download:
        raw, payload = _download_json(IPCA_URL)
        raw_path.write_bytes(raw)
    else:
        payload = json.loads(raw_path.read_bytes().decode("utf-8"))

    frame = pd.DataFrame(payload)
    if not {"data", "valor"}.issubset(frame.columns):
        raise ValueError("Resposta do SGS não contém os campos 'data' e 'valor'.")

    frame = frame.rename(columns={"data": "date", "valor": "value"})
    frame["date"] = pd.to_datetime(frame["date"], format="%d/%m/%Y")
    frame["value"] = (
        frame["value"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )
    return frame[["date", "value"]].sort_values("date").reset_index(drop=True)
