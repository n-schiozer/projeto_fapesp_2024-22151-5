"""Caminhos centralizados do projeto e de seus dados.

As variáveis de ambiente permitem executar o mesmo código em outra máquina:
``SFC_IO_ABM_DATA_DIR`` para a pasta da TRU e
``SFC_IO_ABM_ARQUIVO_CEI`` para a planilha CEI.
"""

from __future__ import annotations

import os
from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = PASTA_PROJETO
DATA_ROOT = PASTA_MODELO / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_DIR = PASTA_MODELO / "outputs"
DEMOGRAFIA_RAW_DIR = RAW_DIR / "ibge" / "demografia_empresas"

_DATA_DIR_PADRAO = PROCESSED_DIR / "tru" / "nivel_20"
_ARQUIVO_CEI_PADRAO = PROCESSED_DIR / "cei" / "CEI2020_adaptado_V1.xlsx"


DATA_DIR = Path(os.environ.get("SFC_IO_ABM_DATA_DIR", _DATA_DIR_PADRAO))
ARQUIVO_CEI = Path(
    os.environ.get("SFC_IO_ABM_ARQUIVO_CEI", _ARQUIVO_CEI_PADRAO)
)


def validar_caminhos_dados(
    data_dir: Path = DATA_DIR,
    arquivo_cei: Path = ARQUIVO_CEI,
) -> tuple[Path, Path]:
    """Valida as entradas antes da simulação e devolve caminhos normalizados."""

    data_dir = Path(data_dir)
    arquivo_cei = Path(arquivo_cei)
    faltantes = []
    if not data_dir.is_dir():
        faltantes.append(f"pasta TRU: {data_dir}")
    if not arquivo_cei.is_file():
        faltantes.append(f"planilha CEI: {arquivo_cei}")
    if faltantes:
        detalhe = "\n- ".join(faltantes)
        raise FileNotFoundError(
            "Dados de entrada ausentes. Configure SFC_IO_ABM_DATA_DIR e "
            "SFC_IO_ABM_ARQUIVO_CEI (veja .env.example):\n- " + detalhe
        )
    return data_dir, arquivo_cei
