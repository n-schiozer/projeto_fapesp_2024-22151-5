"""Tendência real e conversão nominal da demanda autônoma do ABM."""

from __future__ import annotations

import pandas as pd


def calcular_demanda_autonoma(
    *,
    governo_base: pd.Series,
    fbcf_fixa_base: pd.Series,
    exportacoes_base: pd.Series,
    precos_setoriais: pd.Series,
    periodo: int,
    taxa_crescimento_demanda_autonoma: float,
    fator_governo: float = 1.0,
    fator_investimento: float = 1.0,
    fator_exportacoes: float = 1.0,
) -> dict[str, pd.Series]:
    """Aplica uma tendência real comum e, separadamente, choques de nível.

    As três bases são quantidades reais do ano-base. A multiplicação pelos
    preços setoriais correntes converte os fluxos para valores nominais sem
    alterar sua tendência real.
    """

    taxa = float(taxa_crescimento_demanda_autonoma)
    if taxa <= -1.0:
        raise ValueError(
            "taxa_crescimento_demanda_autonoma deve ser maior que -1."
        )
    if isinstance(periodo, bool) or int(periodo) != periodo or periodo < 0:
        raise ValueError("periodo deve ser um inteiro não negativo.")

    fator_tendencia = (1.0 + taxa) ** int(periodo)
    reais = {
        "governo_real_planejado": governo_base * fator_tendencia * fator_governo,
        "fbcf_fixa_real_planejada": (
            fbcf_fixa_base * fator_tendencia * fator_investimento
        ),
        "exportacoes_real_planejadas": (
            exportacoes_base * fator_tendencia * fator_exportacoes
        ),
    }
    return {
        **reais,
        "governo_nominal": reais["governo_real_planejado"] * precos_setoriais,
        "fbcf_fixa_nominal": (
            reais["fbcf_fixa_real_planejada"] * precos_setoriais
        ),
        "exportacoes_nominais": (
            reais["exportacoes_real_planejadas"] * precos_setoriais
        ),
    }
