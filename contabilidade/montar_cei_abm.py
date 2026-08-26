"""Adaptador compatível para a interface histórica da montagem CEI."""

from __future__ import annotations

import pandas as pd

from contabilidade.calcular_fluxos_cei import estruturar_fluxos_cei
from contabilidade.montar_cei import montar_cei


def montar_cei_abm(
    estrutura_cei: pd.DataFrame,
    distribuicao: dict,
    importacoes_nominais: float,
    exportacoes_nominais: float,
    consumo_governo: float,
    fbcf: dict,
    estoques: dict,
    teste_flag: bool,
) -> dict:
    """Preserva o contrato antigo e delega à única montagem matricial."""

    fluxos_cei = estruturar_fluxos_cei(
        distribuicao=distribuicao,
        importacoes_nominais=importacoes_nominais,
        exportacoes_nominais=exportacoes_nominais,
        consumo_governo=consumo_governo,
        fbcf=fbcf,
        estoques=estoques,
    )
    return montar_cei(
        estrutura_cei=estrutura_cei,
        fluxos_cei=fluxos_cei,
        teste_flag=teste_flag,
    )
