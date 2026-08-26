"""Rateio contábil proporcional das transações já definidas pelo mercado."""

from __future__ import annotations

import pandas as pd


def ratear_atendimento_proporcional(
    demandas_pb_nominal: dict[str, pd.Series],
    participantes_industriais: pd.DataFrame,
    participantes_leilao: pd.DataFrame,
    setores: list[str],
) -> dict:
    """Decompõe vendas agregadas sem alterar shares, preços ou mercados."""
    desejado = pd.DataFrame(demandas_pb_nominal).reindex(index=setores).fillna(0.0)
    participantes = pd.concat([participantes_industriais, participantes_leilao])
    realizado = pd.DataFrame(0.0, index=setores, columns=desejado.columns)
    realizado_domestico = realizado.copy()
    realizado_importado = realizado.copy()
    for setor in setores:
        d = float(desejado.loc[setor].sum())
        vendas = participantes.loc[participantes["setor"] == setor]
        for _, fornecedor in vendas.iterrows():
            v = float(fornecedor["vendas_nominal"])
            parcela_fornecedor = 0.0 if d == 0.0 else v / d
            alocado = desejado.loc[setor] * parcela_fornecedor
            realizado.loc[setor] += alocado
            (realizado_importado if fornecedor["tipo"] == "importado" else realizado_domestico).loc[setor] += alocado
    nao_atendido = desejado - realizado
    if (nao_atendido < -1e-6).any().any():
        raise RuntimeError("Rateio categorial excedeu a demanda desejada.")
    return {
        "desejado_pb_nominal": desejado,
        "realizado_pb_nominal": realizado,
        "nao_atendido_pb_nominal": nao_atendido.clip(lower=0.0),
        "realizado_domestico_pb_nominal": realizado_domestico,
        "realizado_importado_pb_nominal": realizado_importado,
    }
