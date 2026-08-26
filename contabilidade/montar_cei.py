"""Montagem contábil pura da matriz CEI."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import COLUNAS_SETORES, L


def _total_lado(
    movimentos: Mapping[str, Mapping[str, float]],
    lado: str,
) -> float:
    return float(sum(float(valor) for valor in movimentos[lado].values()))


def montar_cei(
    *,
    estrutura_cei: pd.DataFrame,
    fluxos_cei: Mapping[str, Mapping[str, Mapping[str, float]]],
    teste_flag: bool,
) -> dict:
    """Localiza lançamentos, fecha saldos e devolve os diagnósticos da CEI."""

    cei = estrutura_cei.copy(deep=True)
    cei.iloc[1:17, 1:11] = np.nan

    for conta, movimentos in fluxos_cei.items():
        linha = L[conta]
        for nome, valor in movimentos["entradas"].items():
            entrada, _ = COLUNAS_SETORES[nome]
            cei.iloc[linha, entrada] = float(valor)
        for nome, valor in movimentos["saidas"].items():
            _, saida = COLUNAS_SETORES[nome]
            cei.iloc[linha, saida] = float(valor)

    fechamentos = {
        "juros": "juros",
        "dividendos": "dividendos",
        "contribuições sociais": "contribuicoes_sociais",
        "aposentadorias": "aposentadorias",
        "outras transferências": "outras_transferencias",
    }
    for nome, conta in fechamentos.items():
        recebido = _total_lado(fluxos_cei[conta], "entradas")
        pago = _total_lado(fluxos_cei[conta], "saidas")
        if not np.isclose(recebido, pago, atol=1e-6):
            raise RuntimeError(
                f"O fluxo '{nome}' não fecha: recebido={recebido}, pago={pago}."
            )

    capacidade = {}
    for nome, (entrada, saida) in COLUNAS_SETORES.items():
        entradas = np.asarray(cei.iloc[1:16, entrada], dtype=float)
        saidas = np.asarray(cei.iloc[1:16, saida], dtype=float)
        saldo = float(
            np.nan_to_num(entradas, nan=0.0).sum()
            - np.nan_to_num(saidas, nan=0.0).sum()
        )
        capacidade[nome] = saldo
        cei.iloc[L["capacidade"], entrada] = saldo

    saldo_linhas = {}
    colunas_entrada = [posicoes[0] for posicoes in COLUNAS_SETORES.values()]
    colunas_saida = [posicoes[1] for posicoes in COLUNAS_SETORES.values()]
    for linha in range(1, 16):
        entradas = float(
            np.nansum(cei.iloc[linha, colunas_entrada].to_numpy(dtype=float))
        )
        saidas = float(
            np.nansum(cei.iloc[linha, colunas_saida].to_numpy(dtype=float))
        )
        saldo_linhas[str(cei.iloc[linha, 0])] = entradas - saidas

    discrepancia = float(sum(capacidade.values()))
    fechou = bool(np.isclose(discrepancia, 0.0, atol=1e-6))
    if teste_flag and not fechou:
        print(
            "\nATENÇÃO: A CEI NÃO FECHA"
            f"\nDiscrepância = {discrepancia:,.6f}\n"
        )

    return {
        "cei": cei,
        "capacidade_financiamento": capacidade,
        "saldo_linhas": saldo_linhas,
        "discrepancia": discrepancia,
        "fechou": fechou,
    }
