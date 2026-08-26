"""Clearing determinístico de mercados de leilão a preço uniforme."""

import numpy as np
import pandas as pd


def executar_leilao_uniforme(
    ofertas: pd.DataFrame,
    demanda_nominal_pb: float,
) -> dict:
    """Despacha ofertas por mérito sob orçamento nominal a preços básicos.

    Para cada preço candidato da escada de oferta, a função testa se a
    quantidade compatível com o orçamento (``D / p``) está entre a oferta
    acumulada anterior e a acumulada até o candidato. O primeiro degrau que
    satisfaz essa condição é a solução: seu ofertante é marginal e recebe
    despacho parcial quando necessário. Se a oferta total for insuficiente
    mesmo ao maior preço, toda ela é despachada e fica demanda não atendida.
    """

    colunas = {
        "participante",
        "tipo",
        "oferta_disponivel_real",
        "preco_oferta",
    }
    if not colunas.issubset(ofertas.columns):
        faltantes = sorted(colunas.difference(ofertas.columns))
        raise KeyError(f"Ofertas sem colunas obrigatórias: {faltantes}.")
    if demanda_nominal_pb < 0.0:
        raise ValueError("demanda_nominal_pb não pode ser negativa.")

    resultado = ofertas.copy()
    resultado["oferta_disponivel_real"] = resultado[
        "oferta_disponivel_real"
    ].astype(float)
    resultado["preco_oferta"] = resultado["preco_oferta"].astype(float)
    if (resultado["oferta_disponivel_real"] < 0.0).any():
        raise ValueError("A oferta disponível não pode ser negativa.")
    if (resultado["preco_oferta"] <= 0.0).any() or not np.isfinite(
        resultado["preco_oferta"]
    ).all():
        raise ValueError("Todo preço de oferta deve ser positivo e finito.")
    if resultado["participante"].duplicated().any():
        raise ValueError("Os identificadores dos ofertantes devem ser únicos.")

    resultado = resultado.sort_values(
        ["preco_oferta", "participante"], kind="stable"
    ).reset_index(drop=True)
    resultado["quantidade_despachada_real"] = 0.0
    positivos = resultado[resultado["oferta_disponivel_real"] > 0.0]
    if positivos.empty:
        return {
            "ofertas": resultado,
            "preco_transacao_setorial": np.nan,
            "ofertante_marginal": None,
            "preco_oferta_marginal": np.nan,
            "quantidade_demandada_real": 0.0,
            "demanda_nao_atendida_real": 0.0,
            "oferta_total_real": 0.0,
            "escassez": False,
        }

    acumulada_anterior = 0.0
    indice_marginal = None
    quantidade_demandada = 0.0
    tolerancia = 1e-10
    for indice, oferta in positivos.iterrows():
        preco = float(oferta["preco_oferta"])
        quantidade_no_preco = float(demanda_nominal_pb / preco)
        acumulada = acumulada_anterior + float(oferta["oferta_disponivel_real"])
        if (
            quantidade_no_preco >= acumulada_anterior - tolerancia
            and quantidade_no_preco <= acumulada + tolerancia
        ):
            indice_marginal = indice
            quantidade_demandada = quantidade_no_preco
            break
        acumulada_anterior = acumulada

    oferta_total = float(positivos["oferta_disponivel_real"].sum())
    if indice_marginal is None:
        maior_preco = float(positivos["preco_oferta"].iloc[-1])
        quantidade_no_maior_preco = demanda_nominal_pb / maior_preco
        if quantidade_no_maior_preco > oferta_total + tolerancia:
            indice_marginal = int(positivos.index[-1])
            quantidade_demandada = quantidade_no_maior_preco
            resultado.loc[positivos.index, "quantidade_despachada_real"] = positivos[
                "oferta_disponivel_real"
            ]
            preco_transacao = maior_preco
            return {
                "ofertas": resultado,
                "preco_transacao_setorial": preco_transacao,
                "ofertante_marginal": resultado.at[indice_marginal, "participante"],
                "preco_oferta_marginal": preco_transacao,
                "quantidade_demandada_real": quantidade_demandada,
                "demanda_nao_atendida_real": quantidade_demandada - oferta_total,
                "oferta_total_real": oferta_total,
                "escassez": True,
            }
        raise RuntimeError(
            "Não existe degrau de oferta consistente com o orçamento nominal; "
            "a curva escalonada exige uma regra adicional de demanda."
        )

    preco_transacao = float(resultado.at[indice_marginal, "preco_oferta"])
    anteriores = positivos.index[positivos.index < indice_marginal]
    if len(anteriores):
        resultado.loc[anteriores, "quantidade_despachada_real"] = resultado.loc[
            anteriores, "oferta_disponivel_real"
        ]
    quantidade_marginal = max(0.0, quantidade_demandada - acumulada_anterior)
    resultado.at[indice_marginal, "quantidade_despachada_real"] = min(
        float(resultado.at[indice_marginal, "oferta_disponivel_real"]),
        quantidade_marginal,
    )
    return {
        "ofertas": resultado,
        "preco_transacao_setorial": preco_transacao,
        "ofertante_marginal": resultado.at[indice_marginal, "participante"],
        "preco_oferta_marginal": preco_transacao,
        "quantidade_demandada_real": quantidade_demandada,
        "demanda_nao_atendida_real": 0.0,
        "oferta_total_real": oferta_total,
        "escassez": False,
    }
