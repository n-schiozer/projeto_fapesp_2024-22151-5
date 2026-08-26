"""Calibração direta da FBCF das firmas não financeiras do ABM."""

from __future__ import annotations

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import C, L
from investimento.investimento_abm import SETOR_CONSTRUCAO


def calibrar_investimento_nf_abm(
    condicoes_iniciais: dict,
    config: dict | None = None,
) -> dict:
    """Calibra capital e demanda de investimento NF a preços básicos.

    A FBCF institucional observada é distribuída entre fornecedores pela
    composição da TRU a preços de comprador e convertida uma única vez para
    preços básicos. No estado-base estacionário, ``K = v * Y`` e, portanto,
    ``I = depreciação * K``.
    """

    config = condicoes_iniciais["config"] if config is None else config
    setores = list(condicoes_iniciais["setores"])
    setores_excluidos = list(
        config.get(
            "setores_excluidos_investimento_nf",
            [setores[config["setor_financeiro"]]],
        )
    )
    desconhecidos = set(setores_excluidos).difference(setores)
    if desconhecidos:
        raise KeyError(
            "Setores excluídos do investimento NF ausentes: "
            f"{sorted(desconhecidos)}"
        )
    setores_nf = [setor for setor in setores if setor not in setores_excluidos]

    depreciacao = 1.0 / float(config["vida_util_capital"])
    if depreciacao <= 0.0:
        raise ValueError("vida_util_capital deve ser positiva.")

    fbcf_total_tru_pm = (
        condicoes_iniciais["tru_base"].gross_investment_sector.iloc[:, 0]
        .reindex(setores)
        .astype(float)
    )

    fbcf_nf_total_pm = float(
        condicoes_iniciais["valores_cei"].iat[L["fbcf"], C["nf_s"]]
    )

    fbcf_familias_pm = float(
        condicoes_iniciais["valores_cei"].iat[L["fbcf"], C["familias_s"]]
    )

    if SETOR_CONSTRUCAO not in setores:
        raise KeyError("O setor de Construção não foi encontrado na TRU.")

    pesos_investimento_familias = pd.Series(
        0.0,
        index=setores,
        name="peso_investimento_familias",
    )
    pesos_investimento_familias.at[SETOR_CONSTRUCAO] = 1.0
    fbcf_familias_por_fornecedor = (
        pesos_investimento_familias * fbcf_familias_pm
    )

    base_composicao_nf_pm = fbcf_total_tru_pm - fbcf_familias_por_fornecedor

    if (base_composicao_nf_pm < -1e-8).any():
        raise ValueError("A FBCF das famílias excede a FBCF da TRU.")
    base_composicao_nf_pm = base_composicao_nf_pm.clip(lower=0.0)
    total_base_composicao = float(base_composicao_nf_pm.sum())

    if fbcf_nf_total_pm <= 0.0 or total_base_composicao <= 0.0:
        raise ValueError("A base da FBCF NF deve ser positiva.")

    fbcf_nf_pm = (
        base_composicao_nf_pm / total_base_composicao * fbcf_nf_total_pm
    ).rename("fbcf_nf_pm")

    fbcf_fixa_base = (
        fbcf_total_tru_pm
        - fbcf_familias_por_fornecedor
        - fbcf_nf_pm
    ).rename("fbcf_fixa_base")

    fbcf_nf_pb = (
        condicoes_iniciais["conversao_de_pm_pb"].reindex(
            index=setores,
            columns=setores,
        )
        @ fbcf_nf_pm
    ).rename("fbcf_nf_pb")

    fbcf_nf_total_pb = float(fbcf_nf_pb.sum())
    if fbcf_nf_total_pb <= 0.0:
        raise ValueError("A FBCF NF a preços básicos deve ser positiva.")

    fbcf_nf_pb_positiva = fbcf_nf_pb.clip(lower=0.0)

    soma_positiva = float(fbcf_nf_pb_positiva.sum())

    if soma_positiva <= 0.0:
        raise ValueError("A composição positiva da FBCF NF é nula.")
    pesos_bens_capital_nf = (
        fbcf_nf_pb_positiva / soma_positiva
    ).rename("peso_bens_capital_nf")

    producao_base = (
        condicoes_iniciais["conversao_domestica"]
        @ condicoes_iniciais["demanda_final_base"]
    ).reindex(setores).astype(float)

    producao_anterior = producao_base.loc[setores_nf].rename(
        "producao_anterior"
    )

    denominador_v = depreciacao * float(producao_anterior.sum())

    if denominador_v <= 0.0:
        raise ValueError("A produção NF do estado-base deve ser positiva.")
        
    v = fbcf_nf_total_pb / denominador_v

    estoque_capital_inicial = (v * producao_anterior).rename(
        "estoque_capital_inicial"
    )

    investimento_liquido_base = pd.Series(
        0.0,
        index=setores_nf,
        name="investimento_liquido_base",
    )

    investimento_reposicao_base = (
        depreciacao * estoque_capital_inicial
    ).rename("investimento_reposicao_base")

    investimento_bruto_base = (
        investimento_liquido_base + investimento_reposicao_base
    ).clip(lower=0.0).rename("investimento_bruto_base")

    if not np.isclose(float(fbcf_nf_pm.sum()), fbcf_nf_total_pm, atol=1e-6):
        raise RuntimeError("A FBCF NF PM não reproduziu a CEI.")
    if not np.isclose(
        float(investimento_bruto_base.sum()), fbcf_nf_total_pb, atol=1e-6
    ):
        raise RuntimeError("A regra de investimento não reproduziu a FBCF NF PB.")
    if not np.allclose(
        investimento_reposicao_base,
        depreciacao * estoque_capital_inicial,
        atol=1e-9,
    ):
        raise RuntimeError("A reposição não coincide com a depreciação do capital.")

    return {
        "v": v,
        "depreciacao": depreciacao,
        "setores_nf": setores_nf,
        "fbcf_nf_pm": fbcf_nf_pm,
        "fbcf_nf_pb": fbcf_nf_pb,
        "fbcf_nf_total_pm": fbcf_nf_total_pm,
        "fbcf_nf_total_pb": fbcf_nf_total_pb,
        "pesos_investimento_familias": pesos_investimento_familias,
        "fbcf_fixa_base": fbcf_fixa_base,
        "pesos_bens_capital_nf": pesos_bens_capital_nf,
        "producao_anterior": producao_anterior,
        "estoque_capital_inicial": estoque_capital_inicial,
        "investimento_liquido_base": investimento_liquido_base,
        "investimento_reposicao_base": investimento_reposicao_base,
        "investimento_bruto_base": investimento_bruto_base,
    }
