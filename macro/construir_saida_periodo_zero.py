"""Construção read-only da fotografia econômica inicial da trajetória."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from agentes.agregar_firmas import agregar_firmas
from contabilidade.estrutura_cei import C, COLUNAS_SETORES, L, VA
from financeiro.financeiro_abm import inicializar_financeiro_abm
from macro.executar_periodo import construir_diagnostico_capacidade_setorial


def construir_saida_periodo_zero(
    *,
    firmas: dict[str, Any],
    importados: dict[str, Any],
    estado: dict[str, Any],
    condicoes_iniciais: dict[str, Any],
    calibracoes: dict[str, Any],
    CONFIG: dict[str, Any],
    CONFIG_ABM: dict[str, Any],
) -> dict[str, Any]:
    """Retorna a fotografia de t=0 sem realizar a dinâmica de um período."""

    # Integram a fronteira comum com executar_periodo, mas a fotografia de
    # ano-base não toma decisões dos importados nem consome aleatoriedade.
    del importados, CONFIG_ABM

    setores = list(condicoes_iniciais["setores"])
    valores_cei = condicoes_iniciais["valores_cei"]

    consumo_nominal_base = float(
        valores_cei.iat[L["consumo"], C["familias_s"]]
    )
    fbcf_familias_base = float(
        valores_cei.iat[L["fbcf"], C["familias_s"]]
    )
    estoques_base = (
        condicoes_iniciais["tru_base"].stocks_investment_sector.iloc[:, 0]
        .reindex(setores)
        .rename("estoques_base")
    )

    razao_estoque_producao = float(CONFIG["razao_estoque_producao"])
    if razao_estoque_producao < 0.0:
        raise ValueError("razao_estoque_producao não pode ser negativa.")
    setores_com_estoques = (estoques_base.abs() > 1e-9).rename(
        "setor_com_estoques"
    )
    producao_estoques_base = (
        calibracoes["investimento"]["legado"]["producao_real"]
        .loc[CONFIG["ano"], setores]
        .copy()
    )
    estoque_referencia = (
        razao_estoque_producao
        * producao_estoques_base
        * setores_com_estoques.astype(float)
    ).rename("estoque_referencia")
    estoque_ciclico = pd.Series(
        0.0,
        index=setores,
        name="estoque_ciclico",
    )
    estoque_real = (estoque_referencia + estoque_ciclico).rename(
        "estoque_real"
    )

    # Os fluxos observados de juros não pertencem ao estado persistente.
    financeiro_base = inicializar_financeiro_abm(
        valores_cei,
        CONFIG,
        estado["macro"]["taxa_juros_nominal"],
    )
    juros_recebidos_base = financeiro_base["juros_recebidos_base"]
    juros_pagos_base = financeiro_base["juros_pagos_base"]
    juros_liquidos_base = financeiro_base["juros_liquidos_base"]
    estoque_financeiro_inicial = (
        estado["financeiro"]["ativos_financeiros"]
        - estado["financeiro"]["passivos_financeiros"]
    ).rename("estoque_financeiro")

    capacidade_base = {
        nome: float(
            valores_cei.iloc[1:16, entrada].sum()
            - valores_cei.iloc[1:16, saida].sum()
        )
        for nome, (entrada, saida) in COLUNAS_SETORES.items()
    }
    cei_historico_zero = condicoes_iniciais["cei_original"].copy(deep=True)
    for nome, (entrada, _) in COLUNAS_SETORES.items():
        cei_historico_zero.iloc[L["capacidade"], entrada] = capacidade_base[nome]

    pib_base = float(
        condicoes_iniciais["va_base"].loc[VA["total"]].sum()
        + (
            condicoes_iniciais["taxa_impostos"]
            * condicoes_iniciais["demanda_final_base"]
        ).sum()
    )
    pea_base = float(calibracoes["cei"]["parametros"]["pea"])
    macro = {
        "periodo": 0,
        "ano": CONFIG["ano"],
        "indice_populacao": 1.0,
        "pea": pea_base,
        "indice_precos": 1.0,
        "inflacao": estado["macro"]["inflacao"],
        "indice_salarios": 1.0,
        "indice_cambio": 1.0,
        "taxa_juros_nominal": estado["macro"]["taxa_juros_nominal"],
        "pib_real": pib_base,
        "pib_nominal": pib_base,
        "emprego": float(
            condicoes_iniciais["va_base"]
            .loc["Fator trabalho (ocupações)"]
            .sum()
        ),
        "taxa_desemprego": CONFIG["taxa_desemprego_inicial"],
        "consumo_real": float(condicoes_iniciais["consumo_base"].sum()),
        "consumo_nominal": consumo_nominal_base / pib_base,
        "poupanca_familias_nominal": estado["familias"][
            "poupanca_familias_anterior"
        ],
        "fbcf_familias_nominal": fbcf_familias_base,
        "fbcf_nf_real": float(
            calibracoes["investimento"]["investimento_nf_base"].sum()
        ),
        "fbcf_nf_nominal": float(
            calibracoes["investimento"]["investimento_nf_base"].sum()
        ),
        "fbcf_fixa_nominal": float(
            calibracoes["investimento"]["fbcf_fixa_base"].sum()
        ),
        "variacao_estoques_real": float(estoques_base.sum()),
        "variacao_estoques_nominal": float(estoques_base.sum()),
        "variacao_autonoma_estoques_real": float(estoques_base.sum()),
        "variacao_ciclica_estoques_real": 0.0,
        "estoque_real": float(estoque_real.sum()),
        "investimento_liquido_nf_real": float(
            calibracoes["investimento"]["investimento_liquido_base"].sum()
        ),
        "investimento_reposicao_nf_real": float(
            calibracoes["investimento"]["investimento_reposicao_base"].sum()
        ),
        "ajuste_piso_investimento_nf_real": 0.0,
        "estoque_capital_nf_real": float(
            calibracoes["investimento"]["estoque_capital_inicial"].sum()
        ),
        "setores_no_piso_investimento_nf": int(
            (
                calibracoes["investimento"]["investimento_liquido_base"]
                + calibracoes["investimento"]["investimento_reposicao_base"]
                < 0.0
            ).sum()
        ),
        "residuo_consumo": 0.0,
        "iteracoes_consumo": 0,
        "deficit_governo": -capacidade_base["governo"] / pib_base,
        "saldo_setor_externo": capacidade_base["setor_externo"] / pib_base,
        "discrepancia_cei": sum(capacidade_base.values()),
    }

    diagnostico_capacidade = construir_diagnostico_capacidade_setorial(
        firmas=firmas,
        setores=setores,
        periodo=0,
        depreciacao=None,
    )
    zeros_financeiros = pd.Series(
        0.0,
        index=list(COLUNAS_SETORES),
    )
    # agregar_firmas ainda chama métodos contábeis que materializam atributos
    # derivados. A cópia é somente local e impede que a fotografia altere os
    # agentes vivos; nenhum segundo estado é persistido.
    agregados_firmas = agregar_firmas(deepcopy(firmas), setores)

    return {
        "macro": macro,
        "setores": {
            "agregados_firmas": agregados_firmas,
            "precos": {
                "precos_comprador": pd.Series(
                    1.0, index=setores, name="preco_comprador"
                ),
                "precos_basicos": pd.Series(
                    1.0, index=setores, name="preco_basico"
                ),
                "precos_importacoes": pd.Series(
                    1.0, index=setores, name="preco_importacoes"
                ),
                "precos_comprador_esperados": pd.Series(
                    1.0, index=setores, name="preco_comprador_esperado"
                ),
                "inflacao_precos_setorial": pd.Series(
                    0.0, index=setores, name="inflacao_pc_setorial"
                ),
            },
            "investimento_capital_estoques": {
                "investimento_nf_real": calibracoes["investimento"][
                    "investimento_nf_base"
                ],
                "investimento_nf_nominal": calibracoes["investimento"][
                    "fbcf_nf_pm"
                ],
                "fbcf_fixa_nominal": calibracoes["investimento"][
                    "fbcf_fixa_base"
                ],
                "variacao_estoques_real": estoques_base,
                "variacao_estoques_nominal": estoques_base,
                "variacao_autonoma_estoques_real": estoques_base,
                "variacao_ciclica_estoques_real": pd.Series(
                    0.0, index=setores
                ),
                "estoque_real": estoque_real,
                "estoque_referencia_real": estoque_referencia,
                "estoque_ciclico_real": estoque_ciclico,
                "investimento_nf_por_setor_investidor": calibracoes[
                    "investimento"
                ]["investimento_bruto_base"],
                "estoque_capital_nf_real": calibracoes["investimento"][
                    "estoque_capital_inicial"
                ],
            },
        },
        "cei": {"cei": cei_historico_zero},
        "financeiro": {
            "capacidade_financiamento": capacidade_base,
            "estoque_financeiro": estoque_financeiro_inicial,
            "aquisicao_ativos_financeiros": zeros_financeiros.rename(
                "aquisicao_ativos_financeiros"
            ),
            "emissao_passivos_financeiros": zeros_financeiros.rename(
                "emissao_passivos_financeiros"
            ),
            "juros_liquidos": juros_liquidos_base,
            "juros_recebidos": juros_recebidos_base,
            "juros_pagos": juros_pagos_base,
            "reavaliacao_financeira": zeros_financeiros.rename(
                "reavaliacao_financeira"
            ),
        },
        "diagnosticos": {"capacidade_setorial": diagnostico_capacidade},
        "preco_capital": 1.0,
    }
