"""Cálculo econômico dos fluxos institucionais entregues à CEI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


INSTITUICOES_CEI = (
    "familias",
    "governo",
    "firmas_financeiras",
    "firmas_nao_financeiras",
    "setor_externo",
)


def estruturar_fluxos_cei(
    *,
    distribuicao: Mapping[str, Any],
    importacoes_nominais: float,
    exportacoes_nominais: float,
    consumo_governo: float,
    fbcf: Mapping[str, float],
    estoques: Mapping[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
    """Converte fluxos econômicos finais em lançamentos institucionais.

    Esta função é compartilhada pelo período zero e pelos períodos simulados.
    Ela não conhece posições de linhas ou colunas da matriz CEI.
    """

    juros_recebidos = distribuicao["juros_recebidos"]
    juros_pagos = distribuicao["juros_pagos"]

    return {
        "va": {
            "entradas": {
                "firmas_financeiras": float(
                    distribuicao["va_planejado_ff"]
                    + distribuicao["impostos_produtos_ff"]
                ),
                "firmas_nao_financeiras": float(
                    distribuicao["va_planejado_nf"]
                    + distribuicao["impostos_produtos_nf"]
                ),
                "setor_externo": float(importacoes_nominais),
            },
            "saidas": {"setor_externo": float(exportacoes_nominais)},
        },
        "salarios": {
            "entradas": {
                "familias": float(
                    distribuicao["salarios_ff"] + distribuicao["salarios_nf"]
                )
            },
            "saidas": {
                "firmas_financeiras": float(distribuicao["salarios_ff"]),
                "firmas_nao_financeiras": float(distribuicao["salarios_nf"]),
            },
        },
        "contribuicoes_efetivas": {
            "entradas": {
                "familias": float(
                    distribuicao["contribuicoes_efetivas_ff"]
                    + distribuicao["contribuicoes_efetivas_nf"]
                )
            },
            "saidas": {
                "firmas_financeiras": float(
                    distribuicao["contribuicoes_efetivas_ff"]
                ),
                "firmas_nao_financeiras": float(
                    distribuicao["contribuicoes_efetivas_nf"]
                ),
            },
        },
        "impostos_produtos": {
            "entradas": {
                "governo": float(
                    distribuicao["impostos_produtos_ff"]
                    + distribuicao["impostos_produtos_nf"]
                )
            },
            "saidas": {
                "firmas_financeiras": float(
                    distribuicao["impostos_produtos_ff"]
                ),
                "firmas_nao_financeiras": float(
                    distribuicao["impostos_produtos_nf"]
                ),
            },
        },
        "outros_impostos": {
            "entradas": {
                "governo": float(
                    distribuicao["outros_impostos_ff"]
                    + distribuicao["outros_impostos_nf"]
                )
            },
            "saidas": {
                "firmas_financeiras": float(
                    distribuicao["outros_impostos_ff"]
                ),
                "firmas_nao_financeiras": float(
                    distribuicao["outros_impostos_nf"]
                ),
            },
        },
        "juros": {
            "entradas": {
                nome: float(juros_recebidos.loc[nome])
                for nome in INSTITUICOES_CEI
            },
            "saidas": {
                nome: float(juros_pagos.loc[nome])
                for nome in INSTITUICOES_CEI
            },
        },
        "dividendos": {
            "entradas": {
                "familias": float(distribuicao["dividendos_familias"]),
                "setor_externo": float(distribuicao["dividendos_exterior"]),
            },
            "saidas": {
                "firmas_financeiras": float(distribuicao["dividendos_ff"]),
                "firmas_nao_financeiras": float(distribuicao["dividendos_nf"]),
            },
        },
        "ir": {
            "entradas": {
                "governo": float(
                    distribuicao["ir_familias"]
                    + distribuicao["ir_ff"]
                    + distribuicao["ir_nf"]
                )
            },
            "saidas": {
                "familias": float(distribuicao["ir_familias"]),
                "firmas_financeiras": float(distribuicao["ir_ff"]),
                "firmas_nao_financeiras": float(distribuicao["ir_nf"]),
            },
        },
        "contribuicoes_sociais": {
            "entradas": {
                "governo": float(distribuicao["previdencia_publica"]),
                "firmas_financeiras": float(
                    distribuicao["previdencia_privada"]
                ),
            },
            "saidas": {
                "familias": float(distribuicao["previdencia_familias"])
            },
        },
        "beneficios": {
            "entradas": {"familias": float(distribuicao["beneficios"])},
            "saidas": {"governo": float(distribuicao["beneficios"])},
        },
        "aposentadorias": {
            "entradas": {"familias": float(distribuicao["aposentadorias"])},
            "saidas": {
                "governo": float(distribuicao["aposentadorias_governo"]),
                "firmas_financeiras": float(distribuicao["aposentadorias_ff"]),
            },
        },
        "outras_transferencias": {
            "entradas": {
                "familias": float(
                    distribuicao["outras_transferencias_familias"]
                ),
                "governo": float(
                    distribuicao["outras_transferencias_governo"]
                ),
            },
            "saidas": {
                "firmas_financeiras": float(
                    distribuicao["outras_transferencias_ff"]
                ),
                "firmas_nao_financeiras": float(
                    distribuicao["outras_transferencias_nf"]
                ),
                "setor_externo": float(
                    distribuicao["outras_transferencias_exterior"]
                ),
            },
        },
        "consumo": {
            "entradas": {},
            "saidas": {
                "familias": float(distribuicao["consumo_nominal"]),
                "governo": float(consumo_governo),
            },
        },
        "fbcf": {
            "entradas": {},
            "saidas": {nome: float(valor) for nome, valor in fbcf.items()},
        },
        "estoques": {
            "entradas": {},
            "saidas": {nome: float(valor) for nome, valor in estoques.items()},
        },
    }


def calcular_fluxos_cei(
    *,
    distribuicao_pre_mercado: Mapping[str, Any],
    agregados_realizados: pd.DataFrame,
    impostos_produtos_total_realizado: float,
    fbcf_familias: float,
    fbcf_nf_nominal_realizada: float,
    fbcf_fixa_nominal: pd.Series,
    importacoes_nominais: float,
    exportacoes_nominais: pd.Series,
    governo_nominal: pd.Series,
    variacao_estoques_nominal: float,
    condicoes_iniciais: Mapping[str, Any],
    calibracoes: Mapping[str, Any],
    CONFIG: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """Calcula os fluxos econômicos realizados da CEI sem montar a matriz."""

    setor_financeiro = int(CONFIG["setor_financeiro"])
    va_realizado_ff = float(
        agregados_realizados["valor_adicionado"].iloc[setor_financeiro]
    )
    va_realizado_nf = float(
        agregados_realizados["valor_adicionado"].sum() - va_realizado_ff
    )

    parcela_impostos_ff = calibracoes["cei"]["parametros"][
        "parcela_impostos_produtos_ff"
    ]
    impostos_produtos_ff = float(
        parcela_impostos_ff * impostos_produtos_total_realizado
    )
    impostos_produtos_nf = float(
        impostos_produtos_total_realizado - impostos_produtos_ff
    )

    distribuicao_realizada = {
        **distribuicao_pre_mercado,
        "va_planejado_ff": va_realizado_ff,
        "va_planejado_nf": va_realizado_nf,
        "impostos_produtos_ff": impostos_produtos_ff,
        "impostos_produtos_nf": impostos_produtos_nf,
    }

    fbcf_fixa_total_base = float(
        calibracoes["investimento"]["fbcf_fixa_base"].sum()
    )
    fbcf_fixa_total = float(fbcf_fixa_nominal.sum())
    fator_fbcf_fixa = (
        1.0
        if np.isclose(fbcf_fixa_total_base, 0.0)
        else fbcf_fixa_total / fbcf_fixa_total_base
    )
    fbcf_base = condicoes_iniciais["fbcf_fixa_cei_base"]
    fbcf = {
        "familias": float(fbcf_familias),
        "governo": float(fbcf_base["governo"] * fator_fbcf_fixa),
        "firmas_financeiras": float(
            fbcf_base["firmas_financeiras"] * fator_fbcf_fixa
        ),
        "firmas_nao_financeiras": float(fbcf_nf_nominal_realizada),
        "setor_externo": float(
            fbcf_base["setor_externo"] * fator_fbcf_fixa
        ),
    }
    estoques = {
        "familias": 0.0,
        "governo": 0.0,
        "firmas_financeiras": 0.0,
        "firmas_nao_financeiras": float(variacao_estoques_nominal),
        "setor_externo": 0.0,
    }

    return estruturar_fluxos_cei(
        distribuicao=distribuicao_realizada,
        importacoes_nominais=float(importacoes_nominais),
        exportacoes_nominais=float(exportacoes_nominais.sum()),
        consumo_governo=float(governo_nominal.sum()),
        fbcf=fbcf,
        estoques=estoques,
    )
