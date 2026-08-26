"""Inicialização do estado agregado persistente da trajetória."""

from __future__ import annotations

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import C, L
from financeiro.financeiro_abm import inicializar_financeiro_abm


def inicializar_estado(
    *,
    condicoes_iniciais: dict,
    calibracoes: dict,
    firmas: dict,
    CONFIG: dict,
    CONFIG_ABM: dict,
    seed=None,
) -> dict:
    """Cria somente o estado agregado necessário para executar o período 1."""

    # ``calibracoes`` e ``firmas`` integram a fronteira arquitetural da função.
    # O estado agregado atualmente herdado não duplica nenhum valor deles.
    del calibracoes, firmas

    inflacao = float(CONFIG["a0"])
    if inflacao <= -1.0:
        raise ValueError("inflacao deve ser maior que -1.")
    taxa_crescimento_populacional = float(
        CONFIG["taxa_crescimento_populacional"]
    )
    if taxa_crescimento_populacional < 0.0:
        raise ValueError(
            "taxa_crescimento_populacional não pode ser negativa."
        )

    taxa_juros_nominal = (
        (1.0 + float(CONFIG["taxa_juros_real"])) * (1.0 + inflacao) - 1.0
    )
    financeiro_inicial = inicializar_financeiro_abm(
        condicoes_iniciais["valores_cei"],
        CONFIG,
        taxa_juros_nominal,
    )

    setores = list(condicoes_iniciais["setores"])
    pc_anterior = pd.Series(
        1.0,
        index=setores,
        name="preco_comprador_anterior",
    )
    poupanca_familias_anterior = float(
        condicoes_iniciais["valores_cei"].iloc[1:13, C["familias_e"]].sum()
        - condicoes_iniciais["valores_cei"].iloc[1:9, C["familias_s"]].sum()
        - condicoes_iniciais["valores_cei"].iat[
            L["consumo"],
            C["familias_s"],
        ]
    )

    seed_qualidade = (
        CONFIG_ABM["semente_qualidade"] if seed is None else seed
    )
    sequencia_seed = np.random.SeedSequence(seed_qualidade)
    seed_qualidade_firmas, seed_produtividade_firmas = (
        sequencia_seed.spawn(2)
    )

    return {
        "macro": {
            "inflacao": inflacao,
            "indice_populacao": 1.0,
            "indice_salarios": 1.0 + inflacao,
            "indice_cambio": 1.0 + inflacao,
            "taxa_juros_nominal": taxa_juros_nominal,
        },
        "precos": {
            "indice_precos_anterior": 1.0,
            "pc_anterior": pc_anterior,
            "pc_anterior_2": pc_anterior / (1.0 + inflacao),
        },
        "familias": {
            "poupanca_familias_anterior": poupanca_familias_anterior,
        },
        "financeiro": {
            "ativos_financeiros": financeiro_inicial[
                "ativos_financeiros"
            ].copy(deep=True),
            "passivos_financeiros": financeiro_inicial[
                "passivos_financeiros"
            ].copy(deep=True),
        },
        "aleatoriedade": {
            "rng_qualidade": np.random.default_rng(seed_qualidade_firmas),
            "rng_produtividade_idiossincratica": np.random.default_rng(
                seed_produtividade_firmas
            ),
        },
    }
