"""Inicialização dos estoques financeiros do ABM."""

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import COLUNAS_SETORES, L


def inicializar_financeiro_abm(
    valores_cei: pd.DataFrame,
    cfg: dict,
    taxa_juros_nominal: float,
) -> dict:
    """Infere os estoques financeiros iniciais a partir dos juros da CEI."""

    taxa_juros_real = float(cfg["taxa_juros_real"])
    if taxa_juros_real <= 0.0:
        raise ValueError("taxa_juros_real deve ser positiva.")

    fracao_reavaliacao_financeira = float(
        cfg["fracao_reavaliacao_financeira"]
    )
    if not 0.0 <= fracao_reavaliacao_financeira <= 1.0:
        raise ValueError(
            "fracao_reavaliacao_financeira deve estar entre 0 e 1."
        )

    juros_recebidos_base = pd.Series(
        {
            nome: float(valores_cei.iat[L["juros"], entrada])
            for nome, (entrada, saida) in COLUNAS_SETORES.items()
        },
        name="juros_recebidos",
    )
    juros_pagos_base = pd.Series(
        {
            nome: float(valores_cei.iat[L["juros"], saida])
            for nome, (entrada, saida) in COLUNAS_SETORES.items()
        },
        name="juros_pagos",
    )
    juros_liquidos_base = (
        juros_recebidos_base - juros_pagos_base
    ).rename("juros_liquidos")

    if not np.isclose(juros_liquidos_base.sum(), 0.0, atol=1e-9):
        raise RuntimeError("Os juros líquidos da CEI-base não somam zero.")

    ativos_financeiros = (
        juros_recebidos_base / taxa_juros_nominal
    ).rename("ativos_financeiros")
    passivos_financeiros = (
        juros_pagos_base / taxa_juros_nominal
    ).rename("passivos_financeiros")

    if np.any(ativos_financeiros < 0.0) or np.any(passivos_financeiros < 0.0):
        raise RuntimeError(
            "A CEI-base gerou ativo ou passivo financeiro negativo."
        )
    estoque_financeiro = (
        ativos_financeiros - passivos_financeiros
    ).rename("estoque_financeiro")

    if not np.isclose(
        ativos_financeiros.sum(),
        passivos_financeiros.sum(),
        atol=1e-8,
    ):
        raise RuntimeError(
            "Ativos e passivos financeiros iniciais não possuem o mesmo total."
        )
    if not np.isclose(estoque_financeiro.sum(), 0.0, atol=1e-8):
        raise RuntimeError("Os estoques financeiros iniciais não somam zero.")

    return {
        "taxa_juros_real": taxa_juros_real,
        "fracao_reavaliacao_financeira": fracao_reavaliacao_financeira,
        "juros_recebidos_base": juros_recebidos_base,
        "juros_pagos_base": juros_pagos_base,
        "juros_liquidos_base": juros_liquidos_base,
        "ativos_financeiros": ativos_financeiros,
        "passivos_financeiros": passivos_financeiros,
        "estoque_financeiro": estoque_financeiro,
    }
