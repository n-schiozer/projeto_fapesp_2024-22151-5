"""Inicialização do investimento das firmas não financeiras do ABM."""

import pandas as pd


def inicializar_investimento_nf(
    calibracao_nf: dict,
    ano: int,
    inicializacao_investimento_nf: str,
) -> dict:
    """Monta o estado inicial do acelerador de investimento das firmas NF."""

    beta_investimento_nf = float(calibracao_nf["beta"])
    v_investimento_nf = float(calibracao_nf["v"])
    depreciacao_capital_nf = float(calibracao_nf["depreciacao"])
    setores_nf = calibracao_nf["setores_nf"]
    producao_nf_corrente = (
        calibracao_nf["producao_real"].loc[ano, setores_nf].copy()
    )
    investimento_nf_base_por_investidor = calibracao_nf[
        "investimento_nf_base_por_investidor"
    ].copy()

    if inicializacao_investimento_nf == "estacionaria":
        # Supõe que a produção observada em t=0 também ocorreu em t=-1.
        # Portanto, ΔY_e,0 = 0 e todo o investimento bruto observado no
        # ano-base é reposição da depreciação: K_0 = I_0 / depreciação.
        producao_nf_anterior = producao_nf_corrente.copy()
        estoque_capital_nf = (
            investimento_nf_base_por_investidor / depreciacao_capital_nf
        ).rename("estoque_capital_nf_base_estacionario")
        investimento_liquido_nf_base = pd.Series(
            0.0,
            index=setores_nf,
            name="investimento_liquido_nf_base",
        )
        investimento_reposicao_nf_base = (
            depreciacao_capital_nf * estoque_capital_nf
        ).rename("investimento_reposicao_nf_base")
    elif inicializacao_investimento_nf == "historica":
        # Usa 2019 e 2020 para carregar a variação observada da pandemia para
        # a primeira expectativa da simulação.
        producao_nf_anterior = (
            calibracao_nf["producao_real"].loc[ano - 1, setores_nf].copy()
        )
        estoque_capital_nf = calibracao_nf["estoque_capital_nf_base"].copy()
        investimento_liquido_nf_base = calibracao_nf[
            "investimento_liquido_nf_base_por_investidor"
        ].copy()
        investimento_reposicao_nf_base = calibracao_nf[
            "investimento_reposicao_nf_base_por_investidor"
        ].copy()
    else:
        raise ValueError(
            "inicializacao_investimento_nf deve ser "
            "'estacionaria' ou 'historica'."
        )
    pesos_bens_capital_nf = calibracao_nf["pesos_bens_capital_nf"].copy()

    return {
        "beta_investimento_nf": beta_investimento_nf,
        "v_investimento_nf": v_investimento_nf,
        "depreciacao_capital_nf": depreciacao_capital_nf,
        "setores_nf": setores_nf,
        "producao_nf_corrente": producao_nf_corrente,
        "investimento_nf_base_por_investidor": investimento_nf_base_por_investidor,
        "inicializacao_investimento_nf": inicializacao_investimento_nf,
        "producao_nf_anterior": producao_nf_anterior,
        "estoque_capital_nf": estoque_capital_nf,
        "investimento_liquido_nf_base": investimento_liquido_nf_base,
        "investimento_reposicao_nf_base": investimento_reposicao_nf_base,
        "pesos_bens_capital_nf": pesos_bens_capital_nf,
    }
