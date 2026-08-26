"""Regras legadas de investimento, estoques e demanda autônoma do ABM.

Nada neste módulo altera as regras econômicas vigentes; ele apenas separa as
decisões por função econômica para tornar o ciclo temporal auditável.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


SETOR_CONSTRUCAO = "F - Construção"


def inicializar_taxas_retorno_firmas(
    firmas: dict,
    setores_nf: pd.Index | list[str],
    depreciacao: float,
    taxa_juros_real: float,
    preco_capital: float = 1.0,
) -> None:
    """Calibra o retorno de referência do ano-base para as firmas NF.

    Esta é a mesma inicialização usada no laboratório antes do primeiro
    período. Ela fornece a defasagem necessária para a regra de investimento
    dos participantes de leilão, sem alterar estoques de capital ou produção.
    """

    if preco_capital <= 0.0:
        raise ValueError("preco_capital deve ser positivo.")

    setores_nf = set(setores_nf)
    for firma in firmas.values():
        firma.calcular_taxa_retorno_observada(
            preco_capital=preco_capital,
            depreciacao=depreciacao,
            taxa_juros_real=taxa_juros_real,
        )

    parametros_setoriais = {}
    for setor in setores_nf:
        firmas_setor = [firma for firma in firmas.values() if firma.setor == setor]
        capital = sum(firma.estoque_capital_real for firma in firmas_setor)
        if capital <= 0.0:
            continue
        eob = sum(firma.eob_misto_realizado for firma in firmas_setor)
        retorno_observado = eob / (preco_capital * capital) - depreciacao
        parametros_setoriais[setor] = retorno_observado - taxa_juros_real

    for firma in firmas.values():
        if firma.setor not in parametros_setoriais:
            continue
        firma.taxa_retorno_parametro = parametros_setoriais[firma.setor]
        firma.calcular_taxa_retorno_observada(
            preco_capital=preco_capital,
            depreciacao=depreciacao,
            taxa_juros_real=taxa_juros_real,
        )
        firma.taxa_retorno_ajustada_anterior = firma.taxa_retorno_ajustada


def calcular_fbcf_familias(
    poupanca_familias_nominal_anterior: float,
    pc_anterior: pd.Series,
    pc_atual: pd.Series,
    setor_construcao: str,
    prop_invest_fbcf_familias: float,
) -> dict[str, float]:
    """Calcula a FBCF familiar corrente a partir da poupança de t-1."""

    fbcf_real = (
        prop_invest_fbcf_familias
        * poupanca_familias_nominal_anterior
        / float(pc_anterior.loc[setor_construcao])
    )
    return {
        "fbcf_familias_real": fbcf_real,
        "fbcf_familias_nominal": fbcf_real * float(pc_atual.loc[setor_construcao]),
    }


def calcular_estoques_legado_periodo(
    producao_corrente: pd.Series,
    producao_anterior: pd.Series,
    producao_base: pd.Series,
    variacao_autonoma: pd.Series,
    estoque_referencia: pd.Series,
    estoque_ciclico: pd.Series,
    setores_com_estoques: pd.Series,
    beta: float,
    razao_estoque_producao: float,
    velocidade_ajuste: float,
) -> dict:
    """Aplica o ajuste parcial de estoques legado, sem usar vendas realizadas."""

    producao_esperada = (
        producao_corrente + beta * (producao_corrente - producao_anterior)
    ).clip(lower=0.0).rename("producao_esperada_estoques")
    variacao_autonoma_periodo = variacao_autonoma.copy()
    estoque_referencia_periodo = estoque_referencia.copy()
    estoque_ciclico_desejado = (
        razao_estoque_producao
        * (producao_esperada - producao_base)
        * setores_com_estoques.astype(float)
    ).rename("estoque_ciclico_desejado")
    variacao_ciclica = (
        velocidade_ajuste * (estoque_ciclico_desejado - estoque_ciclico)
    ).rename("variacao_ciclica_estoques")
    variacao_ciclica = variacao_ciclica.clip(
        lower=-(estoque_referencia_periodo + estoque_ciclico)
    )
    variacao_ciclica.loc[~setores_com_estoques] = 0.0
    estoque_ciclico_periodo = (
        estoque_ciclico + variacao_ciclica
    ).rename("estoque_ciclico")
    variacao_real = (
        variacao_autonoma_periodo + variacao_ciclica
    ).rename("variacao_estoques_real")
    variacao_real.loc[~setores_com_estoques] = 0.0
    estoque_real_periodo = (
        estoque_referencia_periodo + estoque_ciclico_periodo
    ).rename("estoque_real")
    if np.any(estoque_real_periodo < -1e-9):
        raise RuntimeError("Estoque físico negativo no período.")
    return {
        "producao_esperada": producao_esperada,
        "variacao_autonoma": variacao_autonoma_periodo,
        "variacao_ciclica": variacao_ciclica,
        "variacao_real": variacao_real,
        "estoque_referencia": estoque_referencia_periodo,
        "estoque_ciclico": estoque_ciclico_periodo,
        "estoque_real": estoque_real_periodo,
    }


def calcular_investimento_nf_periodo(
    producao_corrente: pd.Series,
    producao_anterior: pd.Series,
    estoque_capital_anterior: pd.Series,
    pesos_bens_capital: pd.Series,
    pc: pd.Series,
    beta: float,
    v: float,
    depreciacao: float,
) -> dict:
    """Calcula acelerador, reposição, piso zero e estoque de capital das NF."""

    variacao_producao_esperada = (
        beta * (producao_corrente - producao_anterior)
    ).rename("variacao_producao_esperada_nf")
    investimento_liquido = (
        v * variacao_producao_esperada
    ).rename("investimento_liquido_nf")
    investimento_reposicao = (
        depreciacao * estoque_capital_anterior
    ).rename("investimento_reposicao_nf")
    investimento_sem_piso = (
        investimento_liquido + investimento_reposicao
    ).rename("investimento_nf_sem_piso")
    por_investidor = investimento_sem_piso.clip(lower=0.0).rename(
        "investimento_nf_por_setor_investidor"
    )
    estoque_capital = (
        por_investidor + (1.0 - depreciacao) * estoque_capital_anterior
    ).rename("estoque_capital_nf")
    investimento_real = (
        pesos_bens_capital * float(por_investidor.sum())
    ).rename("investimento_nf_real")
    investimento_nominal = (investimento_real * pc).rename(
        "investimento_nf_nominal"
    )
    return {
        "variacao_producao_esperada": variacao_producao_esperada,
        "investimento_liquido": investimento_liquido,
        "investimento_reposicao": investimento_reposicao,
        "investimento_sem_piso": investimento_sem_piso,
        "por_investidor": por_investidor,
        "real": investimento_real,
        "nominal": investimento_nominal,
        "estoque_capital": estoque_capital,
    }


def atualizar_demandas_autonomas(
    governo_anterior: pd.Series,
    fbcf_fixa_anterior: pd.Series,
    exportacoes_anterior: pd.Series,
    pc: pd.Series,
    pc_anterior: pd.Series,
    periodo: int,
    periodo_choque: int,
    choque_permanente: bool,
    multiplicador_governo: float,
    multiplicador_investimento: float,
    multiplicador_exportacoes: float,
) -> dict[str, pd.Series]:
    """Atualiza G, FBCF fixa e X pela regra nominal legada de choque."""

    # Os dois ramos eram intencionalmente iguais no modelo estável e ficam
    # separados aqui apenas para preservar a semântica configurável existente.
    variacao_precos = pc / pc_anterior
    governo = governo_anterior * variacao_precos
    fbcf_fixa = fbcf_fixa_anterior * variacao_precos
    exportacoes = exportacoes_anterior * variacao_precos
    if periodo == periodo_choque:
        governo *= multiplicador_governo
        fbcf_fixa *= multiplicador_investimento
        exportacoes *= multiplicador_exportacoes
    elif not choque_permanente and periodo == periodo_choque + 1:
        governo /= multiplicador_governo
        fbcf_fixa /= multiplicador_investimento
        exportacoes /= multiplicador_exportacoes
    return {
        "governo_nominal": governo,
        "fbcf_fixa_nominal": fbcf_fixa,
        "exportacoes_nominais": exportacoes,
    }


def montar_investimento_e_cei_legado(
    fbcf_fixa_nominal: pd.Series,
    variacao_estoques_real: pd.Series,
    fbcf_familias_nominal: float,
    pesos_investimento_familias: pd.Series,
    investimento_nf_nominal: pd.Series,
    pc: pd.Series,
    fbcf_fixa_base: pd.Series,
    fbcf_fixa_cei_base: dict,
) -> dict:
    """Monta a FBCF por produto e os fluxos fixos institucionais legados."""

    variacao_estoques_nominal = (variacao_estoques_real * pc).rename(
        "variacao_estoques_nominal"
    )
    investimento_nominal = (
        fbcf_fixa_nominal
        + variacao_estoques_nominal
        + pesos_investimento_familias * fbcf_familias_nominal
        + investimento_nf_nominal
    )
    total_base = float(fbcf_fixa_base.sum())
    total_atual = float(fbcf_fixa_nominal.sum())
    if np.isclose(total_base, 0.0):
        if not np.isclose(total_atual, 0.0):
            raise RuntimeError("A FBCF fixa partiu de zero e tornou-se não nula.")
        fator_fbcf_fixa = 1.0
    else:
        fator_fbcf_fixa = total_atual / total_base
    investimentos_fixos_cei = {
        "fbcf_governo": fbcf_fixa_cei_base["governo"] * fator_fbcf_fixa,
        "fbcf_firmas_financeiras": (
            fbcf_fixa_cei_base["firmas_financeiras"] * fator_fbcf_fixa
        ),
        "fbcf_setor_externo": (
            fbcf_fixa_cei_base["setor_externo"] * fator_fbcf_fixa
        ),
        "estoques_governo": 0.0,
        "estoques_firmas_financeiras": 0.0,
        "estoques_firmas_nao_financeiras": float(variacao_estoques_nominal.sum()),
        "estoques_setor_externo": 0.0,
    }
    return {
        "variacao_estoques_nominal": variacao_estoques_nominal,
        "investimento_nominal": investimento_nominal,
        "fbcf_fixa_total": total_atual,
        "investimentos_fixos_cei": investimentos_fixos_cei,
    }
