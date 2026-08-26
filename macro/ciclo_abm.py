"""Helpers simples do ciclo ABM.

As funções apenas isolam blocos antes inline em ``simul_``. Elas preservam as
equações, o timing e as estruturas públicas de resultado da versão estável.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import COLUNAS_SETORES, VA
from mercados.precos_firmas import agregar_precos_firmas


def calcular_precos_ex_ante(
    firmas: dict,
    importados: dict,
    setores: list[str],
    pc_anterior: pd.Series,
    pc_anterior_2: pd.Series,
    lambda_expectativa_precos: float,
    indice_salarios: float,
    indice_cambio: float,
    G: pd.DataFrame,
    Sd: pd.DataFrame,
    Sm: pd.DataFrame,
) -> dict:
    """Forma Pc esperado, preços das firmas, Pm e o Pc anterior ao mercado."""

    inflacao_pc_anterior = (pc_anterior / pc_anterior_2 - 1.0).rename(
        "inflacao_pc_anterior"
    )
    pc_esperado = (
        pc_anterior
        * (1.0 + lambda_expectativa_precos * inflacao_pc_anterior)
    ).rename("preco_comprador_esperado")
    if np.any(pc_esperado <= 0.0):
        raise RuntimeError("Preço esperado não positivo.")

    for firma in firmas.values():
        firma.atualizar_custo_e_preco(pc_esperado, indice_salarios)
    pb = agregar_precos_firmas(firmas, setores).rename("preco_basico")

    for setor, importado in importados.items():
        setor_homogeneo = any(
            firma.setor == setor and firma.regime == "leilao"
            for firma in firmas.values()
        )
        if setor_homogeneo:
            importado.custo_unitario_importado = (
                importado.custo_unitario_importado_base * indice_cambio
            )
            importado.preco_oferta_importado = (
                (1.0 + importado.markup_importado)
                * importado.custo_unitario_importado
            )
            importado.preco_importado = importado.preco_oferta_importado
        else:
            importado.preco_importado = (
                importado.preco_importado_base * indice_cambio
            )
    pm = pd.Series(
        {setor: importados[setor].preco_importado for setor in setores},
        index=setores,
        dtype=float,
        name="preco_importacoes",
    )
    precos = calcular_precos_realizados(pb, pm, G, Sd, Sm, pc_anterior)
    if np.any(precos["pc"] <= 0.0) or np.any(precos["pb"] <= 0.0):
        raise RuntimeError("Preço não positivo.")
    for firma in firmas.values():
        firma.registrar_custo_intermediario_realizado(precos["pc"])
    return {"pc_esperado": pc_esperado, **precos}


def montar_demandas_periodo(
    demanda_intermediaria_real: dict,
    consumo_cei: float,
    pesos_consumo: pd.Series,
    governo_nominal: pd.Series,
    investimento_nominal: pd.Series,
    exportacoes_nominais: pd.Series,
    pc: pd.Series,
    conversao_de_pm_pb: pd.DataFrame,
    investimento_nf_pb_nominal: pd.Series | None = None,
) -> dict:
    """Monta CI e demanda total nominal nas óticas PM e PB."""

    demanda_intermediaria_real = demanda_intermediaria_real

    demanda_total_pm_nominal = (
        consumo_cei * pesos_consumo
        + governo_nominal
        + investimento_nominal
        + exportacoes_nominais
        + demanda_intermediaria_real * pc
    ).rename("demanda_total_pm_nominal")

    investimento_nf_pb_nominal = (
        pd.Series(0.0, index=pc.index)
        if investimento_nf_pb_nominal is None
        else investimento_nf_pb_nominal.reindex(pc.index).fillna(0.0)
    )
    investimento_pb_nominal = (
        conversao_de_pm_pb @ investimento_nominal
        + investimento_nf_pb_nominal
    ).rename("investimento_pb_nominal")
    demanda_total_pb_nominal = (
        conversao_de_pm_pb @ (demanda_total_pm_nominal - investimento_nominal)
        + investimento_pb_nominal
    ).rename("demanda_total_pb_nominal")

    if (demanda_total_pb_nominal < -1e-8).any():
        setores_negativos = list(
            demanda_total_pb_nominal[demanda_total_pb_nominal < -1e-8].index
        )
        raise RuntimeError(
            "Orçamento básico negativo: " f"{setores_negativos}"
        )
    # O clip já fazia parte do comportamento estável.
    demanda_total_pb_nominal = demanda_total_pb_nominal.clip(lower=0.0)
    return {
        "demanda_intermediaria_real": demanda_intermediaria_real,
        "demanda_total_pm_nominal": demanda_total_pm_nominal,
        "demanda_total_pb_nominal": demanda_total_pb_nominal,
        "demandas_pm_nominal": {
            "consumo": (consumo_cei * pesos_consumo).rename("consumo"),
            "governo": governo_nominal.rename("governo"),
            "fbcf": investimento_nominal.rename("fbcf"),
            "exportacoes": exportacoes_nominais.rename("exportacoes"),
            "ci": (demanda_intermediaria_real * pc).rename("ci"),
        },
        "demandas_pb_nominal": {
            "consumo": (conversao_de_pm_pb @ (consumo_cei * pesos_consumo)).rename("consumo"),
            "governo": (conversao_de_pm_pb @ governo_nominal).rename("governo"),
            "fbcf": investimento_pb_nominal.rename("fbcf"),
            "exportacoes": (conversao_de_pm_pb @ exportacoes_nominais).rename("exportacoes"),
            "ci": (conversao_de_pm_pb @ (demanda_intermediaria_real * pc)).rename("ci"),
        },
    }


def calcular_precos_realizados(
    pb: pd.Series,
    pm: pd.Series,
    G: pd.DataFrame,
    Sd: pd.DataFrame,
    Sm: pd.DataFrame,
    pc_anterior: pd.Series,
) -> dict:
    """Aplica a fórmula existente de Pc aos Pb e Pm observados."""

    pc = (G @ (Sd @ pb + Sm @ pm)).rename("preco_comprador")
    inflacao_pc_setorial = (pc / pc_anterior - 1.0).rename(
        "inflacao_pc_setorial"
    )
    return {"pb": pb, "pm": pm, "pc": pc, "inflacao_pc_setorial": inflacao_pc_setorial}


def calcular_inflacao_periodo(
    consumo_base: pd.Series,
    pc: pd.Series,
    indice_precos_anterior: float,
) -> dict[str, float]:
    """Calcula o índice de preços da cesta-base e sua inflação corrente."""

    indice_precos = float(consumo_base.sum() / (consumo_base / pc).sum())
    return {
        "indice_precos": indice_precos,
        "inflacao": indice_precos / indice_precos_anterior - 1.0,
    }


def calcular_juros_periodo(
    ativos_financeiros: pd.Series,
    passivos_financeiros: pd.Series,
    indice_precos: float,
    indice_precos_anterior: float,
    taxa_juros_real: float,
    taxa_juros_nominal_anterior: float,
    inertia_pm: float,
    periodo: int,
) -> dict:
    """Corrige estoques financeiros e calcula juros nominais antes da CEI."""

    fator_inflacao = indice_precos / indice_precos_anterior
    ativos_corrigidos = (ativos_financeiros * fator_inflacao).rename(
        "ativos_financeiros"
    )
    passivos_corrigidos = (passivos_financeiros * fator_inflacao).rename(
        "passivos_financeiros"
    )
    inflacao = fator_inflacao - 1.0
    taxa_anterior = taxa_juros_nominal_anterior
    if periodo == 1:
        taxa_anterior = (1.0 + taxa_juros_real) * (1.0 + inflacao) - 1.0
    taxa_nominal = (1.0 - inertia_pm) * (
        (1.0 + taxa_juros_real) * (1.0 + inflacao) - 1.0
    ) + inertia_pm * taxa_anterior
    recebidos = (taxa_nominal * ativos_corrigidos).rename("juros_recebidos")
    pagos = (taxa_nominal * passivos_corrigidos).rename("juros_pagos")
    return {
        "ativos_corrigidos": ativos_corrigidos,
        "passivos_corrigidos": passivos_corrigidos,
        "taxa_juros_nominal": taxa_nominal,
        "juros_recebidos": recebidos,
        "juros_pagos": pagos,
        "juros_liquidos": (recebidos - pagos).rename("juros_liquidos"),
    }


def calcular_mercado_trabalho(
    emprego: float,
    pea: float,
    taxa_desemprego_base: float,
    a0: float,
    a1: float,
    a3: float,
) -> dict[str, float]:
    """Preserva a regra salarial que usa o desemprego corrente em t+1."""

    taxa_desemprego = max(0.0, (pea - emprego) / pea)
    
    if taxa_desemprego == 0.0:
        variacao_salarios = 1.1
    else:
        variacao_salarios = max(
            -0.99,
            float(
                a0
                + a1
                * ((taxa_desemprego_base / taxa_desemprego) ** a3 - 1.0)
            ),
        )
        variacao_salarios = min(1.1, variacao_salarios)
    return {
        "emprego": emprego,
        "taxa_desemprego": taxa_desemprego,
        "variacao_salarios": variacao_salarios,
    }


def atualizar_financeiro_periodo(
    capacidade: dict[str, float],
    fracao_reavaliacao_financeira: float,
    ativos_financeiros_corrigidos: pd.Series,
    passivos_financeiros_corrigidos: pd.Series,
) -> dict:
    """Transforma o B.9 final em reavaliação, fluxos e estoques financeiros."""

    capacidade_serie = pd.Series(
        capacidade,
        index=list(COLUNAS_SETORES),
        name="capacidade_financiamento",
        dtype=float,
    )
    reavaliacao_financeira = (
        -fracao_reavaliacao_financeira * capacidade_serie
    ).rename("reavaliacao_financeira")
    saldo_financeiro_incorporado = (
        capacidade_serie + reavaliacao_financeira
    ).rename("saldo_financeiro_incorporado")
    aquisicao_ativos = saldo_financeiro_incorporado.clip(lower=0.0).rename(
        "aquisicao_ativos_financeiros"
    )
    emissao_passivos = (-saldo_financeiro_incorporado).clip(lower=0.0).rename(
        "emissao_passivos_financeiros"
    )
    ativos_financeiros_periodo = (
        ativos_financeiros_corrigidos + aquisicao_ativos
    ).rename("ativos_financeiros")
    passivos_financeiros_periodo = (
        passivos_financeiros_corrigidos + emissao_passivos
    ).rename("passivos_financeiros")
    estoque_financeiro_periodo = (
        ativos_financeiros_periodo - passivos_financeiros_periodo
    ).rename("estoque_financeiro")

    #if not np.isclose(reavaliacao_financeira.sum(), 0.0, atol=1e-4):
    #    raise RuntimeError(
    #        "As reavaliações financeiras não somam zero: "
    #        f"resíduo = {reavaliacao_financeira.sum()}."
    #    )

    if not np.isclose(
        ativos_financeiros_periodo.sum(),
        passivos_financeiros_periodo.sum(),
        atol=1e-4,
    ):
        raise RuntimeError(
            "Os totais de ativos e passivos financeiros deixaram de fechar."
        )
    return {
        "capacidade_serie": capacidade_serie,
        "reavaliacao_financeira": reavaliacao_financeira,
        "aquisicao_ativos": aquisicao_ativos,
        "emissao_passivos": emissao_passivos,
        "ativos_financeiros_periodo": ativos_financeiros_periodo,
        "passivos_financeiros_periodo": passivos_financeiros_periodo,
        "estoque_financeiro_periodo": estoque_financeiro_periodo,
    }


def calcular_pib_legado(
    dados_setoriais_firmas: dict,
    pb: pd.Series,
    pc: pd.Series,
    impostos_produtos_legados: pd.Series,
) -> dict[str, float]:
    """Expõe o PIB ainda calculado com VA das firmas e impostos legados."""

    valor_adicionado_firmas = dados_setoriais_firmas["valor_adicionado"]
    va_nominal_domestico = float(valor_adicionado_firmas.loc[VA["total"]].sum())
    va_real_domestico = float(
        (valor_adicionado_firmas.loc[VA["total"]] / pb).sum()
    )
    return {
        "pib_nominal": (
            va_nominal_domestico + float(impostos_produtos_legados.sum())
        ),
        "pib_real": (
            va_real_domestico + float((impostos_produtos_legados / pc).sum())
        ),
    }


def montar_registro_historico(d: dict) -> dict:
    """Monta, sem recalcular decisões, a mesma linha pública de histórico."""

    capacidade = d["capacidade"]
    pib_nominal = d["pib_nominal"]
    return {
        "periodo": d["periodo"],
        "ano": d["ano"],
        "indice_precos": d["indice_precos"],
        "inflacao": d["inflacao"],
        "indice_salarios": d["indice_salarios"],
        "indice_cambio": d["indice_cambio"],
        "taxa_juros_nominal": d["taxa_juros_nominal"],
        #"pib_real": d["pib_real"],
        #"pib_nominal": pib_nominal,
        "emprego": d["emprego"],
        "taxa_desemprego": d["taxa_desemprego"],
        #"consumo_real": np.nan,
        #"consumo_nominal": d["consumo_cei"] / pib_nominal,
        #"poupanca_familias_nominal": d["poupanca_familias"],
        "fbcf_familias_nominal": d["fbcf_familias"],
        "fbcf_nf_real": d["investimento_nf_real_total"],
        "fbcf_nf_nominal": d["fbcf_nf_nominal"],
        "fbcf_fixa_nominal": d["fbcf_fixa_total"],
        #"variacao_estoques_real": float(d["variacao_estoques_real"].sum()),
        #"variacao_autonoma_estoques_real": float(
        #   d["variacao_autonoma_estoques_periodo"].sum()
        #),
        #"variacao_ciclica_estoques_real": float(
        #    d["variacao_ciclica_estoques"].sum()
        #),
        #"estoque_real": float(d["estoque_real_periodo"].sum()),
        "investimento_liquido_nf_real": float(
            d["investimento_liquido_nf"].sum()
        ),
        "investimento_reposicao_nf_real": float(
            d["investimento_reposicao_nf"].sum()
        ),
        "ajuste_piso_investimento_nf_real": float(
            (d["investimento_nf_por_investidor"] - d["investimento_nf_sem_piso"]).sum()
        ),
        "estoque_capital_nf_real": float(d["estoque_capital_nf_periodo"].sum()),
        "setores_no_piso_investimento_nf": int(
            (d["investimento_nf_sem_piso"] < 0.0).sum()
        ),
        #"residuo_consumo": d["residuo_consumo"],
        #"iteracoes_consumo": d["iteracao"],
        "deficit_governo": -capacidade["governo"] / pib_nominal,
        "saldo_setor_externo": capacidade["setor_externo"] / pib_nominal,
        "discrepancia_cei": sum(capacidade.values()),
    }


def atualizar_estado_periodo(d: dict) -> dict:
    """Constrói os estados herdados por t+1 sem alterar o timing legado."""

    dados_setoriais = d["dados_setoriais_firmas"]
    return {
        "indice_salarios": d["indice_salarios"] * (1.0 + d["variacao_salarios"]),
        "indice_cambio": d["indice_cambio"]
        * (1.0 + d["repasse_inflacao_cambio"] * d["inflacao"]),
        "indice_precos_anterior": d["indice_precos"],
        "pc_anterior_2": d["pc_anterior"].copy(),
        "pc_anterior": d["pc"].copy(),
        "governo_nominal_anterior": d["governo_nominal"].copy(),
        "fbcf_fixa_nominal_anterior": d["fbcf_fixa_nominal"].copy(),
        "exportacoes_nominais_anterior": d["exportacoes_nominais"].copy(),
        #"poupanca_familias_anterior": d["poupanca_familias"],
        "estoque_capital_nf": d["estoque_capital_nf_periodo"].copy(),
        "ativos_financeiros": d["ativos_financeiros_periodo"].copy(),
        "passivos_financeiros": d["passivos_financeiros_periodo"].copy(),
        "estoque_financeiro": d["estoque_financeiro_periodo"].copy(),
        #"estoque_real": d["estoque_real_periodo"].copy(),
        #"estoque_referencia": d["estoque_referencia_periodo"].copy(),
        #"estoque_ciclico": d["estoque_ciclico_periodo"].copy(),
        "producao_nf_anterior": d["producao_nf_corrente"].copy(),
        "producao_nf_corrente": dados_setoriais["producao_real"]
        .loc[d["setores_nf"]]
        .copy(),
        "producao_estoques_anterior": d["producao_estoques_corrente"].copy(),
        "producao_estoques_corrente": dados_setoriais["producao_real"].copy(),
    }
