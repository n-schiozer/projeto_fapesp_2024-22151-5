"""Leitura e calibração conjunta da TRU e da CEI no ano-base."""

from pathlib import Path

import numpy as np
import pandas as pd

from contabilidade.cei_abm.tru_sector_sector import (
    load_tru_data,
    transform_tru_to_sector_sector,
)
from calibracao.calibracao_investimento_nf import calibrar_investimento_nf
from contabilidade.estrutura_cei import (
    C,
    L,
    VA,
    LINHAS_BASE_IR_FIRMAS,
    LINHAS_OBRIGATORIAS,
)


def preparar_condicoes_iniciais(
    config: dict,
    data_dir: Path,
    arquivo_cei: Path,
) -> dict:
    """Lê as bases e devolve somente os dados usados por ``simul_``."""

    # ------------------------------------------------------------------
    # TRU: demanda, oferta e matrizes reais
    # ------------------------------------------------------------------
    dados = load_tru_data(
        data_dir=data_dir,
        year=config["ano"],
        level=config["nivel"],
    )

    tru = transform_tru_to_sector_sector(dados, validate=True)
    
    # Os nomes dos setores passam a vir da própria TRU transformada. Todos os
    # vetores econômicos abaixo são Series e todas as matrizes são DataFrames;
    # assim, os rótulos sobrevivem a somas e multiplicações matriciais.

    setores = list(tru.household_consumption_sector.index)
    n = len(setores)
    I = pd.DataFrame(np.eye(n), index=setores, columns=setores)

    consumo = (
        tru.household_consumption_sector.iloc[:, 0]
        + tru.npo_consumption_sector.iloc[:, 0]
    ).rename("consumo")

    governo = tru.gov_cons_sector.iloc[:, 0].rename("governo")

    investimento = (
        tru.gross_investment_sector.iloc[:, 0]
        + tru.stocks_investment_sector.iloc[:, 0]
    ).rename("investimento")
    exportacoes = tru.exports_sector.iloc[:, 0].rename("exportacoes")

    demanda_final_componentes = pd.concat(
        [consumo, governo, investimento, exportacoes],
        axis="columns",
    )
    df = demanda_final_componentes.sum(axis="columns").rename("demanda_final")

    demanda_final = (tru.leontief_inverse @ df).rename(
        "demanda_final"
    )

    consumo_intermediario = tru.intermediate_consumption_sector.copy()

    importacoes = tru.imports_sector.iloc[:, 0].rename("importacoes")

    impostos = tru.taxes_sector.iloc[:, 0].rename("impostos")

    taxa_impostos = impostos.div(
        demanda_final.replace(0.0, np.nan)
    ).fillna(0.0).rename("taxa_impostos")

    T = pd.DataFrame(
        np.diag(taxa_impostos),
        index=setores,
        columns=setores,
    )

    # Nas tabelas de oferta, margem positiva identifica o produto sobre o qual
    # comércio/transporte é cobrado; margem negativa identifica o fornecedor do
    # serviço. M tem sinal positivo no produto e negativo no fornecedor, para
    # que C = I - T - M reduza o produto e crie demanda por esses serviços.
    # Por construção, cada coluna de M soma zero.

    M = pd.DataFrame(0.0, index=setores, columns=setores)

    gasto_comercio = tru.trade_margin_sector.iloc[:, 0].rename("gasto_comercio")

    gasto_transporte = tru.transport_margin_sector.iloc[:, 0].rename("gasto_transporte")

    margem_comercio = gasto_comercio.div(
        demanda_final.replace(0.0, np.nan)
    ).fillna(0.0)

    margem_transporte = gasto_transporte.div(
        demanda_final.replace(0.0, np.nan)
    ).fillna(0.0)
    

    for margem in (tru.trade_margin_sector, tru.transport_margin_sector):
        margem = margem.iloc[:, 0]
        cobrancas = margem.clip(lower=0.0)
        recebedores = (-margem).clip(lower=0.0)
        if not np.isclose(cobrancas.sum(), recebedores.sum(), atol=1e-6):
            raise ValueError("As margens não formam uma transferência de soma zero.")
        taxa = cobrancas.div(
            demanda_final.replace(0.0, np.nan)
        ).fillna(0.0)
        transferencia = pd.DataFrame(
            np.outer(recebedores / recebedores.sum(), taxa),
            index=setores,
            columns=setores,
        )
        M += pd.DataFrame(
            np.diag(taxa),
            index=setores,
            columns=setores,
        ) - transferencia

    margens_transporte = M

    conversao_de_pm_pb = I - T - M

    oferta_basica = conversao_de_pm_pb @ demanda_final

    parcela_importada = importacoes.div(
        oferta_basica.replace(0.0, np.nan)
    ).fillna(0.0).rename("parcela_importada")

    Sd = pd.DataFrame(
        np.diag(1.0 - parcela_importada),
        index=setores,
        columns=setores,
    )
    
    Sm = pd.DataFrame(
        np.diag(parcela_importada),
        index=setores,
        columns=setores,
    )
    conversao_domestica = Sd @ conversao_de_pm_pb
    producao_domestica = (conversao_domestica @ demanda_final).rename(
        "producao_domestica"
    )

    ci_domestico = conversao_domestica @ consumo_intermediario
    denominador_producao = producao_domestica.replace(0.0, np.nan)
    A_real = ci_domestico.div(
        denominador_producao,
        axis="columns",
    ).fillna(0.0)
    A_precos = consumo_intermediario.div(
        denominador_producao,
        axis="columns",
    ).fillna(0.0)
    leontief_domestica = pd.DataFrame(
        np.linalg.inv((I - A_real).to_numpy()),
        index=setores,
        columns=setores,
    )

    tabela_va = pd.DataFrame(
        dados.value_added_components,
        index=dados.va_components_names,
        columns=setores,
        dtype=float,
    )


    eob_misto_ff = tabela_va.loc[
        "Excedente operacional bruto e rendimento misto bruto"
    ][
        "K - Atividades financeiras, de seguros e serviços relacionados"
    ]

    eob_misto_nf = (
        tabela_va.loc[
            "Excedente operacional bruto e rendimento misto bruto"
        ].sum()
        - eob_misto_ff
    )

    razoes_va = tabela_va.div(
        tabela_va.loc["Valor da produção"].replace(0.0, np.nan),
        axis="columns",
    ).fillna(0.0)
    va_base = razoes_va.mul(
        producao_domestica,
        axis="columns",
    )
    participacoes_va = va_base.div(
        va_base.loc[VA["total"]].replace(0.0, np.nan),
        axis="columns",
    ).fillna(0.0)
    va_total = tru.value_added_sector.iloc[:, 0].rename("valor_adicionado")
    v0 = va_total.div(
        producao_domestica.replace(0.0, np.nan)
    ).fillna(0.0).rename("coeficiente_va")

    # Sistema de preços da Planilha2.
    G = pd.DataFrame(
        np.linalg.solve((I - T).to_numpy(), conversao_de_pm_pb.T.to_numpy()),
        index=setores,
        columns=setores,
    )
    pm0 = pd.Series(1.0, index=setores, name="preco_importacoes")
    pc0 = pd.Series(
        np.linalg.solve(
            (I - G @ Sd @ A_precos.T).to_numpy(),
            (G @ (Sd @ v0 + Sm @ pm0)).to_numpy(),
        ),
        index=setores,
        name="preco_comprador",
    )
    pb0 = (A_precos.T @ pc0 + v0).rename("preco_basico")
    if not np.allclose(pc0, 1.0, atol=1e-10) or not np.allclose(
        pb0, 1.0, atol=1e-10
    ):
        raise RuntimeError("O sistema de preços não reproduziu Pc0 = Pb0 = 1.")

    # ------------------------------------------------------------------
    # CEI: leitura e parâmetros comportamentais do ano-base
    # ------------------------------------------------------------------
    planilha = pd.read_excel(arquivo_cei, sheet_name=config["aba_cei"])
    linhas_validas = planilha.iloc[:, 1:].fillna(0).ne(0).any(axis=1)
    cei = (
        planilha.loc[linhas_validas, planilha.columns[:11]]
        .copy()
        .reset_index(drop=True)
    )
    if cei.shape[0] < 17 or cei.shape[1] < 11:
        raise ValueError("A CEI não possui as 17 linhas e 11 colunas esperadas.")
    # ``valores`` conserva as mesmas posições de ``cei``. A coluna zero
    # continua contendo os nomes das contas e apenas as dez colunas de fluxos
    # são convertidas para número. Portanto, ``C[...]`` nunca precisa de ``- 1``.
    valores = cei.copy(deep=True)
    valores.iloc[:, 1:11] = (
        valores.iloc[:, 1:11]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    fbcf_fixa_cei_base = {
        "governo": float(valores.iat[L["fbcf"], C["governo_s"]]),
        "firmas_financeiras": float(valores.iat[L["fbcf"], C["ff_s"]]),
        "setor_externo": float(valores.iat[L["fbcf"], C["externo_s"]]),
    }
    outras_transferencias_base = {
        "familias_recebidas": float(
            valores.iat[L["outras_transferencias"], C["familias_e"]]
        ),
        "governo_recebidas": float(
            valores.iat[L["outras_transferencias"], C["governo_e"]]
        ),
        "ff_pagas": float(
            valores.iat[L["outras_transferencias"], C["ff_s"]]
        ),
        "nf_pagas": float(
            valores.iat[L["outras_transferencias"], C["nf_s"]]
        ),
        "exterior_pagas": float(
            valores.iat[L["outras_transferencias"], C["externo_s"]]
        ),
    }

    renda_ff = float(valores.iat[L["dividendos"], C["ff_s"]])
    renda_nf = float(valores.iat[L["dividendos"], C["nf_s"]])
    renda_total = renda_ff + renda_nf
    entradas_ff = float(valores.iloc[1:7, C["ff_e"]].sum())
    entradas_nf = float(valores.iloc[1:7, C["nf_e"]].sum())
    saidas_ff = float(valores.iloc[LINHAS_OBRIGATORIAS, C["ff_s"]].sum())
    saidas_nf = float(valores.iloc[LINHAS_OBRIGATORIAS, C["nf_s"]].sum())

    base_ir_ff = float(
        valores.iloc[LINHAS_BASE_IR_FIRMAS, C["ff_e"]].sum()
        - valores.iloc[LINHAS_BASE_IR_FIRMAS, C["ff_s"]].sum()
    )
    base_ir_nf = float(
        valores.iloc[LINHAS_BASE_IR_FIRMAS, C["nf_e"]].sum()
        - valores.iloc[LINHAS_BASE_IR_FIRMAS, C["nf_s"]].sum()
    )
    base_ir_familias = float(valores.iloc[1:7, C["familias_e"]].sum())
    renda_disponivel = float(
        valores.iloc[1:13, C["familias_e"]].sum()
        - valores.iloc[1:9, C["familias_s"]].sum()
    )

    poupanca_familias = float(
        valores.iloc[1:13, C["familias_e"]].sum()
        - valores.iloc[1:9, C["familias_s"]].sum()
        - valores.iloc[L["consumo"], C["familias_s"]]
    )

    prop_invest_prev_familias = (
        valores.iloc[L["contribuicoes_sociais"], C["familias_s"]]
        / poupanca_familias
    )
    prop_invest_fbcf_familias = (
        valores.iloc[L["fbcf"], C["familias_s"]] / poupanca_familias
    )
    contribuicoes_familias = valores.iloc[
        L["contribuicoes_sociais"], C["familias_s"]
    ]
    prop_prev_publica = (
        valores.iloc[L["contribuicoes_sociais"], C["governo_e"]]
        / contribuicoes_familias
    )
    prop_prev_privada = (
        valores.iloc[L["contribuicoes_sociais"], C["ff_e"]]
        / contribuicoes_familias
    )

    emprego = float(va_base.loc["Fator trabalho (ocupações)"].sum())
    pea = emprego / (1.0 - config["taxa_desemprego_inicial"])
    populacao = pea / config["parcela_ativa_populacao"]
    aposentados = config["parcela_aposentados_inativos"] * (populacao - pea)

    salarios = va_base.loc[VA["salarios"]].astype(float)
    ocupacoes = va_base.loc["Fator trabalho (ocupações)"].astype(float)
    validos = (salarios > 0.0) & (ocupacoes > 0.0)
    beneficio_individual = 0.80 * float(
        (salarios[validos] * 1_000_000.0 / ocupacoes[validos]).min()
    )
    beneficios_base = float(valores.iat[L["beneficios"], C["familias_e"]])
    beneficio_fixo = (
        beneficios_base
        - beneficio_individual * (pea - emprego) / 1_000_000.0
    )
    aposentadoria_total = float(
        valores.iat[L["aposentadorias"], C["familias_e"]]
    )

    parametros = {
        "parcela_dividendos_familias": float(
            valores.iat[L["dividendos"], C["familias_e"]] / renda_total
        ),
        "parcela_dividendos_exterior": float(
            valores.iat[L["dividendos"], C["externo_e"]] / renda_total
        ),
        "razao_dividendos_ff": renda_ff / (entradas_ff - saidas_ff),
        "razao_dividendos_nf": renda_nf / (entradas_nf - saidas_nf),
        "razao_divendos_eob_ff": renda_ff / eob_misto_ff,
        "razao_divendos_eob_nf": renda_nf / eob_misto_nf,
        "taxa_ir_familias": float(
            valores.iat[L["ir"], C["familias_s"]] / base_ir_familias
        ),
        "taxa_ir_ff": float(valores.iat[L["ir"], C["ff_s"]] / base_ir_ff),
        "taxa_ir_nf": float(valores.iat[L["ir"], C["nf_s"]] / base_ir_nf),
        "propensao_consumir": float(
            valores.iat[L["consumo"], C["familias_s"]] / renda_disponivel
        ),
        "prop_invest_prev_familias": prop_invest_prev_familias,
        "prop_invest_fbcf_familias": prop_invest_fbcf_familias,
        "prop_prev_privada": prop_prev_privada,
        "prop_prev_publica": prop_prev_publica,
        "pea": pea,
        "aposentados": aposentados,
        "beneficio_fixo": beneficio_fixo,
        "beneficio_por_desempregado": beneficio_individual,
        "aposentadoria_por_pessoa": aposentadoria_total / aposentados,
        "parcela_governo_aposentadoria": float(
            valores.iat[L["aposentadorias"], C["governo_s"]]
            / aposentadoria_total
        ),
        "parcela_ff_aposentadoria": float(
            valores.iat[L["aposentadorias"], C["ff_s"]]
            / aposentadoria_total
        ),
    }

    # ------------------------------------------------------------------
    # Investimento das firmas não financeiras
    # ------------------------------------------------------------------
    # beta é estimado com a regressão, sem intercepto:
    # ΔY_t = beta * ΔY_(t-1).
    # Em seguida, v é recalibrado para que a equação do estoque de capital
    # reproduza exatamente a FBCF das firmas NF observada no ano-base.
    # A exclusão abaixo afeta apenas quem decide investimento pelo acelerador.
    # Os mesmos setores continuam permitidos como fornecedores de bens de
    # capital na TRU. O valor padrão preserva arquivos CONFIG mais antigos.
    setores_excluidos_investimento_nf = config.get(
        "setores_excluidos_investimento_nf",
        [setores[config["setor_financeiro"]]],
    )
    investimento_nf = calibrar_investimento_nf(
        data_dir,
        arquivo_cei,
        ano_base=config["ano"],
        nivel=config["nivel"],
        aba_cei=config["aba_cei"],
        vida_util_capital=config["vida_util_capital"],
        ano_inicial_beta=config["ano_inicial_beta"],
        ano_final_beta=config["ano_final_beta"],
        setores_excluidos=setores_excluidos_investimento_nf,
    )
    if investimento_nf["setores_nf"] != [
        setor
        for setor in setores
        if setor not in setores_excluidos_investimento_nf
    ]:
        raise RuntimeError("Os setores da calibração do investimento não coincidem.")

    return {
        "config": config,
        "tru_base": tru,
        "setores": setores,
        "demanda_final_componentes_base": demanda_final_componentes,
        "consumo_base": consumo,
        "governo_base": governo,
        "investimento_base": investimento,
        "exportacoes_base": exportacoes,
        "demanda_final_base": demanda_final,
        "conversao_de_pm_pb": conversao_de_pm_pb,
        "conversao_domestica": conversao_domestica,
        "leontief_domestica": leontief_domestica,
        "A_precos": A_precos,
        "Sd": Sd,
        "Sm": Sm,
        "G": G,
        "v0": v0,
        "taxa_impostos": taxa_impostos,
        "margens_transporte": margens_transporte,
        "parcela_importada": parcela_importada,
        "razoes_va": razoes_va,
        "participacoes_va": participacoes_va,
        "va_base": va_base,
        "cei_original": cei,
        "valores_cei": valores,
        "fbcf_fixa_cei_base": fbcf_fixa_cei_base,
        "outras_transferencias_base": outras_transferencias_base,
        "parametros_cei": parametros,
        "investimento_nf": investimento_nf,
        "margem_comercio": margem_comercio,
        "margem_transporte": margem_transporte,
        "consumo_intermediario":consumo_intermediario
    }
