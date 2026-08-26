"""Três blocos auditáveis: TRU, CEI e ciclo temporal."""

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import (
    C,
    L,
    VA,
    COLUNAS_SETORES,
    LINHAS_BASE_IR_FIRMAS,
    LINHAS_OBRIGATORIAS,
)


def simul_TRU(
    demandas_nominais: dict,
    precos: dict,
    parametros: dict,
) -> dict:
    """Converte a demanda nominal e calcula a TRU real e nominal."""

    pc = precos["pc"]
    pb = precos["pb"]
    pm = precos["pm"]
    indice_salarios = precos["indice_salarios"]

    # Demanda final real: cada vetor nominal é dividido por Pc.
    C_real = demandas_nominais["consumo"] / pc
    G_real = demandas_nominais["governo"] / pc
    I_real = demandas_nominais["investimento"] / pc
    X_real = demandas_nominais["exportacoes"] / pc
    DF_real = C_real + G_real + I_real + X_real

    # Demanda final -> produção total -> importações, CI, VA e emprego.
    producao_domestica = (
        parametros["leontief_domestica"]
        @ parametros["conversao_domestica"]
        @ DF_real
    ).rename("producao_domestica")
    producao_comprador = pd.Series(
        np.linalg.solve(
            parametros["conversao_domestica"].to_numpy(),
            producao_domestica.to_numpy(),
        ),
        index=parametros["setores"],
        name="producao_comprador",
    )
    oferta_basica = (
        parametros["conversao_de_pm_pb"] @ producao_comprador
    ).rename("oferta_basica")
    importacoes = (
        parametros["parcela_importada"] * oferta_basica
    ).rename("importacoes")
    impostos = (
        parametros["taxa_impostos"] * producao_comprador
    ).rename("impostos_produtos")
    consumo_intermediario = parametros["A_precos"].mul(
        producao_domestica,
        axis="columns",
    )
    valor_adicionado = parametros["razoes_va"].mul(
        producao_domestica,
        axis="columns",
    )

    # A TRU nominal é a TRU real valorizada por Pc, Pb e Pm.
    ci_nominal = consumo_intermediario.mul(pc, axis="index")
    producao_domestica_nominal = producao_domestica * pb

    # VA nominal total = produção doméstica nominal - CI nominal.
    va_total_nominal = (
        producao_domestica_nominal
        - ci_nominal.sum(axis="index")
    )

    # Distribuição do VA nominal:
    # - salários seguem a regra comportamental;
    # - componentes elementares restantes mantêm sua razão com o VA;
    # - subtotais são reconstruídos;
    # - EOB é o resíduo que fecha o VA total.
    va_nominal = pd.DataFrame(
        0.0,
        index=valor_adicionado.index,
        columns=valor_adicionado.columns,
    )
    va_nominal.loc[VA["total"]] = va_total_nominal
    va_nominal.loc[VA["salarios"]] = (
        valor_adicionado.loc[VA["salarios"]] * indice_salarios
    )
    componentes_proporcionais_ao_va = (
        "previdencia_oficial",
        "previdencia_privada",
        "contribuicoes_imputadas",
        "rendimento_misto",
        "outros_impostos",
        "outros_subsidios",
    )
    for componente in componentes_proporcionais_ao_va:
        nome_linha = VA[componente]
        va_nominal.loc[nome_linha] = (
            parametros["participacoes_va"].loc[nome_linha]
            * va_total_nominal
        )
    va_nominal.loc[VA["contribuicoes_efetivas"]] = (
        va_nominal.loc[VA["previdencia_oficial"]]
        + va_nominal.loc[VA["previdencia_privada"]]
    )
    va_nominal.loc[VA["remuneracoes"]] = (
        va_nominal.loc[VA["salarios"]]
        + va_nominal.loc[VA["contribuicoes_efetivas"]]
        + va_nominal.loc[VA["contribuicoes_imputadas"]]
    )
    va_nominal.loc[VA["eob"]] = (
        va_nominal.loc[VA["total"]]
        - va_nominal.loc[VA["remuneracoes"]]
        - va_nominal.loc[VA["rendimento_misto"]]
        - va_nominal.loc[VA["outros_impostos"]]
        - va_nominal.loc[VA["outros_subsidios"]]
    )
    va_nominal.loc[VA["eob_mais_misto"]] = (
        va_nominal.loc[VA["rendimento_misto"]]
        + va_nominal.loc[VA["eob"]]
    )
    va_nominal.loc[VA["producao"]] = producao_domestica_nominal
    va_nominal.loc[VA["ocupacoes"]] = valor_adicionado.loc[VA["ocupacoes"]]
    real = {
        "consumo_familias": C_real,
        "consumo_governo": G_real,
        "investimento": I_real,
        "exportacoes": X_real,
        "demanda_final": DF_real,
        "consumo_intermediario": consumo_intermediario,
        "producao_comprador": producao_comprador,
        "oferta_basica": oferta_basica,
        "producao_domestica": producao_domestica,
        "importacoes": importacoes,
        "impostos_produtos": impostos,
        "valor_adicionado": valor_adicionado,
    }
    nominal = {
        "consumo_familias": C_real * pc,
        "consumo_governo": G_real * pc,
        "investimento": I_real * pc,
        "exportacoes": X_real * pc,
        "demanda_final": DF_real * pc,
        "consumo_intermediario": ci_nominal,
        "producao_comprador": producao_comprador * pc,
        "oferta_basica": producao_domestica * pb + importacoes * pm,
        "producao_domestica": producao_domestica_nominal,
        "importacoes": importacoes * pm,
        "impostos_produtos": impostos * pc,
        "valor_adicionado": va_nominal,
    }
    return {"real": real, "nominal": nominal}


def simul_CEI(
    parametros: dict,
    results_TRU: dict,
    CEI_inicial: pd.DataFrame,
    inflation_index: float,
) -> dict:
    """Atualiza explicitamente as linhas da CEI e calcula o consumo nominal."""

    p = parametros["calibrados"]
    valores_base = parametros["valores_base"]
    indice_salarios = parametros["indice_salarios"]
    tru_real = results_TRU["real"]
    tru_nominal = results_TRU["nominal"]

    # Primeiro, todos os valores sem função própria conservam seu valor real.
    cei = CEI_inicial.copy(deep=True)
    cei.iloc[1:16, 1:11] = (
        valores_base.iloc[1:16, 1:11].fillna(0.0).to_numpy(dtype=float)
        * inflation_index
    )

    # Juros recebidos e pagos permanecem separados. Isso é necessário porque
    # alguns cálculos da CEI, como o IR das famílias, usam a entrada bruta de
    # juros como base tributária. Trabalhar apenas com o líquido alteraria essa
    # base mesmo quando o saldo líquido fosse exatamente igual ao ano-base.
    juros_recebidos = parametros["juros_recebidos"]
    juros_pagos = parametros["juros_pagos"]
    soma_juros_liquidos = float((juros_recebidos - juros_pagos).sum())
    if abs(soma_juros_liquidos) > 1e-4:
        raise RuntimeError(
            "Os juros recebidos e pagos não fecham: "
            f"resíduo = {soma_juros_liquidos}."
        )
    for nome, (entrada, saida) in COLUNAS_SETORES.items():
        cei.iloc[L["juros"], entrada] = float(juros_recebidos.loc[nome])
        cei.iloc[L["juros"], saida] = float(juros_pagos.loc[nome])

    # Linha 1: VA das firmas e transações com o exterior vêm da TRU nominal.
    sf = parametros["setor_financeiro"]
    va = tru_nominal["valor_adicionado"]
    setor_financeiro = va.columns[sf]
    impostos = tru_nominal["impostos_produtos"]
    impostos_ff = float(impostos.iloc[sf])
    impostos_nf = float(impostos.sum() - impostos_ff)
    cei.iloc[L["va"], C["ff_e"]] = (
        float(va.loc[VA["total"], setor_financeiro]) + impostos_ff
    )
    cei.iloc[L["va"], C["nf_e"]] = (
        float(
            va.loc[VA["total"]].sum()
            - va.loc[VA["total"], setor_financeiro]
        )
        + impostos_nf
    )
    cei.iloc[L["va"], C["externo_e"]] = float(
        tru_nominal["importacoes"].sum()
    )
    cei.iloc[L["va"], C["externo_s"]] = float(
        tru_nominal["exportacoes"].sum()
    )

    # Impostos sobre produtos.
    cei.iloc[L["impostos_produtos"], C["ff_s"]] = impostos_ff
    cei.iloc[L["impostos_produtos"], C["nf_s"]] = impostos_nf
    cei.iloc[L["impostos_produtos"], C["governo_e"]] = impostos_ff + impostos_nf

    # Salários e contribuições efetivas.
    salarios_ff = float(va.loc[VA["salarios"], setor_financeiro])
    salarios_nf = float(va.loc[VA["salarios"]].sum() - salarios_ff)
    cei.iloc[L["salarios"], C["ff_s"]] = salarios_ff
    cei.iloc[L["salarios"], C["nf_s"]] = salarios_nf
    cei.iloc[L["salarios"], C["familias_e"]] = salarios_ff + salarios_nf

    contribuicoes_ff = float(
        va.loc[VA["contribuicoes_efetivas"], setor_financeiro]
    )
    contribuicoes_nf = float(
        va.loc[VA["contribuicoes_efetivas"]].sum() - contribuicoes_ff
    )
    cei.iloc[L["contribuicoes_efetivas"], C["ff_s"]] = contribuicoes_ff
    cei.iloc[L["contribuicoes_efetivas"], C["nf_s"]] = contribuicoes_nf
    cei.iloc[L["contribuicoes_efetivas"], C["familias_e"]] = (
        contribuicoes_ff + contribuicoes_nf
    )

    # Outros impostos sobre a produção.
    outros_impostos_ff = float(
        va.loc[VA["outros_impostos"], setor_financeiro]
        + va.loc[VA["outros_subsidios"], setor_financeiro]
    )
    outros_impostos_nf = float(
        va.loc[VA["outros_impostos"]].sum()
        + va.loc[VA["outros_subsidios"]].sum()
        - outros_impostos_ff
    )
    cei.iloc[L["outros_impostos"], C["ff_s"]] = outros_impostos_ff
    cei.iloc[L["outros_impostos"], C["nf_s"]] = outros_impostos_nf
    cei.iloc[L["outros_impostos"], C["governo_e"]] = (
        outros_impostos_ff + outros_impostos_nf
    )

    # Imposto de renda de FF e NF: mesma base, excluindo IR e dividendos.
    base_ff = float(
        np.asarray(cei.iloc[LINHAS_BASE_IR_FIRMAS, C["ff_e"]], dtype=float).sum()
        - np.asarray(cei.iloc[LINHAS_BASE_IR_FIRMAS, C["ff_s"]], dtype=float).sum()
    )
    base_nf = float(
        np.asarray(cei.iloc[LINHAS_BASE_IR_FIRMAS, C["nf_e"]], dtype=float).sum()
        - np.asarray(cei.iloc[LINHAS_BASE_IR_FIRMAS, C["nf_s"]], dtype=float).sum()
    )
    cei.iloc[L["ir"], C["ff_s"]] = p["taxa_ir_ff"] * base_ff
    cei.iloc[L["ir"], C["nf_s"]] = p["taxa_ir_nf"] * base_nf

    # Dividendos das firmas e seus destinatários.
    entradas_ff = float(np.asarray(cei.iloc[1:7, C["ff_e"]], dtype=float).sum())
    entradas_nf = float(np.asarray(cei.iloc[1:7, C["nf_e"]], dtype=float).sum())
    saidas_ff = float(
        np.asarray(cei.iloc[LINHAS_OBRIGATORIAS, C["ff_s"]], dtype=float).sum()
    )
    saidas_nf = float(
        np.asarray(cei.iloc[LINHAS_OBRIGATORIAS, C["nf_s"]], dtype=float).sum()
    )
    dividendos_ff = p["razao_dividendos_ff"] * (entradas_ff - saidas_ff)
    dividendos_nf = p["razao_dividendos_nf"] * (entradas_nf - saidas_nf)
    dividendos = dividendos_ff + dividendos_nf
    cei.iloc[L["dividendos"], C["ff_s"]] = dividendos_ff
    cei.iloc[L["dividendos"], C["nf_s"]] = dividendos_nf
    cei.iloc[L["dividendos"], C["familias_e"]] = (
        p["parcela_dividendos_familias"] * dividendos
    )
    cei.iloc[L["dividendos"], C["externo_e"]] = (
        p["parcela_dividendos_exterior"] * dividendos
    )

    # IR das famílias e arrecadação do governo.
    ir_familias = p["taxa_ir_familias"] * float(
        np.asarray(cei.iloc[1:7, C["familias_e"]], dtype=float).sum()
    )
    cei.iloc[L["ir"], C["familias_s"]] = ir_familias
    cei.iloc[L["ir"], C["governo_e"]] = (
        ir_familias
        + float(cei.iloc[L["ir"], C["ff_s"]])
        + float(cei.iloc[L["ir"], C["nf_s"]])
    )

    # Benefícios e aposentadorias.
    emprego = float(
        tru_real["valor_adicionado"].loc["Fator trabalho (ocupações)"].sum()
    )
    desempregados = p["pea"] - emprego
    beneficios = (
        p["beneficio_fixo"] * inflation_index
        + p["beneficio_por_desempregado"]
        * indice_salarios
        * desempregados
        / 1_000_000.0
    )
    cei.iloc[L["beneficios"], C["familias_e"]] = beneficios
    cei.iloc[L["beneficios"], C["governo_s"]] = beneficios

    aposentadorias = (
        p["aposentadoria_por_pessoa"] * inflation_index * p["aposentados"]
    )
    cei.iloc[L["aposentadorias"], C["familias_e"]] = aposentadorias
    cei.iloc[L["aposentadorias"], C["governo_s"]] = (
        p["parcela_governo_aposentadoria"] * aposentadorias
    )
    cei.iloc[L["aposentadorias"], C["ff_s"]] = (
        p["parcela_ff_aposentadoria"] * aposentadorias
    )

    # Consumo nominal induzido pela renda disponível.
    renda_disponivel = float(
        np.asarray(cei.iloc[1:13, C["familias_e"]], dtype=float).sum()
        - np.asarray(cei.iloc[1:9, C["familias_s"]], dtype=float).sum()
    )
    consumo_cei = p["propensao_consumir"] * renda_disponivel
    cei.iloc[L["consumo"], C["familias_s"]] = consumo_cei
    cei.iloc[L["consumo"], C["governo_s"]] = float(
        tru_nominal["consumo_governo"].sum()
    )

    # Alocação da poupança das famílias:

    poupanca_familias = float(
        cei.iloc[1:13, C["familias_e"]].sum()
        - cei.iloc[1:9, C["familias_s"]].sum()
        - cei.iloc[L["consumo"], C["familias_s"]]
    )

    previdencia_familias = (
        p["prop_invest_prev_familias"] * poupanca_familias
    )
    cei.iloc[L["contribuicoes_sociais"], C["familias_s"]] = (
        previdencia_familias
    )
    cei.iloc[L["contribuicoes_sociais"], C["governo_e"]] = (
        p["prop_prev_publica"] * previdencia_familias
    )
    cei.iloc[L["contribuicoes_sociais"], C["ff_e"]] = (
        p["prop_prev_privada"] * previdencia_familias
    )
    # A FBCF das famílias foi decidida no início do período com base na
    # poupança observada no período anterior. Portanto, ela permanece fixa
    # durante as substituições do consumo corrente.
    fbcf_familias = parametros["fbcf_familias"]
    cei.iloc[L["fbcf"], C["familias_s"]] = fbcf_familias
    cei.iloc[L["estoques"], C["familias_s"]] = 0.0

    # A FBCF das firmas não financeiras vem da equação do estoque de capital
    # calculada no início do período. Como a CEI é nominal, recebe aqui a soma
    # já valorizada pelos preços dos bens de capital.
    fbcf_nf = parametros["fbcf_nf"]
    cei.iloc[L["fbcf"], C["nf_s"]] = fbcf_nf

    # Os demais componentes não são um resíduo do investimento da TRU.
    # A FBCF fixa e a variação endógena de estoques possuem trajetórias
    # próprias, calculadas no FOR temporal e recebidas explicitamente aqui.
    # Assim, a participação institucional no investimento total pode mudar
    # quando a FBCF endógena das famílias ou das firmas NF mudar.
    investimentos_fixos = parametros["investimentos_fixos"]
    cei.iloc[L["fbcf"], C["governo_s"]] = investimentos_fixos[
        "fbcf_governo"
    ]
    cei.iloc[L["fbcf"], C["ff_s"]] = investimentos_fixos[
        "fbcf_firmas_financeiras"
    ]
    cei.iloc[L["fbcf"], C["externo_s"]] = investimentos_fixos[
        "fbcf_setor_externo"
    ]
    cei.iloc[L["estoques"], C["governo_s"]] = investimentos_fixos[
        "estoques_governo"
    ]
    cei.iloc[L["estoques"], C["ff_s"]] = investimentos_fixos[
        "estoques_firmas_financeiras"
    ]
    cei.iloc[L["estoques"], C["nf_s"]] = investimentos_fixos[
        "estoques_firmas_nao_financeiras"
    ]
    cei.iloc[L["estoques"], C["externo_s"]] = investimentos_fixos[
        "estoques_setor_externo"
    ]

    # Fechamentos obrigatórios entre a TRU nominal e a CEI. A primeira
    # igualdade isola as firmas NF; a segunda verifica todo o investimento,
    # incluindo famílias, demais instituições e variação de estoques.
    fbcf_nf_cei = float(cei.iloc[L["fbcf"], C["nf_s"]])
    if not np.isclose(fbcf_nf_cei, fbcf_nf, atol=1e-6):
        raise RuntimeError("A FBCF NF da TRU não coincide com a CEI.")

    colunas_saidas_institucionais = [
        C["familias_s"],
        C["governo_s"],
        C["ff_s"],
        C["nf_s"],
        C["externo_s"],
    ]
    investimento_total_cei = float(
        np.asarray(
            cei.iloc[
                [L["fbcf"], L["estoques"]],
                colunas_saidas_institucionais,
            ],
            dtype=float,
        ).sum()
    )
    investimento_total_tru = float(tru_nominal["investimento"].sum())
    if not np.isclose(
        investimento_total_cei,
        investimento_total_tru,
        atol=1e-6,
    ):
        raise RuntimeError("O investimento total da TRU não coincide com a CEI.")

    capacidade = {}
    for nome, (entrada, saida) in COLUNAS_SETORES.items():
        saldo = float(
            np.asarray(cei.iloc[1:16, entrada], dtype=float).sum()
            - np.asarray(cei.iloc[1:16, saida], dtype=float).sum()
        )
        capacidade[nome] = saldo
        cei.iloc[L["capacidade"], entrada] = saldo

    # Para as famílias, a capacidade é o ativo financeiro adquirido como
    # resíduo: poupança - previdência - FBCF. Ela não possui parâmetro próprio.

    return {
        "cei": cei,
        "consumo_nominal": float(consumo_cei),
        "poupanca_familias": float(poupanca_familias),
        "fbcf_familias": float(fbcf_familias),
        "fbcf_nf": float(fbcf_nf),
        "capacidade_financiamento": capacidade,
        "emprego": emprego,
    }


def simul_(periodos: int, condicoes_iniciais: dict) -> dict:
    """Executa consumo induzido e investimento familiar/NF endógeno."""

    ci = condicoes_iniciais
    cfg = ci["config"]
    p = ci["parametros_cei"]
    periodo_choque = int(cfg["periodo_choque"])
    choque_permanente = cfg["choque_permanente"]
    if not isinstance(choque_permanente, bool):
        raise TypeError("choque_permanente deve ser True ou False.")
    if periodo_choque < 1 or periodo_choque > periodos:
        raise ValueError(
            "periodo_choque deve estar entre 1 e o número de períodos."
        )
    n = len(ci["setores"])
    I = pd.DataFrame(
        np.eye(n),
        index=ci["setores"],
        columns=ci["setores"],
    )
    inertia_pm = float(cfg["inertia_pm"])
    taxa_juros_nominal = ( 1 + cfg["taxa_juros_real"]) * ( 1 + cfg["a0"]) - 1

    # O período 0 é a base normalizada em 1. Para que a inflação exista de
    # fato desde o primeiro período simulado, os custos domésticos e importados
    # de t=1 já incorporam o nível herdado. Como o sistema de preços é
    # homogêneo, isso faz Pc_1 = Pb_1 = Pm_1 = 1 + inflacao_inicial.

    # a0 é simultaneamente o componente autônomo da variação salarial e a
    # inflação herdada utilizada para iniciar a simulação.

    inflacao_inicial = float(cfg["a0"])
   
    if inflacao_inicial <= -1.0:
        raise ValueError("inflacao_inicial deve ser maior que -1.")

    indice_salarios = 1.0 + inflacao_inicial
    indice_cambio = 1.0 + inflacao_inicial
    indice_precos_anterior = 1.0
    consumo_nominal = float(
        ci["valores_cei"].iat[L["consumo"], C["familias_s"]]
    )
    pesos_consumo = ci["consumo_base"] / float(ci["consumo_base"].sum())

    # A FBCF das famílias é destinada integralmente ao setor de Construção.
    fbcf_familias_base = float(
        ci["valores_cei"].iat[L["fbcf"], C["familias_s"]]
    )
    setor_construcao = "F - Construção"

    if setor_construcao not in ci["setores"]:
        raise KeyError(
            f"O setor '{setor_construcao}' não foi encontrado na TRU."
        )

    pesos_investimento_familias = pd.Series(
        0.0,
        index=ci["setores"],
        name="peso_investimento_familias",
    )

    pesos_investimento_familias.loc[setor_construcao] = 1.0
    
    investimento_familias_base = (
        pesos_investimento_familias * fbcf_familias_base
    )

    calibracao_nf = ci["investimento_nf"]
    investimento_nf_base = calibracao_nf[
        "fbcf_nf_base_fornecedor"
    ].copy()

    # Componentes fixos do investimento. Eles são mantidos separados para que
    # a TRU seja construída pelas decisões institucionais, e não para que a CEI
    # receba posteriormente um resíduo do investimento total.

    fbcf_fixa_base = calibracao_nf["fbcf_outros_base"].copy()

    estoques_base = calibracao_nf["estoques_base"].copy()

    investimento_fixo_base = fbcf_fixa_base + estoques_base
    if not np.allclose(
        investimento_familias_base
        + investimento_nf_base
        + investimento_fixo_base,
        ci["investimento_base"],
        atol=1e-6,
    ):
        raise RuntimeError("A decomposição do investimento inicial não fechou.")

    # Valores institucionais observados na CEI-base. Governo, firmas
    # financeiras, exterior e estoques permanecem componentes fixos por
    # enquanto. Famílias e firmas NF são substituídas pelas funções endógenas.
    fbcf_fixa_cei_base = {
        "governo": float(
            ci["valores_cei"].iat[L["fbcf"], C["governo_s"]]
        ),
        "firmas_financeiras": float(
            ci["valores_cei"].iat[L["fbcf"], C["ff_s"]]
        ),
        "setor_externo": float(
            ci["valores_cei"].iat[L["fbcf"], C["externo_s"]]
        ),
    }
    estoques_cei_base = {
        "governo": float(
            ci["valores_cei"].iat[L["estoques"], C["governo_s"]]
        ),
        "firmas_financeiras": float(
            ci["valores_cei"].iat[L["estoques"], C["ff_s"]]
        ),
        "firmas_nao_financeiras": float(
            ci["valores_cei"].iat[L["estoques"], C["nf_s"]]
        ),
        "setor_externo": float(
            ci["valores_cei"].iat[L["estoques"], C["externo_s"]]
        ),
    }
    if not np.isclose(
        sum(fbcf_fixa_cei_base.values()),
        float(fbcf_fixa_base.sum()),
        atol=1e-6,
    ):
        raise RuntimeError("A FBCF fixa da CEI não coincide com a TRU-base.")
    if not np.isclose(
        sum(estoques_cei_base.values()),
        float(estoques_base.sum()),
        atol=1e-6,
    ):
        raise RuntimeError("Os estoques da CEI não coincidem com a TRU-base.")
    if not np.isclose(
        estoques_cei_base["governo"]
        + estoques_cei_base["firmas_financeiras"]
        + estoques_cei_base["setor_externo"],
        0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "A hipótese atual exige que os estoques pertençam às firmas NF."
        )

    # Estados reais do acelerador. beta e v são sempre os mesmos parâmetros
    # calibrados; somente a forma de construir o estado inicial pode mudar.
    beta_investimento_nf = float(calibracao_nf["beta"])
    v_investimento_nf = float(calibracao_nf["v"])
    depreciacao_capital_nf = float(calibracao_nf["depreciacao"])
    setores_nf = calibracao_nf["setores_nf"]
    producao_nf_corrente = (
        calibracao_nf["producao_real"]
        .loc[cfg["ano"], setores_nf]
        .copy()
    )
    investimento_nf_base_por_investidor = calibracao_nf[
        "investimento_nf_base_por_investidor"
    ].copy()

    inicializacao_investimento_nf = cfg["inicializacao_investimento_nf"]
    if inicializacao_investimento_nf == "estacionaria":
        # Supõe que a produção observada em t=0 também ocorreu em t=-1.
        # Portanto, ΔY_e,0 = 0 e todo o investimento bruto observado no
        # ano-base é reposição da depreciação: K_0 = I_0 / depreciação.
        producao_nf_anterior = producao_nf_corrente.copy()
        estoque_capital_nf = (
            investimento_nf_base_por_investidor
            / depreciacao_capital_nf
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
            calibracao_nf["producao_real"]
            .loc[cfg["ano"] - 1, setores_nf]
            .copy()
        )
        estoque_capital_nf = calibracao_nf[
            "estoque_capital_nf_base"
        ].copy()
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
    pesos_bens_capital_nf = calibracao_nf[
        "pesos_bens_capital_nf"
    ].copy()

    # ------------------------------------------------------------------
    # Estoques das firmas não financeiras
    # ------------------------------------------------------------------
    # Um zero na TRU-base é tratado como zero estrutural: esse setor nunca
    # forma estoques. Valores positivos e negativos identificam os setores que
    # participam da dinâmica.
    #
    # A variação observada em 2020 é mantida como componente autônomo
    # recorrente: ela representa a desova/depreciação normal dos estoques no
    # estado estacionário e, por isso, não é acumulada no estoque cíclico. O
    # modelo acumula separadamente apenas a variação provocada por desvios da
    # produção esperada. Sem mudança da produção, a variação total permanece
    # igual à TRU-base em todos os períodos.
    setores_com_estoques = (estoques_base.abs() > 1e-9).rename(
        "setor_com_estoques"
    )
    razao_estoque_producao = float(cfg["razao_estoque_producao"])
    velocidade_ajuste_estoques = float(cfg["velocidade_ajuste_estoques"])
    if razao_estoque_producao < 0.0:
        raise ValueError("razao_estoque_producao não pode ser negativa.")
    if not 0.0 <= velocidade_ajuste_estoques <= 1.0:
        raise ValueError("velocidade_ajuste_estoques deve estar entre 0 e 1.")

    producao_estoques_base = (
        calibracao_nf["producao_real"]
        .loc[cfg["ano"], ci["setores"]]
        .copy()
    )
    producao_estoques_corrente = producao_estoques_base.copy()
    producao_estoques_anterior = producao_estoques_corrente.copy()
    variacao_autonoma_estoques = estoques_base.copy().rename(
        "variacao_autonoma_estoques"
    )
    estoque_referencia = (
        razao_estoque_producao
        * producao_estoques_base
        * setores_com_estoques.astype(float)
    ).rename("estoque_referencia")
    estoque_ciclico = pd.Series(
        0.0,
        index=ci["setores"],
        name="estoque_ciclico",
    )
    estoque_real = (estoque_referencia + estoque_ciclico).rename(
        "estoque_real"
    )

    # Valores nominais das demandas autônomas no período anterior. O choque
    # será aplicado uma única vez no período definido em CONFIG. Antes e
    # depois dele, esses valores são corrigidos pela variação dos preços.
    governo_nominal_anterior = ci["governo_base"].copy()
    fbcf_fixa_nominal_anterior = fbcf_fixa_base.copy()
    exportacoes_nominais_anterior = ci["exportacoes_base"].copy()
    pc_anterior = pd.Series(
        1.0,
        index=ci["setores"],
        name="preco_comprador_anterior",
    )

    # ------------------------------------------------------------------
    # Ativos e passivos financeiros por setor institucional
    # ------------------------------------------------------------------
    # Os dois estoques brutos do ano-base são inferidos separadamente:
    #   ativos_0  = juros recebidos_0 / taxa real;
    #   passivos_0 = juros pagos_0 / taxa real.
    # Assim, os juros brutos observados na CEI são reproduzidos sem perder a
    # distinção entre entrada e saída. O estoque líquido é apenas o resultado
    # ativos - passivos, e não mais o único estoque mantido pelo modelo.
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
            nome: float(ci["valores_cei"].iat[L["juros"], entrada])
            for nome, (entrada, saida) in COLUNAS_SETORES.items()
        },
        name="juros_recebidos",
    )
    juros_pagos_base = pd.Series(
        {
            nome: float(ci["valores_cei"].iat[L["juros"], saida])
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

    # Essa poupança do ano-base determina a FBCF das famílias no período 1.
    poupanca_familias_anterior = float(
        ci["valores_cei"].iloc[1:13, C["familias_e"]].sum()
        - ci["valores_cei"].iloc[1:9, C["familias_s"]].sum()
        - ci["valores_cei"].iat[L["consumo"], C["familias_s"]]
    )

    capacidade_base = {
        nome: float(
            ci["valores_cei"].iloc[1:16, entrada].sum()
            - ci["valores_cei"].iloc[1:16, saida].sum()
        )
        for nome, (entrada, saida) in COLUNAS_SETORES.items()
    }

    # A última linha existente no arquivo da CEI não coincide com os fluxos
    # corrigidos das linhas 1–15 para FF e NF. O período 0 deve mostrar os
    # saldos recalculados acima, que reproduzem o gabarito, e não os valores
    # antigos gravados na planilha de entrada.
    cei_periodo_zero = ci["cei_original"].copy(deep=True)

    for nome, (entrada, _) in COLUNAS_SETORES.items():
        cei_periodo_zero.iloc[L["capacidade"], entrada] = capacidade_base[nome]


    pib_base = float(
        ci["va_base"].loc[VA["total"]].sum()
        + (ci["taxa_impostos"] * ci["demanda_final_base"]).sum()
    )

    # Salvar dados:
   
    historico = [{
        "periodo": 0,
        "ano": cfg["ano"],
        # O nível de preços do ano-base é normalizado em 1. A inflação inicial
        # informa a passagem para t=1; ela não altera retroativamente os valores
        # nominais observados na CEI-base.
        "indice_precos": 1.0,
        "inflacao": inflacao_inicial,
        "indice_salarios": 1.0,
        "indice_cambio": 1.0,
        "taxa_juros_nominal": ( 1+ taxa_juros_real) * ( 1 + inflacao_inicial) - 1,
        "pib_real": pib_base,
        "pib_nominal": pib_base,
        "emprego": float(ci["va_base"].loc["Fator trabalho (ocupações)"].sum()),
        "taxa_desemprego": cfg["taxa_desemprego_base"],
        "consumo_real": float(ci["consumo_base"].sum()),
        "consumo_nominal": consumo_nominal/pib_base,
        "poupanca_familias_nominal": poupanca_familias_anterior,
        "fbcf_familias_nominal": fbcf_familias_base,
        "fbcf_nf_real": float(investimento_nf_base.sum()),
        "fbcf_nf_nominal": float(investimento_nf_base.sum()),
        "fbcf_fixa_nominal": float(fbcf_fixa_base.sum()),
        "variacao_estoques_real": float(estoques_base.sum()),
        "variacao_estoques_nominal": float(estoques_base.sum()),
        "variacao_autonoma_estoques_real": float(estoques_base.sum()),
        "variacao_ciclica_estoques_real": 0.0,
        "estoque_real": float(estoque_real.sum()),
        "investimento_liquido_nf_real": float(
            investimento_liquido_nf_base.sum()
        ),
        "investimento_reposicao_nf_real": float(
            investimento_reposicao_nf_base.sum()
        ),
        "ajuste_piso_investimento_nf_real": 0.0,
        "estoque_capital_nf_real": float(estoque_capital_nf.sum()),
        "setores_no_piso_investimento_nf": int(
            (
                investimento_liquido_nf_base
                + investimento_reposicao_nf_base
                < 0.0
            ).sum()
        ),
        "residuo_consumo": 0.0,
        "iteracoes_consumo": 0,
        "deficit_governo": -capacidade_base["governo"]/pib_base,
        "saldo_setor_externo": capacidade_base["setor_externo"]/pib_base,
        "discrepancia_cei": sum(capacidade_base.values()),
    }]
    pc_periodos = {
        0: pd.Series(1.0, index=ci["setores"], name="preco_comprador")
    }
    pb_periodos = {
        0: pd.Series(1.0, index=ci["setores"], name="preco_basico")
    }
    pm_periodos = {
        0: pd.Series(1.0, index=ci["setores"], name="preco_importacoes")
    }
    cei_periodos = {0: cei_periodo_zero}
    capacidades = {0: capacidade_base}
    ativos_financeiros_periodos = {0: ativos_financeiros.copy()}
    passivos_financeiros_periodos = {0: passivos_financeiros.copy()}
    estoque_financeiro_periodos = {0: estoque_financeiro.copy()}
    aquisicao_ativos_periodos = {
        0: pd.Series(
            0.0,
            index=list(COLUNAS_SETORES),
            name="aquisicao_ativos_financeiros",
        )
    }
    emissao_passivos_periodos = {
        0: pd.Series(
            0.0,
            index=list(COLUNAS_SETORES),
            name="emissao_passivos_financeiros",
        )
    }

    juros_liquidos_periodos = {0: juros_liquidos_base.copy()}
    juros_recebidos_periodos = {0: juros_recebidos_base.copy()}
    juros_pagos_periodos = {0: juros_pagos_base.copy()}
    reavaliacao_financeira_periodos = {
        0: pd.Series(
            0.0,
            index=list(COLUNAS_SETORES),
            name="reavaliacao_financeira",
        )
    }
    tru_real_periodos = {}
    tru_nominal_periodos = {}
    investimento_nf_real_periodos = {0: investimento_nf_base.copy()}
    investimento_nf_nominal_periodos = {0: investimento_nf_base.copy()}
    fbcf_fixa_nominal_periodos = {0: fbcf_fixa_base.copy()}
    variacao_estoques_real_periodos = {0: estoques_base.copy()}
    variacao_estoques_nominal_periodos = {0: estoques_base.copy()}
    variacao_autonoma_estoques_real_periodos = {0: estoques_base.copy()}
    variacao_ciclica_estoques_real_periodos = {
        0: pd.Series(0.0, index=ci["setores"])
    }
    estoque_real_periodos = {0: estoque_real.copy()}
    estoque_referencia_periodos = {0: estoque_referencia.copy()}
    estoque_ciclico_periodos = {0: estoque_ciclico.copy()}
    investimento_nf_por_investidor_periodos = {
        0: investimento_nf_base_por_investidor.copy()
    }
    estoque_capital_nf_periodos = {0: estoque_capital_nf.copy()}

    # ==================================================================
    # FOR TEMPORAL
    # ==================================================================

    for t in range(1, periodos + 1):

        # A FBCF familiar do período t depende da poupança observada em t-1.
        # Como a poupança passada está em valor nominal, primeiro retiramos o
        # preço da Construção de t-1. Isso preserva a quantidade real que as
        # famílias conseguem comprar. Depois de calcular os preços de t, essa
        # quantidade será valorizada pelo preço corrente da Construção.
        poupanca_familias_real_anterior = (
            poupanca_familias_anterior
            / float(pc_anterior.loc[setor_construcao])
        )
        fbcf_familias_real = (
            p["prop_invest_fbcf_familias"]
            * poupanca_familias_real_anterior
        )

        # ESTOQUES
        # A variação autônoma observada em 2020 se repete como desova normal,
        # mas não reduz o estoque de referência. O estoque cíclico mede somente
        # o desvio provocado pela produção esperada em relação à produção-base:
        #   Y_e,t       = Y_(t-1) + beta * [Y_(t-1) - Y_(t-2)]
        #   E_cíclico*  = razão_estoque * (Y_e,t - Y_base)
        #   DeltaE_cíc. = velocidade * (E_cíclico* - E_cíclico_(t-1))
        #   DeltaE      = DeltaE_autônoma + DeltaE_cíclica
        # Sem choque, Y_e,t = Y_base e a parte cíclica permanece igual a zero.

        producao_esperada_estoques = (
            producao_estoques_corrente
            + beta_investimento_nf
            * (producao_estoques_corrente - producao_estoques_anterior)
        ).clip(lower=0.0).rename("producao_esperada_estoques")

        variacao_autonoma_estoques_periodo = (
            variacao_autonoma_estoques.copy()
        )

        estoque_referencia_periodo = estoque_referencia.copy()

        estoque_ciclico_desejado = (
            razao_estoque_producao
            * (producao_esperada_estoques - producao_estoques_base)
            * setores_com_estoques.astype(float)
        ).rename("estoque_ciclico_desejado")

        variacao_ciclica_estoques = (
            velocidade_ajuste_estoques
            * (estoque_ciclico_desejado - estoque_ciclico)
        ).rename("variacao_ciclica_estoques")

        # Apenas a parte cíclica altera o estoque físico acompanhado pelo
        # modelo e, portanto, é ela que deve respeitar o piso zero.

        variacao_ciclica_estoques = variacao_ciclica_estoques.clip(
            lower=-(estoque_referencia_periodo + estoque_ciclico)
        )

        variacao_ciclica_estoques.loc[~setores_com_estoques] = 0.0

        estoque_ciclico_periodo = (
            estoque_ciclico + variacao_ciclica_estoques
        ).rename("estoque_ciclico")

        variacao_estoques_real = (
            variacao_autonoma_estoques_periodo
            + variacao_ciclica_estoques
        ).rename("variacao_estoques_real")

        variacao_estoques_real.loc[~setores_com_estoques] = 0.0

        estoque_real_periodo = (
            estoque_referencia_periodo + estoque_ciclico_periodo
        ).rename("estoque_real")

        if np.any(estoque_real_periodo < -1e-9):
            raise RuntimeError(f"Estoque físico negativo no período {t}.")

        # Investimento real das firmas não financeiras por setor investidor:
        #   ΔY_e,t       = beta * [Y_(t-1) - Y_(t-2)]
        #   I_líquido,t  = v * ΔY_e,t
        #   I_reposição  = depreciação * K_(t-1)
        #   I_bruto,t    = max[0, I_líquido,t + I_reposição,t]
        #   K_t          = (1 - depreciação) * K_(t-1) + I_bruto,t
        #
        # Se a produção permanecer constante, o investimento líquido é zero e
        # o investimento bruto repõe exatamente a depreciação. Se o resultado
        # bruto for negativo, o setor investe zero e deixa o capital depreciar.
        
        variacao_producao_esperada_nf = (
            beta_investimento_nf
            * (producao_nf_corrente - producao_nf_anterior)
        ).rename("variacao_producao_esperada_nf")

        investimento_liquido_nf = (
            v_investimento_nf * variacao_producao_esperada_nf
        ).rename("investimento_liquido_nf")

        investimento_reposicao_nf = (
            depreciacao_capital_nf * estoque_capital_nf
        ).rename("investimento_reposicao_nf")

        investimento_nf_sem_piso = (
            investimento_liquido_nf + investimento_reposicao_nf
        ).rename("investimento_nf_sem_piso")

        investimento_nf_por_investidor = (
            investimento_nf_sem_piso.clip(lower=0.0)
        ).rename("investimento_nf_por_setor_investidor")

        estoque_capital_nf_periodo = (
            investimento_nf_por_investidor
            + (1.0 - depreciacao_capital_nf) * estoque_capital_nf
        ).rename("estoque_capital_nf")

        investimento_nf_real_total = float(
            investimento_nf_por_investidor.sum()
        )
        investimento_nf_real = (
            pesos_bens_capital_nf * investimento_nf_real_total
        ).rename("investimento_nf_real")

        pm = pd.Series(
            indice_cambio,
            index=ci["setores"],
            name="preco_importacoes",
        )
        v = (ci["v0"] * indice_salarios).rename("coeficiente_va")

        # Preços do período.
        pc = pd.Series(
            np.linalg.solve(
                (I - ci["G"] @ ci["Sd"] @ ci["A_precos"].T).to_numpy(),
                (ci["G"] @ (ci["Sd"] @ v + ci["Sm"] @ pm)).to_numpy(),
            ),
            index=ci["setores"],
            name="preco_comprador",
        )
        pb = (ci["A_precos"].T @ pc + v).rename("preco_basico")
        if np.any(pc <= 0.0) or np.any(pb <= 0.0):
            raise RuntimeError(f"Preço não positivo no período {t}.")

        # A FBCF real decidida a partir da poupança de t-1 é paga ao preço da
        # Construção vigente em t. Assim, inflação não reduz artificialmente o
        # investimento real das famílias.
        fbcf_familias = (
            fbcf_familias_real * float(pc.loc[setor_construcao])
        )

        # A equação do capital produz quantidades reais. Cada bem de capital é
        # valorizado pelo preço comprador do setor que o fornece.
        investimento_nf_nominal = (
            investimento_nf_real * pc
        ).rename("investimento_nf_nominal")

        fbcf_nf_nominal = float(investimento_nf_nominal.sum())
        variacao_estoques_nominal = (
            variacao_estoques_real * pc
        ).rename("variacao_estoques_nominal")

        variacao_autonoma_estoques_nominal = (
            variacao_autonoma_estoques_periodo * pc
        ).rename("variacao_autonoma_estoques_nominal")

        variacao_ciclica_estoques_nominal = (
            variacao_ciclica_estoques * pc
        ).rename("variacao_ciclica_estoques_nominal")
        variacao_estoques_nominal_total = float(
            variacao_estoques_nominal.sum()
        )

        if choque_permanente:
            # O novo nível real permanece após o período do choque. Como pc é
            # um nível acumulado de preços, pc_t/pc_(t-1) corrige o valor
            # nominal somente pela inflação setorial do período.
            variacao_precos_setoriais = pc / pc_anterior
            governo_nominal = (
                governo_nominal_anterior * variacao_precos_setoriais
            )
            fbcf_fixa_nominal = (
                fbcf_fixa_nominal_anterior
                * variacao_precos_setoriais
            )
            exportacoes_nominais = (
                exportacoes_nominais_anterior * variacao_precos_setoriais
            )
        else:
            # O multiplicador provoca um salto permanente no nível nominal,
            # mas esse novo valor nominal fica congelado: não há correção pela
            # inflação. Portanto, seu valor real diminui quando pc aumenta.
            variacao_precos_setoriais = pc / pc_anterior
            governo_nominal = (
                governo_nominal_anterior * variacao_precos_setoriais
            )
            fbcf_fixa_nominal = (
                fbcf_fixa_nominal_anterior
                * variacao_precos_setoriais
            )
            exportacoes_nominais = (
                exportacoes_nominais_anterior * variacao_precos_setoriais
            )

        # O multiplicador é aplicado somente no período escolhido. Quando o
        # choque é permanente em termos reais, o valor nominal também acompanha
        # os preços; no outro caso, o salto nominal permanece sem indexação.
        if t == periodo_choque:

            governo_nominal *= cfg["multiplicador_governo"]
            fbcf_fixa_nominal *= cfg["multiplicador_investimento"]
            exportacoes_nominais *= cfg["multiplicador_exportacoes"]

        elif not choque_permanente and t == periodo_choque + 1:

            governo_nominal /= cfg["multiplicador_governo"]
            fbcf_fixa_nominal /= cfg["multiplicador_investimento"]
            exportacoes_nominais /= cfg["multiplicador_exportacoes"]

        investimento_nominal = (
            fbcf_fixa_nominal
            + variacao_estoques_nominal
            + pesos_investimento_familias * fbcf_familias
            + investimento_nf_nominal
        )

        # Compatibiliza a composição por produto da TRU com as instituições
        # da CEI. Não há resíduo: os fatores abaixo apenas valorizam, pelo
        # índice de preços da respectiva cesta, os valores fixos observados na
        # CEI-base. Famílias e firmas NF continuam fora desses blocos.

        fbcf_fixa_total_base = float(fbcf_fixa_base.sum())

        fbcf_fixa_total = float(fbcf_fixa_nominal.sum())
        
        if np.isclose(fbcf_fixa_total_base, 0.0):
            if not np.isclose(fbcf_fixa_total, 0.0):
                raise RuntimeError("A FBCF fixa partiu de zero e tornou-se não nula.")
            fator_fbcf_fixa = 1.0
        else:
            fator_fbcf_fixa = fbcf_fixa_total / fbcf_fixa_total_base

        investimentos_fixos_cei = {
            "fbcf_governo": (
                fbcf_fixa_cei_base["governo"] * fator_fbcf_fixa
            ),
            "fbcf_firmas_financeiras": (
                fbcf_fixa_cei_base["firmas_financeiras"] * fator_fbcf_fixa
            ),
            "fbcf_setor_externo": (
                fbcf_fixa_cei_base["setor_externo"] * fator_fbcf_fixa
            ),
            "estoques_governo": 0.0,
            "estoques_firmas_financeiras": 0.0,
            "estoques_firmas_nao_financeiras": (
                variacao_estoques_nominal_total
            ),
            "estoques_setor_externo": 0.0,
        }

        indice_precos = float(
            ci["consumo_base"].sum() / (ci["consumo_base"] / pc).sum()
        )
        inflacao = indice_precos / indice_precos_anterior - 1.0

        # Ativos e passivos financeiros são estoques nominais. Antes do cálculo
        # dos juros, ambos são corrigidos pela inflação agregada do período.
        # Como os dois lados recebem exatamente o mesmo fator, o total de ativos
        # continua igual ao total de passivos, enquanto seus valores reais
        # permanecem constantes na ausência de novas transações financeiras.
        fator_inflacao_financeira = (
            indice_precos / indice_precos_anterior
        )
        ativos_financeiros_corrigidos = (
            ativos_financeiros * fator_inflacao_financeira
        ).rename("ativos_financeiros")
        passivos_financeiros_corrigidos = (
            passivos_financeiros * fator_inflacao_financeira
        ).rename("passivos_financeiros")

        # A taxa configurada é real. A taxa nominal incorpora a inflação do
        # período pela relação de Fisher. No cenário-base sem inflação, ambas
        # coincidem em 6%.

        if t == 1:
            taxa_juros_nominal = (1 + taxa_juros_real) * ( 1+ inflacao) - 1

        taxa_juros_nominal = (1 - inertia_pm) * (
            (1.0 + taxa_juros_real) * (1.0 + inflacao) - 1.0
        ) + inertia_pm * taxa_juros_nominal

        # Os juros de t incidem sobre os estoques brutos existentes ao final de
        # t-1. Não é necessária nenhuma regra proporcional baseada no estoque
        # líquido: cada setor conserva seus ativos e passivos separadamente.
        juros_recebidos = (
            taxa_juros_nominal * ativos_financeiros_corrigidos
        ).rename("juros_recebidos")

        juros_pagos = (
            taxa_juros_nominal * passivos_financeiros_corrigidos
        ).rename("juros_pagos")

        juros_liquidos = (juros_recebidos - juros_pagos).rename(
            "juros_liquidos"
        )

        consumo_nominal = consumo_nominal * (
                indice_precos / indice_precos_anterior
            )

        precos = {
            "pc": pc,
            "pb": pb,
            "pm": pm,
            "indice_salarios": indice_salarios,
        }
        parametros_cei_periodo = {
            "calibrados": p,
            "valores_base": ci["valores_cei"],
            "indice_salarios": indice_salarios,
            "setor_financeiro": cfg["setor_financeiro"],
            "fbcf_familias": fbcf_familias,
            "fbcf_nf": fbcf_nf_nominal,
            "investimentos_fixos": investimentos_fixos_cei,
            "juros_recebidos": juros_recebidos,
            "juros_pagos": juros_pagos,
        }

        # ==============================================================
        # CICLO DO CONSUMO NOMINAL
        # A FBCF das famílias já foi determinada pela poupança de t-1.
        # ==============================================================
        convergiu = False
        
        for iteracao in range(cfg["max_iteracoes_consumo"] + 1):

            demandas_nominais = {
                "consumo": pesos_consumo * consumo_nominal,
                "governo": governo_nominal,
                "investimento": investimento_nominal,
                "exportacoes": exportacoes_nominais,
            }

            results_TRU = simul_TRU(demandas_nominais, precos, ci)
            
            results_CEI = simul_CEI(
                parametros_cei_periodo,
                results_TRU,
                ci["cei_original"],
                indice_precos,
            )

            consumo_cei = results_CEI["consumo_nominal"]
            residuo_consumo = consumo_nominal - consumo_cei
            if abs(residuo_consumo) <= cfg["tolerancia_consumo"]:
                convergiu = True
                break
            consumo_nominal = max(0.0, consumo_cei)

        if not convergiu:
            raise RuntimeError(
                f"Consumo não convergiu no período {t} após "
                f"{cfg['max_iteracoes_consumo']} substituições. "
                f"Resíduo final = {residuo_consumo}."
            )

        # Emprego corrente determina o salário usado no próximo período.
        emprego = results_CEI["emprego"]
        taxa_desemprego = max(
            0.0,
            (p["pea"] - emprego) / p["pea"],
        )
        if taxa_desemprego == 0.0:
            variacao_salarios = 1.1
        else:
            variacao_salarios = max(
                -0.99,
                float(
                    cfg["a0"]
                    + cfg["a1"]
                    * (
                        (cfg["taxa_desemprego_base"] / taxa_desemprego)
                        ** cfg["a3"]
                        - 1.0
                    )
                ),
            )
            variacao_salarios = min(1.1, variacao_salarios)

        real = results_TRU["real"]
        nominal = results_TRU["nominal"]
        capacidade = results_CEI["capacidade_financiamento"]
        capacidade_serie = pd.Series(
            capacidade,
            index=list(COLUNAS_SETORES),
            name="capacidade_financiamento",
            dtype=float,
        )
        # A reavaliação possui sinal contrário ao saldo financeiro. A parcela
        # restante é acumulada de forma explícita nos estoques brutos:
        #   saldo positivo -> aquisição de ativos;
        #   saldo negativo -> emissão de passivos.
        # Com fração de reavaliação igual a 1, os dois fluxos são nulos e os
        # estoques ficam constantes. Com fração 0, todo o B.9 é incorporado.
        reavaliacao_financeira = (
            -fracao_reavaliacao_financeira * capacidade_serie
        ).rename("reavaliacao_financeira")
        saldo_financeiro_incorporado = (
            capacidade_serie + reavaliacao_financeira
        ).rename("saldo_financeiro_incorporado")
        aquisicao_ativos = saldo_financeiro_incorporado.clip(
            lower=0.0
        ).rename("aquisicao_ativos_financeiros")
        emissao_passivos = (-saldo_financeiro_incorporado).clip(
            lower=0.0
        ).rename("emissao_passivos_financeiros")
        ativos_financeiros_periodo = (
            ativos_financeiros_corrigidos + aquisicao_ativos
        ).rename("ativos_financeiros")
        passivos_financeiros_periodo = (
            passivos_financeiros_corrigidos + emissao_passivos
        ).rename("passivos_financeiros")
        estoque_financeiro_periodo = (
            ativos_financeiros_periodo - passivos_financeiros_periodo
        ).rename("estoque_financeiro")

        if not np.isclose(
            reavaliacao_financeira.sum(),
            0.0,
            atol=1e-4,
        ):
            raise RuntimeError(
                "As reavaliações financeiras não somam zero."
            )
        if not np.isclose(
            ativos_financeiros_periodo.sum(),
            passivos_financeiros_periodo.sum(),
            atol=1e-4,
        ):
            raise RuntimeError(
                "Os totais de ativos e passivos financeiros deixaram de fechar."
            )

        pib_nominal = float(
                nominal["valor_adicionado"].loc[VA["total"]].sum()
                + nominal["impostos_produtos"].sum()
            )

        historico.append({
            "periodo": t,
            "ano": cfg["ano"] + t,
            "indice_precos": indice_precos,
            "inflacao": inflacao,
            "indice_salarios": indice_salarios,
            "indice_cambio": indice_cambio,
            "taxa_juros_nominal": taxa_juros_nominal,
            "pib_real": float(
                real["valor_adicionado"].loc[VA["total"]].sum()
                + real["impostos_produtos"].sum()
            ),
            "pib_nominal":pib_nominal,
            "emprego": emprego,
            "taxa_desemprego": taxa_desemprego,
            "consumo_real": float(real["consumo_familias"].sum()),
            "consumo_nominal": consumo_cei/pib_nominal,
            "poupanca_familias_nominal": results_CEI["poupanca_familias"],
            "fbcf_familias_nominal": fbcf_familias,
            "fbcf_nf_real": investimento_nf_real_total,
            "fbcf_nf_nominal": fbcf_nf_nominal,
            "fbcf_fixa_nominal": fbcf_fixa_total,
            "variacao_estoques_real": float(variacao_estoques_real.sum()),
            "variacao_estoques_nominal": variacao_estoques_nominal_total,
            "variacao_autonoma_estoques_real": float(
                variacao_autonoma_estoques_periodo.sum()
            ),
            "variacao_ciclica_estoques_real": float(
                variacao_ciclica_estoques.sum()
            ),
            "estoque_real": float(estoque_real_periodo.sum()),
            "investimento_liquido_nf_real": float(
                investimento_liquido_nf.sum()
            ),
            "investimento_reposicao_nf_real": float(
                investimento_reposicao_nf.sum()
            ),
            "ajuste_piso_investimento_nf_real": float(
                (
                    investimento_nf_por_investidor
                    - investimento_nf_sem_piso
                ).sum()
            ),
            "estoque_capital_nf_real": float(estoque_capital_nf_periodo.sum()),
            "setores_no_piso_investimento_nf": int(
                (investimento_nf_sem_piso < 0.0).sum()
            ),
            "residuo_consumo": residuo_consumo,
            "iteracoes_consumo": iteracao,
            "deficit_governo": -capacidade["governo"]/pib_nominal,
            "saldo_setor_externo": capacidade["setor_externo"]/pib_nominal,
            "discrepancia_cei": sum(capacidade.values()),
        })

        pc_periodos[t] = pc.copy()
        pb_periodos[t] = pb.copy()
        pm_periodos[t] = pm.copy()
        cei_periodos[t] = results_CEI["cei"].copy(deep=True)
        capacidades[t] = capacidade.copy()
        ativos_financeiros_periodos[t] = ativos_financeiros_periodo.copy()
        passivos_financeiros_periodos[t] = passivos_financeiros_periodo.copy()
        estoque_financeiro_periodos[t] = estoque_financeiro_periodo.copy()
        aquisicao_ativos_periodos[t] = aquisicao_ativos.copy()
        emissao_passivos_periodos[t] = emissao_passivos.copy()
        juros_liquidos_periodos[t] = juros_liquidos.copy()
        juros_recebidos_periodos[t] = juros_recebidos.copy()
        juros_pagos_periodos[t] = juros_pagos.copy()
        reavaliacao_financeira_periodos[t] = (
            reavaliacao_financeira.copy()
        )
        tru_real_periodos[t] = real
        tru_nominal_periodos[t] = nominal
        investimento_nf_real_periodos[t] = investimento_nf_real.copy()
        investimento_nf_nominal_periodos[t] = investimento_nf_nominal.copy()
        fbcf_fixa_nominal_periodos[t] = fbcf_fixa_nominal.copy()
        variacao_estoques_real_periodos[t] = variacao_estoques_real.copy()
        variacao_estoques_nominal_periodos[t] = (
            variacao_estoques_nominal.copy()
        )
        variacao_autonoma_estoques_real_periodos[t] = (
            variacao_autonoma_estoques_periodo.copy()
        )
        variacao_ciclica_estoques_real_periodos[t] = (
            variacao_ciclica_estoques.copy()
        )
        estoque_real_periodos[t] = estoque_real_periodo.copy()
        estoque_referencia_periodos[t] = estoque_referencia_periodo.copy()
        estoque_ciclico_periodos[t] = estoque_ciclico_periodo.copy()
        investimento_nf_por_investidor_periodos[t] = (
            investimento_nf_por_investidor.copy()
        )
        estoque_capital_nf_periodos[t] = estoque_capital_nf_periodo.copy()
        

        indice_salarios = indice_salarios * ( 1.0 + variacao_salarios )
        indice_cambio = indice_cambio * ( 1 + cfg["repasse_inflacao_cambio"] * inflacao )
        indice_precos_anterior = indice_precos
        pc_anterior = pc.copy()
        governo_nominal_anterior = governo_nominal.copy()
        fbcf_fixa_nominal_anterior = fbcf_fixa_nominal.copy()
        exportacoes_nominais_anterior = exportacoes_nominais.copy()
        consumo_nominal = consumo_cei
        poupanca_familias_anterior = results_CEI["poupanca_familias"]
        estoque_capital_nf = estoque_capital_nf_periodo.copy()
        ativos_financeiros = ativos_financeiros_periodo.copy()
        passivos_financeiros = passivos_financeiros_periodo.copy()
        estoque_financeiro = estoque_financeiro_periodo.copy()
        estoque_real = estoque_real_periodo.copy()
        estoque_referencia = estoque_referencia_periodo.copy()
        estoque_ciclico = estoque_ciclico_periodo.copy()
        producao_nf_anterior = producao_nf_corrente.copy()
        producao_nf_corrente = (
            real["producao_domestica"].loc[setores_nf].copy()
        )
        producao_estoques_anterior = producao_estoques_corrente.copy()
        producao_estoques_corrente = real["producao_domestica"].copy()

    return {
        "inicializacao_investimento_nf": inicializacao_investimento_nf,
        "historico": pd.DataFrame(historico).set_index("periodo"),
        "precos_comprador": pd.DataFrame(pc_periodos).T,
        "precos_basicos": pd.DataFrame(pb_periodos).T,
        "precos_importacoes": pd.DataFrame(pm_periodos).T,
        "cei": cei_periodos,
        "capacidade_financiamento": capacidades,
        "ativos_financeiros": pd.DataFrame(ativos_financeiros_periodos).T,
        "passivos_financeiros": pd.DataFrame(passivos_financeiros_periodos).T,
        "estoque_financeiro": pd.DataFrame(estoque_financeiro_periodos).T,
        "aquisicao_ativos_financeiros": pd.DataFrame(
            aquisicao_ativos_periodos
        ).T,
        "emissao_passivos_financeiros": pd.DataFrame(
            emissao_passivos_periodos
        ).T,
        "juros_liquidos": pd.DataFrame(juros_liquidos_periodos).T,
        "juros_recebidos": pd.DataFrame(juros_recebidos_periodos).T,
        "juros_pagos": pd.DataFrame(juros_pagos_periodos).T,
        "reavaliacao_financeira": pd.DataFrame(
            reavaliacao_financeira_periodos
        ).T,
        "tru_real": tru_real_periodos,
        "tru_nominal": tru_nominal_periodos,
        "investimento_nf_real": investimento_nf_real_periodos,
        "investimento_nf_nominal": investimento_nf_nominal_periodos,
        "fbcf_fixa_nominal": fbcf_fixa_nominal_periodos,
        "setores_com_estoques": setores_com_estoques,
        "variacao_estoques_real": variacao_estoques_real_periodos,
        "variacao_estoques_nominal": variacao_estoques_nominal_periodos,
        "variacao_autonoma_estoques_real": (
            variacao_autonoma_estoques_real_periodos
        ),
        "variacao_ciclica_estoques_real": (
            variacao_ciclica_estoques_real_periodos
        ),
        "estoque_real": estoque_real_periodos,
        "estoque_referencia_real": estoque_referencia_periodos,
        "estoque_ciclico_real": estoque_ciclico_periodos,
        "investimento_nf_por_setor_investidor": (
            investimento_nf_por_investidor_periodos
        ),
        "estoque_capital_nf_real": estoque_capital_nf_periodos,
    }
