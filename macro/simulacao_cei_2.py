"""Três blocos auditáveis: TRU, CEI e ciclo temporal."""

import numpy as np
import pandas as pd

from agentes.agregar_firmas import agregar_firmas, separar_agregados_firmas_cei
from macro.ciclo_abm import (
    atualizar_financeiro_periodo,
    atualizar_estado_periodo,
    calcular_inflacao_periodo,
    calcular_juros_periodo,
    calcular_mercado_trabalho,
    calcular_pib_legado,
    calcular_precos_ex_ante,
    calcular_precos_realizados,
    montar_demandas_periodo,
    montar_registro_historico,
)
from contabilidade.distribuicao_abm import (
    calcular_distribuicao_pre_mercado,
    extrair_fluxos_template_distribuicao,
)
from financeiro.financeiro_abm import inicializar_financeiro_abm
from mercados.atendimento_categorial_abm import ratear_atendimento_proporcional 
from calibracao.calibracao_investimento_nf_abm import calibrar_investimento_nf_abm
from contabilidade.estrutura_cei import (
    C,
    L,
    VA,
    COLUNAS_SETORES,
    LINHAS_BASE_IR_FIRMAS,
    LINHAS_OBRIGATORIAS,
)
from inicializacao.inicializar_firmas import inicializar_firmas
from agentes.importados_abm import inicializar_importados
from investimento.investimento_abm import (
    atualizar_demandas_autonomas,
    calcular_estoques_legado_periodo,
    calcular_fbcf_familias,
    inicializar_taxas_retorno_firmas,
    montar_investimento_e_cei_legado,
)
from mercados.mercados_abm import (
    executar_mercados_periodo,
)
from resultados.resultados_abm_legado import (
    armazenar_resultados_periodo,
    finalizar_resultados,
    inicializar_resultados_abm,
)


def registrar_distribuicao_pre_mercado_na_cei(
    distribuicao: dict,
    fluxos_transitorios: dict,
    CEI_inicial: pd.DataFrame,
    inflation_index: float,
) -> dict:
    """Registra na CEI os fluxos pré-mercado já calculados pelo ABM."""

    # Temporariamente, células sem fluxo próprio ainda carregam o template
    # histórico indexado. A distribuição não é recalculada neste registrador.
    cei = CEI_inicial.copy(deep=True)
    cei.iloc[1:16, 1:11] = (
        CEI_inicial.iloc[1:16, 1:11].fillna(0.0).to_numpy(dtype=float)
        * inflation_index
    )

    juros_recebidos = distribuicao["juros_recebidos"]
    juros_pagos = distribuicao["juros_pagos"]
    soma_juros_liquidos = float((juros_recebidos - juros_pagos).sum())
    if abs(soma_juros_liquidos) > 1e-4:
        raise RuntimeError(
            "Os juros recebidos e pagos não fecham: "
            f"resíduo = {soma_juros_liquidos}."
        )
    for nome, (entrada, saida) in COLUNAS_SETORES.items():
        cei.iloc[L["juros"], entrada] = float(juros_recebidos.loc[nome])
        cei.iloc[L["juros"], saida] = float(juros_pagos.loc[nome])

    cei.iloc[L["va"], C["ff_e"]] = (
        distribuicao["va_planejado_ff"]
        + distribuicao["impostos_produtos_ff"]
    )
    cei.iloc[L["va"], C["nf_e"]] = (
        distribuicao["va_planejado_nf"]
        + distribuicao["impostos_produtos_nf"]
    )
    cei.iloc[L["va"], C["externo_e"]] = float(
        fluxos_transitorios["importacoes"].sum()
    )
    cei.iloc[L["va"], C["externo_s"]] = float(
        fluxos_transitorios["exportacoes"].sum()
    )

    cei.iloc[L["impostos_produtos"], C["ff_s"]] = distribuicao[
        "impostos_produtos_ff"
    ]
    cei.iloc[L["impostos_produtos"], C["nf_s"]] = distribuicao[
        "impostos_produtos_nf"
    ]
    cei.iloc[L["impostos_produtos"], C["governo_e"]] = (
        distribuicao["impostos_produtos_ff"]
        + distribuicao["impostos_produtos_nf"]
    )
    cei.iloc[L["salarios"], C["ff_s"]] = distribuicao["salarios_ff"]
    cei.iloc[L["salarios"], C["nf_s"]] = distribuicao["salarios_nf"]
    cei.iloc[L["salarios"], C["familias_e"]] = (
        distribuicao["salarios_ff"] + distribuicao["salarios_nf"]
    )
    cei.iloc[L["contribuicoes_efetivas"], C["ff_s"]] = distribuicao[
        "contribuicoes_efetivas_ff"
    ]
    cei.iloc[L["contribuicoes_efetivas"], C["nf_s"]] = distribuicao[
        "contribuicoes_efetivas_nf"
    ]
    cei.iloc[L["contribuicoes_efetivas"], C["familias_e"]] = (
        distribuicao["contribuicoes_efetivas_ff"]
        + distribuicao["contribuicoes_efetivas_nf"]
    )
    cei.iloc[L["outros_impostos"], C["ff_s"]] = distribuicao[
        "outros_impostos_ff"
    ]
    cei.iloc[L["outros_impostos"], C["nf_s"]] = distribuicao[
        "outros_impostos_nf"
    ]
    cei.iloc[L["outros_impostos"], C["governo_e"]] = (
        distribuicao["outros_impostos_ff"] + distribuicao["outros_impostos_nf"]
    )
    cei.iloc[L["ir"], C["ff_s"]] = distribuicao["ir_ff"]
    cei.iloc[L["ir"], C["nf_s"]] = distribuicao["ir_nf"]
    cei.iloc[L["dividendos"], C["ff_s"]] = distribuicao["dividendos_ff"]
    cei.iloc[L["dividendos"], C["nf_s"]] = distribuicao["dividendos_nf"]
    cei.iloc[L["dividendos"], C["familias_e"]] = distribuicao[
        "dividendos_familias"
    ]
    cei.iloc[L["dividendos"], C["externo_e"]] = distribuicao[
        "dividendos_exterior"
    ]
    cei.iloc[L["ir"], C["familias_s"]] = distribuicao["ir_familias"]
    cei.iloc[L["ir"], C["governo_e"]] = (
        distribuicao["ir_familias"] + distribuicao["ir_ff"] + distribuicao["ir_nf"]
    )
    cei.iloc[L["beneficios"], C["familias_e"]] = distribuicao["beneficios"]
    cei.iloc[L["beneficios"], C["governo_s"]] = distribuicao["beneficios"]
    cei.iloc[L["aposentadorias"], C["familias_e"]] = distribuicao[
        "aposentadorias"
    ]
    cei.iloc[L["aposentadorias"], C["governo_s"]] = distribuicao[
        "aposentadorias_governo"
    ]
    cei.iloc[L["aposentadorias"], C["ff_s"]] = distribuicao[
        "aposentadorias_ff"
    ]
    cei.iloc[L["consumo"], C["familias_s"]] = distribuicao["consumo_cei"]
    cei.iloc[L["consumo"], C["governo_s"]] = float(
        fluxos_transitorios["consumo_governo"].sum()
    )
    cei.iloc[L["contribuicoes_sociais"], C["familias_s"]] = (
        distribuicao["previdencia_familias"]
    )
    cei.iloc[L["contribuicoes_sociais"], C["governo_e"]] = (
        distribuicao["previdencia_publica"]
    )
    cei.iloc[L["contribuicoes_sociais"], C["ff_e"]] = (
        distribuicao["previdencia_privada"]
    )

    return {
        "cei": cei,
        "consumo_nominal": float(distribuicao["consumo_cei"]),
        "consumo_familias": float(distribuicao["consumo_cei"]),
        "renda_disponivel": float(distribuicao["renda_disponivel_familias"]),
        "poupanca_familias": float(distribuicao["poupanca_familias"]),
        "emprego": distribuicao["emprego"],
    }

def montar_cei_legado(
    distribuicao: dict,
    parametros: dict,
    fluxos_transitorios: dict,
) -> dict:
    """Insere investimento legado, fecha B.9 e valida a matriz CEI."""

    cei = distribuicao["cei"]
    fbcf_familias = parametros["fbcf_familias"]
    fbcf_nf = parametros["fbcf_nf"]

    # A FBCF das famílias foi decidida no início do período com base na
    # poupança observada no período anterior. Portanto, ela permanece fixa
    # durante as substituições do consumo corrente.
    cei.iloc[L["fbcf"], C["familias_s"]] = fbcf_familias
    cei.iloc[L["estoques"], C["familias_s"]] = 0.0

    # A FBCF das firmas não financeiras vem da equação do estoque de capital
    # calculada no início do período. Como a CEI é nominal, recebe aqui a soma
    # já valorizada pelos preços dos bens de capital.
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
    investimento_total_tru = float(fluxos_transitorios["investimento_total"])
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
        "consumo_nominal": distribuicao["consumo_nominal"],
        "consumo_familias": distribuicao["consumo_familias"],
        "renda_disponivel": distribuicao["renda_disponivel"],
        "poupanca_familias": distribuicao["poupanca_familias"],
        "fbcf_familias": float(fbcf_familias),
        "fbcf_nf": float(fbcf_nf),
        "capacidade_financiamento": capacidade,
        "emprego": distribuicao["emprego"],
    }

def simul_CEI(
    parametros: dict,
    distribuicao_pre_mercado: dict,
    fluxos_transitorios: dict,
    CEI_inicial: pd.DataFrame,
    inflation_index: float,
) -> dict:
    """Registra a distribuição pré-mercado e fecha temporariamente a CEI."""

    distribuicao_registrada = registrar_distribuicao_pre_mercado_na_cei(
        distribuicao_pre_mercado,
        fluxos_transitorios,
        CEI_inicial,
        inflation_index,
    )
    return montar_cei_legado(
        distribuicao_registrada, parametros, fluxos_transitorios
    )

def simul_(
    periodos: int,
    condicoes_iniciais: dict,
    config_abm: dict,
    executar_bloco_legado: bool = True,
) -> dict:
    """Executa o ciclo ABM, preservando temporariamente a TRU/CEI legadas."""

    ci = condicoes_iniciais
    cfg = ci["config"]
    setores = list(ci["setores"])
    velocidade_ajuste_estoques_firmas = float(config_abm.get("velocidade_ajuste_estoques_firmas", cfg["velocidade_ajuste_estoques"]))
    if velocidade_ajuste_estoques_firmas < 0.0:
        raise ValueError("velocidade_ajuste_estoques_firmas não pode ser negativa.")
    lambda_expectativa_precos = float(
        config_abm.get("lambda_expectativa_precos", 1.0)
    )
    if lambda_expectativa_precos < 0.0:
        raise ValueError("lambda_expectativa_precos não pode ser negativo.")
    gamma_investimento_retorno = float(
        config_abm.get("gamma_investimento_retorno", 0.5)
    )
    if gamma_investimento_retorno < 0.0:
        raise ValueError("gamma_investimento_retorno não pode ser negativo.")
    gamma_investimento_capacidade = float(
        config_abm.get(
            "gamma_investimento_capacidade",
            gamma_investimento_retorno,
        )
    )
    if gamma_investimento_capacidade < 0.0:
        raise ValueError("gamma_investimento_capacidade não pode ser negativo.")

    # Abertura única antes do FOR temporal: os mesmos objetos sobreviverão a
    # todos os períodos. Produção e preço são definidos pelos objetos Firma.
    calibracao_investimento_nf_abm = calibrar_investimento_nf_abm(ci)
    firmas = inicializar_firmas(
        ci,
        config_abm,
        calibracao_investimento_nf_abm=calibracao_investimento_nf_abm,
    )
    multiplicador_capacidade_importada = float(
        config_abm.get("multiplicador_capacidade_importada", 1.5)
    )
    if multiplicador_capacidade_importada < 0.0:
        raise ValueError("multiplicador_capacidade_importada não pode ser negativo.")
    importados = inicializar_importados(
        ci, firmas, multiplicador_capacidade_importada
    )
    ids_firmas = tuple(id(firma) for firma in firmas.values())
    p = ci["parametros_cei"]
    periodo_choque = int(cfg["periodo_choque"])
    choque_permanente = cfg["choque_permanente"]
    if not isinstance(choque_permanente, bool):
        raise TypeError("choque_permanente deve ser True ou False.")
    if periodo_choque < 1 or periodo_choque > periodos:
        raise ValueError(
            "periodo_choque deve estar entre 1 e o número de períodos."
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
    consumo_nominal_base = float(
        ci["valores_cei"].iat[L["consumo"], C["familias_s"]]
    )
    pesos_consumo = ci["consumo_base"].div(ci["consumo_base"].sum())

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

    calibracao_nf_legada = ci["investimento_nf"]
    investimento_nf_base = (
        calibracao_investimento_nf_abm["pesos_bens_capital_nf"]
        * calibracao_investimento_nf_abm["fbcf_nf_total_pb"]
    ).rename("investimento_nf_real")
    investimento_nf_base_pm = calibracao_investimento_nf_abm[
        "fbcf_nf_pm"
    ].copy()

    # Componentes fixos do investimento. Eles são mantidos separados para que
    # a TRU seja construída pelas decisões institucionais, e não para que a CEI
    # receba posteriormente um resíduo do investimento total.

    fbcf_fixa_base = (
        ci["tru_base"].gross_investment_sector.iloc[:, 0].reindex(setores)
        - investimento_familias_base
        - calibracao_investimento_nf_abm["fbcf_nf_pm"]
    ).rename("fbcf_fixa_base")
    estoques_base = (
        ci["tru_base"].stocks_investment_sector.iloc[:, 0]
        .reindex(setores)
        .rename("estoques_base")
    )

    investimento_fixo_base = fbcf_fixa_base + estoques_base
    if not np.allclose(
        investimento_familias_base
        + investimento_nf_base_pm
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

    # O acelerador usa exclusivamente a calibração ABM em preços básicos.
    # ``beta`` legado permanece apenas no bloco de estoques ainda não migrado.
    beta_investimento_nf = float(calibracao_nf_legada["beta"])
    v_investimento_nf = float(calibracao_investimento_nf_abm["v"])
    depreciacao_capital_nf = float(
        calibracao_investimento_nf_abm["depreciacao"]
    )
    setores_nf = calibracao_investimento_nf_abm["setores_nf"]
    inicializar_taxas_retorno_firmas(
        firmas=firmas,
        setores_nf=setores_nf,
        depreciacao=depreciacao_capital_nf,
        taxa_juros_real=float(cfg["taxa_juros_real"]),
    )
    producao_nf_corrente = calibracao_investimento_nf_abm[
        "producao_anterior"
    ].copy()
    producao_nf_anterior = producao_nf_corrente.copy()
    investimento_nf_base_por_investidor = calibracao_investimento_nf_abm[
        "investimento_bruto_base"
    ].copy()
    inicializacao_investimento_nf = "abm_estacionaria"
    estoque_capital_nf = calibracao_investimento_nf_abm[
        "estoque_capital_inicial"
    ].copy()
    investimento_liquido_nf_base = calibracao_investimento_nf_abm[
        "investimento_liquido_base"
    ].copy()
    investimento_reposicao_nf_base = calibracao_investimento_nf_abm[
        "investimento_reposicao_base"
    ].copy()
    pesos_bens_capital_nf = calibracao_investimento_nf_abm[
        "pesos_bens_capital_nf"
    ].copy()

    # Um zero na TRU-base é tratado como zero estrutural: esse setor nunca
    # forma estoques. Valores positivos e negativos identificam os setores que
    # participam da dinâmica.
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
        calibracao_nf_legada["producao_real"]
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
    estoque_ciclico = pd.Series(0.0, index=ci["setores"], name="estoque_ciclico")
    estoque_real = (estoque_referencia + estoque_ciclico).rename("estoque_real")

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
    # ``pc_anterior_2`` é um nível de Pc observado, não uma taxa. Para que a
    # regra geral de expectativa já incorpore a inflação nominal de referência
    # no primeiro período, Pc_-1 é inicializado abaixo do Pc normalizado de t=0.
    # Logo: Pc_esperado,1 = Pc_0 * [Pc_0 / Pc_-1] = 1 * (1 + a0).
    pc_anterior_2 = pc_anterior / (1.0 + inflacao_inicial)

    # ------------------------------------------------------------------
    # Ativos e passivos financeiros por setor institucional
    # ------------------------------------------------------------------
    # Os dois estoques brutos do ano-base são inferidos separadamente:
    #   ativos_0  = juros recebidos_0 / taxa real;
    #   passivos_0 = juros pagos_0 / taxa real.
    # Assim, os juros brutos observados na CEI são reproduzidos sem perder a
    # distinção entre entrada e saída. O estoque líquido é apenas o resultado
    # ativos - passivos, e não mais o único estoque mantido pelo modelo.
    financeiro_inicial = inicializar_financeiro_abm(
        ci["valores_cei"], cfg, taxa_juros_nominal
    )
    taxa_juros_real = financeiro_inicial["taxa_juros_real"]
    fracao_reavaliacao_financeira = financeiro_inicial[
        "fracao_reavaliacao_financeira"
    ]
    juros_recebidos_base = financeiro_inicial["juros_recebidos_base"]
    juros_pagos_base = financeiro_inicial["juros_pagos_base"]
    juros_liquidos_base = financeiro_inicial["juros_liquidos_base"]
    ativos_financeiros = financeiro_inicial["ativos_financeiros"]
    passivos_financeiros = financeiro_inicial["passivos_financeiros"]
    estoque_financeiro = financeiro_inicial["estoque_financeiro"]

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
   
    historico_zero = {
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
        "consumo_nominal": consumo_nominal_base/pib_base,
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
    }
    resultados = inicializar_resultados_abm(
        {
            "firmas": firmas,
            "importados": importados,
            "inicializacao_investimento_nf": inicializacao_investimento_nf,
            "historico_zero": historico_zero,
            "pc_zero": pd.Series(1.0, index=setores, name="preco_comprador"),
            "pb_zero": pd.Series(1.0, index=setores, name="preco_basico"),
            "pm_zero": pd.Series(1.0, index=setores, name="preco_importacoes"),
            "pc_esperado_zero": pd.Series(
                1.0, index=setores, name="preco_comprador_esperado"
            ),
            "inflacao_pc_zero": pd.Series(
                0.0, index=setores, name="inflacao_pc_setorial"
            ),
            "cei_zero": cei_periodo_zero,
            "capacidade_zero": capacidade_base,
            "ativos_zero": ativos_financeiros,
            "passivos_zero": passivos_financeiros,
            "estoque_financeiro_zero": estoque_financeiro,
            "aquisicao_ativos_zero": pd.Series(
                0.0,
                index=list(COLUNAS_SETORES),
                name="aquisicao_ativos_financeiros",
            ),
            "emissao_passivos_zero": pd.Series(
                0.0,
                index=list(COLUNAS_SETORES),
                name="emissao_passivos_financeiros",
            ),
            "juros_liquidos_zero": juros_liquidos_base,
            "juros_recebidos_zero": juros_recebidos_base,
            "juros_pagos_zero": juros_pagos_base,
            "reavaliacao_zero": pd.Series(
                0.0,
                index=list(COLUNAS_SETORES),
                name="reavaliacao_financeira",
            ),
            "investimento_nf_real_zero": investimento_nf_base,
            "investimento_nf_nominal_zero": investimento_nf_base,
            "fbcf_fixa_zero": fbcf_fixa_base,
            "estoques_zero": estoques_base,
            "estoques_ciclicos_zero": pd.Series(0.0, index=setores),
            "estoque_real_zero": estoque_real,
            "estoque_referencia_zero": estoque_referencia,
            "estoque_ciclico_zero": estoque_ciclico,
            "investimento_nf_investidor_zero": investimento_nf_base_por_investidor,
            "capital_nf_zero": estoque_capital_nf,
        }
    )
    historico = resultados["historico"]
    ids_firmas_periodos = resultados["ids_firmas_por_periodo"]
    agregados_firmas_periodos = resultados["agregados_firmas"]
    dados_firmas_cei_periodos = resultados["dados_firmas_cei"]
    mercados_industriais_periodos = resultados["mercados_industriais"]
    mercados_leilao_periodos = resultados["mercados_leilao"]
    diagnostico_importacoes_periodos = resultados["diagnostico_importacoes"]
    pc_periodos = resultados["precos_comprador"]
    pb_periodos = resultados["precos_basicos"]
    pm_periodos = resultados["precos_importacoes"]
    pc_esperado_periodos = resultados["precos_comprador_esperados"]
    inflacao_pc_setorial_periodos = resultados["inflacao_precos_setorial"]
    cei_periodos = resultados["cei"]
    capacidades = resultados["capacidade_financiamento"]
    ativos_financeiros_periodos = resultados["ativos_financeiros"]
    passivos_financeiros_periodos = resultados["passivos_financeiros"]
    estoque_financeiro_periodos = resultados["estoque_financeiro"]
    aquisicao_ativos_periodos = resultados["aquisicao_ativos_financeiros"]
    emissao_passivos_periodos = resultados["emissao_passivos_financeiros"]
    juros_liquidos_periodos = resultados["juros_liquidos"]
    juros_recebidos_periodos = resultados["juros_recebidos"]
    juros_pagos_periodos = resultados["juros_pagos"]
    reavaliacao_financeira_periodos = resultados["reavaliacao_financeira"]
    investimento_nf_real_periodos = resultados["investimento_nf_real"]
    investimento_nf_nominal_periodos = resultados["investimento_nf_nominal"]
    fbcf_fixa_nominal_periodos = resultados["fbcf_fixa_nominal"]
    variacao_estoques_real_periodos = resultados["variacao_estoques_real"]
    variacao_estoques_nominal_periodos = resultados["variacao_estoques_nominal"]
    variacao_autonoma_estoques_real_periodos = resultados[
        "variacao_autonoma_estoques_real"
    ]
    variacao_ciclica_estoques_real_periodos = resultados[
        "variacao_ciclica_estoques_real"
    ]
    estoque_real_periodos = resultados["estoque_real"]
    estoque_referencia_periodos = resultados["estoque_referencia_real"]
    estoque_ciclico_periodos = resultados["estoque_ciclico_real"]
    investimento_nf_por_investidor_periodos = resultados[
        "investimento_nf_por_setor_investidor"
    ]
    estoque_capital_nf_periodos = resultados["estoque_capital_nf_real"]

    # ==================================================================
    # FOR TEMPORAL
    # ==================================================================

    for t in range(1, periodos + 1):

        # ==============================================================
        # DECISÃO EX ANTE DAS FIRMAS
        # ==============================================================
        # Esta é a primeira decisão produtiva do período. Nada que venha da
        # demanda corrente, da TRU legada ou da CEI abaixo pode alterá-la em t.
        for firma in firmas.values():
            firma.decidir_producao(
                config_abm["parametro_estoque_desejado"],
                velocidade_ajuste_estoques_firmas,
            )
            firma.calcular_demanda_intermediaria()
            firma.calcular_demanda_trabalho()
            if firma.setor in setores_nf:
                firma.decidir_investimento(
                    v=v_investimento_nf,
                    depreciacao=depreciacao_capital_nf,
                    gamma_retorno=gamma_investimento_retorno,
                    gamma_investimento_capacidade=(
                        gamma_investimento_capacidade
                    ),
                )

        demanda_intermediaria_total = pd.Series(
            0.0,
            index=setores,
            name="demanda_intermediaria_real"
        )

        for firma in firmas.values():
            demanda_intermediaria_total += firma.demanda_intermediaria_real


        firmas["A_001"].calcular_demanda_intermediaria()

        # Pc esperado usa apenas informação herdada; Pb e Pm são separados.
        precos_ex_ante = calcular_precos_ex_ante(
            firmas,
            importados,
            setores,
            pc_anterior,
            pc_anterior_2,
            lambda_expectativa_precos,
            indice_salarios,
            indice_cambio,
            ci["G"],
            ci["Sd"],
            ci["Sm"],
        )

        pc_esperado = precos_ex_ante["pc_esperado"]
        pb = precos_ex_ante["pb"]
        pm = precos_ex_ante["pm"]
        pc = precos_ex_ante["pc"]
        inflacao_pc_setorial = precos_ex_ante["inflacao_pc_setorial"]

        agregados_firmas_periodos[t] = agregar_firmas(firmas, setores)

        dados_firmas_cei_periodos[t] = separar_agregados_firmas_cei(
            agregados_firmas_periodos[t],
            setores,
            ci["razoes_va"],
            cfg["setor_financeiro"],
        )

        # A CEI desta etapa continua usando VA e remunerações planejados,
        # calculados antes de os mercados alterarem o estado das firmas.
        dados_firmas_cei_pre_mercado = dados_firmas_cei_periodos[t]
        ids_firmas_periodos[t] = tuple(id(firma) for firma in firmas.values())
        if ids_firmas_periodos[t] != ids_firmas:
            raise RuntimeError("As firmas foram recriadas durante o ciclo temporal.")

        # Opção exclusiva para teste isolado da Etapa 5. A decisão ocorre no
        # mesmo FOR principal; apenas os blocos legados posteriores são pulados.
        if not executar_bloco_legado:
            if t == periodos:
                return {
                    "firmas": firmas,
                    "ids_firmas_por_periodo": ids_firmas_periodos,
                    "agregados_firmas": agregados_firmas_periodos,
                    "dados_firmas_cei": dados_firmas_cei_periodos,
                }
            continue

        # A FBCF familiar do período t depende da poupança observada em t-1.
        # Como a poupança passada está em valor nominal, primeiro retiramos o
        # preço da Construção de t-1. Isso preserva a quantidade real que as
        # famílias conseguem comprar. Depois de calcular os preços de t, essa
        # quantidade será valorizada pelo preço corrente da Construção.
        investimento_familias = calcular_fbcf_familias(
            poupanca_familias_anterior,
            pc_anterior,
            pc,
            setor_construcao,
            p["prop_invest_fbcf_familias"],
        )
        fbcf_familias_real = investimento_familias["fbcf_familias_real"]
        estoques = calcular_estoques_legado_periodo(producao_estoques_corrente, producao_estoques_anterior, producao_estoques_base, variacao_autonoma_estoques, estoque_referencia, estoque_ciclico, setores_com_estoques, beta_investimento_nf, razao_estoque_producao, velocidade_ajuste_estoques)
        variacao_autonoma_estoques_periodo = estoques["variacao_autonoma"]
        variacao_ciclica_estoques = estoques["variacao_ciclica"]
        variacao_estoques_real = estoques["variacao_real"]
        estoque_referencia_periodo = estoques["estoque_referencia"]
        estoque_ciclico_periodo = estoques["estoque_ciclico"]
        estoque_real_periodo = estoques["estoque_real"]

        # As decisões vêm das firmas investidoras; os pesos apenas direcionam
        # sua soma aos setores fornecedores de bens de capital.
        investimento_liquido_nf = agregados_firmas_periodos[t][
            "investimento_liquido"
        ].loc[setores_nf]
        investimento_reposicao_nf = agregados_firmas_periodos[t][
            "investimento_reposicao"
        ].loc[setores_nf]
        investimento_nf_por_investidor = agregados_firmas_periodos[t][
            "investimento_bruto"
        ].loc[setores_nf]
        investimento_nf_sem_piso = (
            investimento_liquido_nf + investimento_reposicao_nf
        ).rename("investimento_nf_sem_piso")
        investimento_nf_real_total = float(investimento_nf_por_investidor.sum())
        investimento_nf_real = (
            pesos_bens_capital_nf * investimento_nf_real_total
        ).rename("investimento_nf_real")
        estoque_capital_nf_periodo = pd.Series(
            {
                setor: sum(
                    (1.0 - depreciacao_capital_nf) * firma.estoque_capital_real
                    + firma.investimento_bruto
                    for firma in firmas.values()
                    if firma.setor == setor
                )
                for setor in setores_nf
            },
            name="estoque_capital_nf",
            dtype=float,
        )

        # A FBCF real decidida a partir da poupança de t-1 é paga ao preço da
        # Construção vigente em t. Assim, inflação não reduz artificialmente o
        # investimento real das famílias.
        fbcf_familias = investimento_familias["fbcf_familias_nominal"]

        # A demanda NF já está em PB: não passa outra vez pela conversão
        # PM -> PB aplicada aos demais componentes da demanda de investimento.
        investimento_nf_pb_nominal = (
            investimento_nf_real * pb
        ).rename("investimento_nf_pb_nominal")
        investimento_nf_nominal = (
            investimento_nf_real * pc
        ).rename("investimento_nf_nominal")

        fbcf_nf_nominal = float(investimento_nf_nominal.sum())

        autonomas = atualizar_demandas_autonomas(
            governo_nominal_anterior,
            fbcf_fixa_nominal_anterior,
            exportacoes_nominais_anterior,
            pc,
            pc_anterior,
            t,
            periodo_choque,
            choque_permanente,
            cfg["multiplicador_governo"],
            cfg["multiplicador_investimento"],
            cfg["multiplicador_exportacoes"],
        )
        governo_nominal = autonomas["governo_nominal"]
        fbcf_fixa_nominal = autonomas["fbcf_fixa_nominal"]
        exportacoes_nominais = autonomas["exportacoes_nominais"]

        investimento_cei = montar_investimento_e_cei_legado(
            fbcf_fixa_nominal,
            variacao_estoques_real,
            fbcf_familias,
            pesos_investimento_familias,
            investimento_nf_nominal,
            pc,
            fbcf_fixa_base,
            fbcf_fixa_cei_base,
        )
        variacao_estoques_nominal = investimento_cei["variacao_estoques_nominal"]
        investimento_nominal = investimento_cei["investimento_nominal"]
        fbcf_fixa_total = investimento_cei["fbcf_fixa_total"]
        investimentos_fixos_cei = investimento_cei["investimentos_fixos_cei"]

        inflacao_periodo = calcular_inflacao_periodo(
            ci["consumo_base"], pc, indice_precos_anterior
        )
        indice_precos = inflacao_periodo["indice_precos"]
        # O índice da CEI permanece o índice pré-mercado de A1. O mercado pode
        # substituir pc por preço realizado, mas não este insumo institucional.
        indice_precos_pre_mercado = indice_precos
        inflacao = inflacao_periodo["inflacao"]
        juros = calcular_juros_periodo(
            ativos_financeiros,
            passivos_financeiros,
            indice_precos,
            indice_precos_anterior,
            taxa_juros_real,
            taxa_juros_nominal,
            inertia_pm,
            t,
        )
        ativos_financeiros_corrigidos = juros["ativos_corrigidos"]
        passivos_financeiros_corrigidos = juros["passivos_corrigidos"]
        taxa_juros_nominal = juros["taxa_juros_nominal"]
        juros_recebidos = juros["juros_recebidos"]
        juros_pagos = juros["juros_pagos"]
        juros_liquidos = juros["juros_liquidos"]

        parametros_cei_pre_mercado = {
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

        # Enquanto os mercados ainda não foram ligados à contabilidade, estes
        # dois fluxos usam a trajetória nominal calibrada na TRU-base.
        fluxos_transitorios_pre_mercado = {
            "impostos_produtos": (
                ci["tru_base"].taxes_sector.iloc[:, 0].reindex(setores)
                * indice_precos_pre_mercado
            ),
            "importacoes": (
                ci["tru_base"].imports_sector.iloc[:, 0].reindex(setores)
                * indice_precos_pre_mercado
            ),
            "exportacoes": exportacoes_nominais.copy(),
            "consumo_governo": governo_nominal.copy(),
            "investimento_total": float(investimento_nominal.sum()),
        }
        fluxos_template_distribuicao = extrair_fluxos_template_distribuicao(
            ci["valores_cei"], indice_precos_pre_mercado
        )
        distribuicao_pre_mercado = calcular_distribuicao_pre_mercado(
            p,
            dados_firmas_cei_pre_mercado,
            fluxos_transitorios_pre_mercado["impostos_produtos"],
            juros_recebidos,
            juros_pagos,
            indice_salarios,
            indice_precos_pre_mercado,
            cfg["setor_financeiro"],
            fluxos_template_distribuicao,
        )
        consumo_cei = distribuicao_pre_mercado["consumo_cei"]
        residuo_consumo = 0.0
        iteracao = 0

        # ==============================================================
        # MERCADO INDUSTRIAL: ESCOLHA, OFERTA E VENDAS
        # ==============================================================
        # A produção já foi decidida e não é alterada por este bloco. A demanda
        # final nominal vem da CEI; a demanda intermediária é quantidade gerada
        # pela tecnologia das firmas, valorizada pelo Pc corrente.

        demandas_periodo = montar_demandas_periodo(
            demanda_intermediaria_total,
            consumo_cei,
            pesos_consumo,
            governo_nominal,
            investimento_nominal - investimento_nf_nominal,
            exportacoes_nominais,
            pc,
            ci["conversao_de_pm_pb"],
            investimento_nf_pb_nominal=investimento_nf_pb_nominal,
        )
        demanda_intermediaria_real = demandas_periodo[
            "demanda_intermediaria_real"
        ]
        demanda_total_pm_nominal = demandas_periodo[
            "demanda_total_pm_nominal"
        ]
        demanda_total_pb_nominal = demandas_periodo[
            "demanda_total_pb_nominal"
        ]

        mercados = executar_mercados_periodo(
            setores,
            firmas,
            importados,
            demanda_total_pm_nominal,
            demanda_total_pb_nominal,
            demanda_intermediaria_real,
            pb,
        )
        pb = mercados["pb"]
        leiloes_setoriais = mercados["setores_leilao"]
        participantes_mercado = mercados["participantes_industriais"]
        participantes_leilao = mercados["participantes_leilao"]
        diagnostico_importacoes_periodos[t] = mercados["diagnostico_importacoes"]
        atendimento_categorial = ratear_atendimento_proporcional(
            demandas_periodo["demandas_pb_nominal"],
            participantes_mercado,
            participantes_leilao,
            setores,
        )
        erro_desejado = float((atendimento_categorial["desejado_pb_nominal"].sum(axis=1) - demanda_total_pb_nominal).abs().max())
        if erro_desejado > 1e-6:
            raise RuntimeError(f"Demanda categorial não conserva total: {erro_desejado}.")
        ci_nao_atendido = atendimento_categorial["nao_atendido_pb_nominal"]["ci"]
        if float(ci_nao_atendido.abs().max()) > 1e-6:
            raise RuntimeError("CI não atendido material; B1 não pode prosseguir.")

        # O preço efetivo do leilão é o Pb do setor. O Pc observado será
        # herdado pela formação de custos do período seguinte; a CEI continua
        # recebendo separadamente o índice pré-mercado de A1.

        if leiloes_setoriais:
            precos_realizados = calcular_precos_realizados(
                pb,
                pm,
                ci["G"],
                ci["Sd"],
                ci["Sm"],
                pc_anterior,
            )
            pb = precos_realizados["pb"]
            pm = precos_realizados["pm"]
            pc = precos_realizados["pc"]
            for firma in firmas.values():
                firma.registrar_custo_intermediario_realizado(pc)
            inflacao_pc_setorial = precos_realizados[
                "inflacao_pc_setorial"
            ]

        agregados_firmas_periodos[t] = agregar_firmas(firmas, setores)
        mercados_industriais_periodos[t] = mercados["registro_industrial"]
        mercados_leilao_periodos[t] = mercados["registro_leilao"]

        # CEI pós-mercado apenas no timing: sua distribuição, VA, impostos,
        # importações e índice continuam sendo exatamente as fontes pré-mercado
        # de A1. Fontes realizadas serão tratadas somente nas etapas A3--A6.
        results_CEI = simul_CEI(
            parametros_cei_pre_mercado,
            distribuicao_pre_mercado,
            fluxos_transitorios_pre_mercado,
            ci["cei_original"],
            indice_precos_pre_mercado,
        )
      

        # Emprego corrente determina o salário usado no próximo período.
        mercado_trabalho = calcular_mercado_trabalho(
            results_CEI["emprego"],
            p["pea"],
            cfg["taxa_desemprego_base"],
            cfg["a0"],
            cfg["a1"],
            cfg["a3"],
        )
        emprego = mercado_trabalho["emprego"]
        taxa_desemprego = mercado_trabalho["taxa_desemprego"]
        variacao_salarios = mercado_trabalho["variacao_salarios"]

        dados_setoriais_firmas = dados_firmas_cei_pre_mercado["setorial"]

        capacidade = results_CEI["capacidade_financiamento"]

        financeiro_periodo = atualizar_financeiro_periodo(
            capacidade,
            fracao_reavaliacao_financeira,
            ativos_financeiros_corrigidos,
            passivos_financeiros_corrigidos,
        )
        reavaliacao_financeira = financeiro_periodo["reavaliacao_financeira"]
        aquisicao_ativos = financeiro_periodo["aquisicao_ativos"]
        emissao_passivos = financeiro_periodo["emissao_passivos"]
        ativos_financeiros_periodo = financeiro_periodo[
            "ativos_financeiros_periodo"
        ]
        passivos_financeiros_periodo = financeiro_periodo[
            "passivos_financeiros_periodo"
        ]
        estoque_financeiro_periodo = financeiro_periodo[
            "estoque_financeiro_periodo"
        ]

        pib_periodo = calcular_pib_legado(
            dados_setoriais_firmas,
            pb,
            pc,
            fluxos_transitorios_pre_mercado["impostos_produtos"],
        )
        pib_nominal = pib_periodo["pib_nominal"]
        pib_real = pib_periodo["pib_real"]

        registro_historico = montar_registro_historico(
                {
                    "periodo": t,
                    "ano": cfg["ano"] + t,
                    "indice_precos": indice_precos,
                    "inflacao": inflacao,
                    "indice_salarios": indice_salarios,
                    "indice_cambio": indice_cambio,
                    "taxa_juros_nominal": taxa_juros_nominal,
                    "pib_real": pib_real,
                    "pib_nominal": pib_nominal,
                    "emprego": emprego,
                    "taxa_desemprego": taxa_desemprego,
                    "consumo_cei": consumo_cei,
                    "poupanca_familias": results_CEI["poupanca_familias"],
                    "fbcf_familias": fbcf_familias,
                    "investimento_nf_real_total": investimento_nf_real_total,
                    "fbcf_nf_nominal": fbcf_nf_nominal,
                    "fbcf_fixa_total": fbcf_fixa_total,
                    "variacao_estoques_real": variacao_estoques_real,
                    "variacao_autonoma_estoques_periodo": variacao_autonoma_estoques_periodo,
                    "variacao_ciclica_estoques": variacao_ciclica_estoques,
                    "estoque_real_periodo": estoque_real_periodo,
                    "investimento_liquido_nf": investimento_liquido_nf,
                    "investimento_reposicao_nf": investimento_reposicao_nf,
                    "investimento_nf_por_investidor": investimento_nf_por_investidor,
                    "investimento_nf_sem_piso": investimento_nf_sem_piso,
                    "estoque_capital_nf_periodo": estoque_capital_nf_periodo,
                    "residuo_consumo": residuo_consumo,
                    "iteracao": iteracao,
                    "capacidade": capacidade,
                }
        )
        armazenar_resultados_periodo(
            resultados,
            t,
            {
                "agregados_firmas": agregados_firmas_periodos[t],
                "dados_firmas_cei": dados_firmas_cei_periodos[t],
                "ids_firmas": ids_firmas_periodos[t],
                "fluxos_transitorios": fluxos_transitorios_pre_mercado,
                "mercado_industrial": mercados_industriais_periodos[t],
                "mercado_leilao": mercados_leilao_periodos[t],
                "diagnostico_importacoes": diagnostico_importacoes_periodos[t],
                "historico": registro_historico,
                "pc": pc, "pb": pb, "pm": pm, "pc_esperado": pc_esperado,
                "inflacao_pc_setorial": inflacao_pc_setorial,
                "cei": results_CEI["cei"], "capacidade": capacidade,
                "ativos_financeiros": ativos_financeiros_periodo,
                "passivos_financeiros": passivos_financeiros_periodo,
                "estoque_financeiro": estoque_financeiro_periodo,
                "aquisicao_ativos": aquisicao_ativos,
                "emissao_passivos": emissao_passivos,
                "juros_liquidos": juros_liquidos,
                "juros_recebidos": juros_recebidos,
                "juros_pagos": juros_pagos,
                "reavaliacao_financeira": reavaliacao_financeira,
                "investimento_nf_real": investimento_nf_real,
                "investimento_nf_nominal": investimento_nf_nominal,
                "fbcf_fixa_nominal": fbcf_fixa_nominal,
                "variacao_estoques_real": variacao_estoques_real,
                "variacao_estoques_nominal": variacao_estoques_nominal,
                "variacao_autonoma_estoques": variacao_autonoma_estoques_periodo,
                "variacao_ciclica_estoques": variacao_ciclica_estoques,
                "estoque_real": estoque_real_periodo,
                "estoque_referencia": estoque_referencia_periodo,
                "estoque_ciclico": estoque_ciclico_periodo,
                "investimento_nf_investidor": investimento_nf_por_investidor,
                "estoque_capital_nf": estoque_capital_nf_periodo,
            },
        )
        

        for firma in firmas.values():
            firma.atualizar_estado(depreciacao_capital_nf)

        estado_proximo = atualizar_estado_periodo(
            {
                "indice_salarios": indice_salarios,
                "variacao_salarios": variacao_salarios,
                "indice_cambio": indice_cambio,
                "repasse_inflacao_cambio": cfg["repasse_inflacao_cambio"],
                "inflacao": inflacao,
                "indice_precos": indice_precos,
                "pc_anterior": pc_anterior,
                "pc": pc,
                "governo_nominal": governo_nominal,
                "fbcf_fixa_nominal": fbcf_fixa_nominal,
                "exportacoes_nominais": exportacoes_nominais,
                "poupanca_familias": results_CEI["poupanca_familias"],
                "estoque_capital_nf_periodo": estoque_capital_nf_periodo,
                "ativos_financeiros_periodo": ativos_financeiros_periodo,
                "passivos_financeiros_periodo": passivos_financeiros_periodo,
                "estoque_financeiro_periodo": estoque_financeiro_periodo,
                "estoque_real_periodo": estoque_real_periodo,
                "estoque_referencia_periodo": estoque_referencia_periodo,
                "estoque_ciclico_periodo": estoque_ciclico_periodo,
                "producao_nf_corrente": producao_nf_corrente,
                "dados_setoriais_firmas": dados_setoriais_firmas,
                "setores_nf": setores_nf,
                "producao_estoques_corrente": producao_estoques_corrente,
            }
        )
        indice_salarios = estado_proximo["indice_salarios"]
        indice_cambio = estado_proximo["indice_cambio"]
        indice_precos_anterior = estado_proximo["indice_precos_anterior"]
        pc_anterior_2 = estado_proximo["pc_anterior_2"]
        pc_anterior = estado_proximo["pc_anterior"]
        governo_nominal_anterior = estado_proximo["governo_nominal_anterior"]
        fbcf_fixa_nominal_anterior = estado_proximo["fbcf_fixa_nominal_anterior"]
        exportacoes_nominais_anterior = estado_proximo[
            "exportacoes_nominais_anterior"
        ]
        poupanca_familias_anterior = estado_proximo["poupanca_familias_anterior"]
        estoque_capital_nf = estado_proximo["estoque_capital_nf"]
        ativos_financeiros = estado_proximo["ativos_financeiros"]
        passivos_financeiros = estado_proximo["passivos_financeiros"]
        estoque_financeiro = estado_proximo["estoque_financeiro"]
        estoque_real = estado_proximo["estoque_real"]
        estoque_referencia = estado_proximo["estoque_referencia"]
        estoque_ciclico = estado_proximo["estoque_ciclico"]
        producao_nf_anterior = estado_proximo["producao_nf_anterior"]
        producao_nf_corrente = estado_proximo["producao_nf_corrente"]
        producao_estoques_anterior = estado_proximo["producao_estoques_anterior"]
        producao_estoques_corrente = estado_proximo["producao_estoques_corrente"]
