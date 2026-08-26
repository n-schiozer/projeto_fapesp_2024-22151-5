"""Laboratório SFC--IO--ABM limpo e auditável.

Preserva o ciclo econômico da referência e remove somente caminhos
legados, testes manuais, gráficos e diagnósticos temporários. As unidades
são explícitas: quantidade real, preço básico (PB) e preço de comprador
(PM/Pc) não são intercambiados.
"""

from copy import deepcopy

from pathlib import Path

from configuracao_projeto import (
    ARQUIVO_CEI,
    DATA_DIR,
    validar_caminhos_dados,
)
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais

import matplotlib.pyplot as plt

"""Três blocos auditáveis: TRU, CEI e ciclo temporal."""

import numpy as np
import pandas as pd

import contabilidade.montar_cei_abm as montar_cei_abm

import mercados.calcular_precos_realizados_abm as calcular_precos_realizados_abm

from agentes.agregar_firmas import agregar_firmas, agregar_resultados_realizados_firmas
from macro.ciclo_abm import (
    atualizar_financeiro_periodo,
    atualizar_estado_periodo,
    calcular_inflacao_periodo,
    calcular_juros_periodo,
    calcular_mercado_trabalho,
    calcular_precos_realizados,
)
from contabilidade.distribuicao_abm import (
    calcular_distribuicao_pre_mercado_abm
)
from financeiro.financeiro_abm import inicializar_financeiro_abm
from contabilidade.estrutura_cei import (
    C,
    L,
    VA,
    COLUNAS_SETORES,
)
from inicializacao.inicializar_firmas import inicializar_firmas

from agentes.fornecedor_importado_abm import inicializar_importados_abm

from investimento.investimento_abm import (
    calcular_fbcf_familias,
    inicializar_taxas_retorno_firmas,
)


from resultados.resultados_abm_legado import (
    inicializar_resultados_abm,
)

from mercados.executar_mercados_periodo_abm import executar_mercados_periodo

from calibracao.calibracao_investimento_nf_abm import (
    calibrar_investimento_nf_abm,
)

# %% =====================================================================
# 1. ARQUIVOS DE DADOS
# ========================================================================
# A TRU de 2020 fica na pasta indicada por DATA_DIR. A CEI é a planilha que
# contém os fluxos institucionais iniciais da economia.

# %% =====================================================================
# 2. HIPÓTESES MACROECONÔMICAS
# ========================================================================
# CONFIG é lido por preparar_condicoes_iniciais() e simul_(). Ele contém as
# hipóteses de demanda autônoma, inflação, juros, mercado de trabalho, estoques
# agregados e investimento das firmas não financeiras.

CONFIG_BASE = {
    "ano": 2020,
    "nivel": 20,
    "aba_cei": "Python",
    "periodos": 50,
    "multiplicador_governo": 1.0,
    "multiplicador_investimento": 1.0,
    "multiplicador_exportacoes": 1.0,
    # Mesmo no cenário sem choque, o período precisa ser válido para simul_().
    "periodo_choque": 20,
    "choque_permanente": False,
    "taxa_desemprego_base": 0.138,
    "taxa_desemprego_inicial": 0.138,
    "parcela_ativa_populacao": 0.50,
    "parcela_aposentados_inativos": 0.50,
    "setor_financeiro": 10,
    "setores_excluidos_investimento_nf": [
        "K - Atividades financeiras, de seguros e serviços relacionados",
        "O - Administração pública, defesa e seguridade social",
        "T - Serviços domésticos",
    ],
    "vida_util_capital": 20.0,
    "ano_inicial_beta": 2010,
    "ano_final_beta": 2019,
    "inicializacao_investimento_nf": "estacionaria",
    "razao_estoque_producao": 1.0 / 12.0,
    "velocidade_ajuste_estoques": 1,
    # a0 inicializa a inflação salarial e nominal do período 1.
    "a0": 0.02,
    "a1": 0.5,
    "a3": 0.5,
    "repasse_inflacao_cambio": 1.0,
    "taxa_juros_real": 0.06,
    "inertia_pm": 0.5,
    "fracao_reavaliacao_financeira": 1.0,
    "tolerancia_consumo": 1e-6,
    "max_iteracoes_consumo": 100,
    "executar_testes": False,
}


# %% =====================================================================
# 3. FIRMAS E MERCADOS ABM
# ========================================================================
# CONFIG_ABM é exclusiva da versão com firmas. Ela define quantas firmas cada
# setor possui, quais setores usam leilão e os parâmetros de concorrência.

SETOR_FINANCEIRO = (
    "K - Atividades financeiras, de seguros e serviços relacionados"
)
SETORES_LEILAO = [
    "A - Agricultura, pecuária, produção florestal, pesca e aquicultura",
    "D - Eletricidade e gás",
]

CONFIG_ABM_BASE = {
    "numero_firmas_industria": 2,
    "numero_firmas_leilao": 2,
    # Mantém a configuração especial anterior: uma firma financeira agregada.
    "numero_firmas_por_setor": {SETOR_FINANCEIRO: 1},
    "setores_leilao": SETORES_LEILAO,
    "eta_preco_padrao": -1.2,
    "eta_qualidade_padrao": 2.0,
    "eta_atendimento_padrao": 1.0,
    "parametro_estoque_desejado": 0.0978561253333731,
    "ajustes_setoriais": {},
    "market_shares_domesticos": {},
    "precos_relativos_iniciais": {},
    # Regra K+S de markup das firmas industriais.
    "parametros_markup": {
        "parametro_markup": 0.1,
        "markup_min": 0.0,
        "markup_max": 10.0,
        "epsilon_market_share": 1e-12,
    },
    # Capacidade real máxima do importado nos setores de leilão, como múltiplo
    # da quantidade importada observada no ano-base.
    "multiplicador_capacidade_importada": 1.5,
    "velocidade_ajuste_estoques_firmas": 0.25,
    "lambda_expectativa_precos": 1.0,
}


def atualizar_configuracao(base: dict, alteracoes: dict | None) -> dict:
    """Atualiza recursivamente uma cópia profunda da configuração-base."""
    configuracao = deepcopy(base)
    if alteracoes is None:
        return configuracao
    for chave, valor in alteracoes.items():
        if isinstance(configuracao.get(chave), dict) and isinstance(valor, dict):
            configuracao[chave] = atualizar_configuracao(configuracao[chave], valor)
        else:
            configuracao[chave] = deepcopy(valor)
    return configuracao


def simular_modelo(
    nome_cenario: str = "baseline",
    config: dict | None = None,
    config_abm: dict | None = None,
    data_dir: Path = DATA_DIR,
    arquivo_cei: Path = ARQUIVO_CEI,
) -> dict:
    """Executa uma economia nova e retorna seus resultados organizados.

    Cópias independentes das configurações-base isolam os cenários. O ciclo
    interno preserva a sequência econômica do laboratório de referência.
    """
    data_dir, arquivo_cei = validar_caminhos_dados(data_dir, arquivo_cei)
    config_execucao = atualizar_configuracao(CONFIG_BASE, config)
    config_abm_execucao = atualizar_configuracao(CONFIG_ABM_BASE, config_abm)
    periodos = config_execucao["periodos"]
    config_abm = config_abm_execucao
    condicoes_iniciais = preparar_condicoes_iniciais(
        config_execucao, data_dir, arquivo_cei,
    )
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
    
    # A calibração ABM é concluída antes da criação das firmas, pois o capital
    # inicial de cada firma é repartido a partir dela.
    calibracao_investimento_nf = calibrar_investimento_nf_abm(ci)
    v_investimento_nf = calibracao_investimento_nf["v"]
    depreciacao_capital_nf = calibracao_investimento_nf["depreciacao"]
    setores_nf = calibracao_investimento_nf["setores_nf"]
    pesos_bens_capital_nf = (
        calibracao_investimento_nf["pesos_bens_capital_nf"]
        .reindex(setores)
        .fillna(0.0)
    )
    estoque_capital_nf = calibracao_investimento_nf[
        "estoque_capital_inicial"
    ].copy()
    investimento_liquido_nf_base = calibracao_investimento_nf[
        "investimento_liquido_base"
    ].copy()
    investimento_reposicao_nf_base = calibracao_investimento_nf[
        "investimento_reposicao_base"
    ].copy()
    investimento_nf_base_por_investidor = calibracao_investimento_nf[
        "investimento_bruto_base"
    ].copy()
    investimento_nf_base = (
        pesos_bens_capital_nf * calibracao_investimento_nf["fbcf_nf_total_pb"]
    ).rename("investimento_nf_base")
    
    # Abertura única antes do FOR temporal: os mesmos objetos sobreviverão a
    # todos os períodos. Produção, preço e capital são definidos pelos objetos.
    firmas = inicializar_firmas(
        ci,
        config_abm,
        calibracao_investimento_nf_abm=calibracao_investimento_nf,
    )
    inicializar_taxas_retorno_firmas(
        firmas=firmas,
        setores_nf=setores_nf,
        depreciacao=float(depreciacao_capital_nf),
        taxa_juros_real=float(cfg["taxa_juros_real"]),
    )
    
    importados = inicializar_importados_abm(
        condicoes_iniciais=ci,
        firmas=firmas,
    )
    
    
    
    
    multiplicador_capacidade_importada = float(
        config_abm.get("multiplicador_capacidade_importada", 1.5)
    )
    if multiplicador_capacidade_importada < 0.0:
        raise ValueError("multiplicador_capacidade_importada não pode ser negativo.")
    
    
    
    
    
    
    ids_firmas = tuple(id(firma) for firma in firmas.values())
    p = ci["parametros_cei"]
    
    
    
    p["parcela_impostos_produtos_ff"] = (
        float(
            ci["valores_cei"].iat[
                L["impostos_produtos"],
                C["ff_s"],
            ]
        )
        / float(
            ci["valores_cei"].iat[
                L["impostos_produtos"],
                C["governo_e"],
            ]
        )
    )
    
    
    
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
    
    ### Investimento ###
    
    # A FBCF NF é calibrada diretamente na nova função. A calibração legada fica
    # restrita ao bloco de estoques que ainda não foi migrado.
    calibracao_nf_legada = ci["investimento_nf"]
    fbcf_nf_pm = calibracao_investimento_nf["fbcf_nf_pm"].copy()
    fbcf_fixa_base = (
        ci["tru_base"].gross_investment_sector.iloc[:, 0].reindex(setores)
        - investimento_familias_base
        - fbcf_nf_pm
    ).rename("fbcf_fixa_base")
    estoques_base = (
        ci["tru_base"].stocks_investment_sector.iloc[:, 0]
        .reindex(setores)
        .rename("estoques_base")
    )
    
    investimento_fixo_base = fbcf_fixa_base + estoques_base
    if not np.allclose(
        investimento_familias_base
        + fbcf_nf_pm
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
    
    beta_investimento_nf = float(calibracao_nf_legada["beta"])
    producao_nf_corrente = calibracao_investimento_nf["producao_anterior"].copy()
    producao_nf_anterior = producao_nf_corrente.copy()
    inicializacao_investimento_nf = "abm_estacionaria"
    
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
            "investimento_nf_nominal_zero": fbcf_nf_pm.copy(),
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
    
    
    
    
    # ==========================================================
    # RECONSTRUÇÃO DA CEI 
    # ==========================================================
    
    # ==========================================================
    # FBCF INSTITUCIONAL - ANO-BASE
    # ==========================================================
    
    fbcf_cei_base = {
        "familias": float(
            fbcf_familias_base
        ),
        "governo": float(
            fbcf_fixa_cei_base["governo"]
        ),
        "firmas_financeiras": float(
            fbcf_fixa_cei_base["firmas_financeiras"]
        ),
        "firmas_nao_financeiras": float(
            fbcf_nf_pm.sum()
        ),
        "setor_externo": float(
            fbcf_fixa_cei_base["setor_externo"]
        ),
    }
    
    # ==========================================================
    # ESTOQUES INSTITUCIONAIS - ANO-BASE
    # ==========================================================
    
    estoques_cei_periodo_zero = {
        "familias": 0.0,
        "governo": float(
            estoques_cei_base["governo"]
        ),
        "firmas_financeiras": float(
            estoques_cei_base["firmas_financeiras"]
        ),
        "firmas_nao_financeiras": float(
            estoques_cei_base["firmas_nao_financeiras"]
        ),
        "setor_externo": float(
            estoques_cei_base["setor_externo"]
        ),
    }
    
    # ==========================================================
    # DADOS DAS FIRMAS PARA A CEI - ANO-BASE
    # ==========================================================
    
    setor_financeiro = setores[
        cfg["setor_financeiro"]
    ]
    
    firmas_ff = [
        firma
        for firma in firmas.values()
        if firma.setor == setor_financeiro
    ]
    
    firmas_nf = [
        firma
        for firma in firmas.values()
        if firma.setor != setor_financeiro
    ]
    
    
    dados_firmas_cei_base = {
    
        # ======================================================
        # FIRMAS FINANCEIRAS
        # ======================================================
    
        "ff": {
    
            "valor_adicionado": float(
                sum(
                    (
                        firma.remuneracao_unitaria_base
                        + firma.eob_misto_unitario_base
                        + firma.outros_va_unitario_base
                    )
                    * firma.producao_base_real
                    for firma in firmas_ff
                )
            ),
    
            "salarios": float(
                sum(
                    firma.salario_unitaria_base
                    * firma.producao_base_real
                    for firma in firmas_ff
                )
            ),
    
            "contribuicoes_efetivas": float(
                sum(
                    firma.contribuicao_unitaria_base
                    * firma.producao_base_real
                    for firma in firmas_ff
                )
            ),
    
            "dividendos": float(
                sum(
                    firma.parametro_dividendos
                    * max(
                        0.0,
                        firma.eob_misto_unitario_base
                        * firma.producao_base_real,
                    )
                    for firma in firmas_ff
                )
            ),
    
            "outros_va": float(
                sum(
                    firma.outros_va_unitario_base
                    * firma.producao_base_real
                    for firma in firmas_ff
                )
            ),
        },
    
    
        # ======================================================
        # FIRMAS NÃO FINANCEIRAS
        # ======================================================
    
        "nf": {
    
            "valor_adicionado": float(
                sum(
                    (
                        firma.remuneracao_unitaria_base
                        + firma.eob_misto_unitario_base
                        + firma.outros_va_unitario_base
                    )
                    * firma.producao_base_real
                    for firma in firmas_nf
                )
            ),
    
            "salarios": float(
                sum(
                    firma.salario_unitaria_base
                    * firma.producao_base_real
                    for firma in firmas_nf
                )
            ),
    
            "contribuicoes_efetivas": float(
                sum(
                    firma.contribuicao_unitaria_base
                    * firma.producao_base_real
                    for firma in firmas_nf
                )
            ),
    
            "dividendos": float(
                sum(
                    firma.parametro_dividendos
                    * max(
                        0.0,
                        firma.eob_misto_unitario_base
                        * firma.producao_base_real,
                    )
                    for firma in firmas_nf
                )
            ),
    
            "outros_va": float(
                sum(
                    firma.outros_va_unitario_base
                    * firma.producao_base_real
                    for firma in firmas_nf
                )
            ),
        },
    
    
        # ======================================================
        # EMPREGO
        # ======================================================
    
        "ocupacoes": float(
            sum(
                firma.ocupacoes_unitario_base
                * firma.producao_base_real
                for firma in firmas.values()
            )
        ),
    }
    
    
    # ==========================================================
    # IMPOSTOS SOBRE PRODUTOS - ANO-BASE
    # ==========================================================
    
    impostos_produtos_base = (
        ci["taxa_impostos"]
        .reindex(setores)
        .fillna(0.0)
        * ci["demanda_final_base"]
        .reindex(setores)
        .fillna(0.0)
    ).rename("impostos_produtos_base")
    
    
    # ==========================================================
    # OUTRAS TRANSFERÊNCIAS CORRENTES - ANO-BASE
    # ==========================================================
    
    outras_transferencias_base = {
    
        "familias_recebidas": float(
            ci["valores_cei"].iat[
                L["outras_transferencias"],
                C["familias_e"],
            ]
        ),
    
        "governo_recebidas": float(
            ci["valores_cei"].iat[
                L["outras_transferencias"],
                C["governo_e"],
            ]
        ),
    
        "ff_pagas": float(
            ci["valores_cei"].iat[
                L["outras_transferencias"],
                C["ff_s"],
            ]
        ),
    
        "nf_pagas": float(
            ci["valores_cei"].iat[
                L["outras_transferencias"],
                C["nf_s"],
            ]
        ),
    
        "exterior_pagas": float(
            ci["valores_cei"].iat[
                L["outras_transferencias"],
                C["externo_s"],
            ]
        ),
    }
    
    
    # ==========================================================
    # DISTRIBUIÇÃO PRÉ-MERCADO - ANO-BASE
    # ==========================================================
    
    distribuicao_cei_base = calcular_distribuicao_pre_mercado_abm(
        p=p,
        dados_firmas=dados_firmas_cei_base,
        impostos_produtos=impostos_produtos_base,
        juros_recebidos=juros_recebidos_base,
        juros_pagos=juros_pagos_base,
        indice_salarios=1.0,
        indice_precos=1.0,
        setor_financeiro=cfg["setor_financeiro"],
        outras_transferencias_base=outras_transferencias_base,
    )
    
    
    # ==========================================================
    # CEI - ANO-BASE
    # ==========================================================
    
    resultado_cei_base = montar_cei_abm.montar_cei_abm(
        estrutura_cei=ci["cei_original"],
        distribuicao=distribuicao_cei_base,
        importacoes_nominais=float(
            (
                ci["parcela_importada"]
                .reindex(setores)
                .fillna(0.0)
                * (
                    ci["conversao_de_pm_pb"]
                    @ ci["demanda_final_base"]
                )
            ).sum()
        ),
        exportacoes_nominais=float(
            ci["exportacoes_base"].sum()
        ),
        consumo_governo=float(
            ci["governo_base"].sum()
        ),
        fbcf=fbcf_cei_base,
        estoques=estoques_cei_periodo_zero,
        teste_flag=CONFIG_BASE["executar_testes"],
    )
    
    cei_periodo_zero = resultado_cei_base["cei"]
    
    
    
    
    
    # ============================================================================
    # 7. CICLO TEMPORAL: decisões, mercados, realização e estado herdado
    # ============================================================================
    
    # Cada período preserva a causalidade: expectativa, decisão, mercado,
    # realização, CEI e atualização do estado para o período seguinte.

    for t in range(1, periodos + 1):
    

        if t == periodo_choque:
            print(f"\n>>> PERÍODO {t}/{periodos} — CHOQUE <<<")
        else:
            print(f"Período {t}/{periodos}")
    
    
        # A. Forma Pc esperado a partir exclusivamente dos preços herdados.
    
    
        # Pc esperado usa apenas informação herdada; Pb e Pm são separados.
    
        # ==========================================================
        # EXPECTATIVA DE PREÇOS
        # ==========================================================
    
        inflacao_pc_anterior = (
            pc_anterior / pc_anterior_2 - 1.0
        )
    
        pc_esperado = (
            pc_anterior
            * (
                1.0
                + lambda_expectativa_precos
                * inflacao_pc_anterior
            )
        )
    
        pc = pc_esperado
    
        if np.any(pc_esperado <= 0.0):
            raise RuntimeError(
                f"Preço esperado não positivo no período {t}."
            )
    
        # ==============================================================
        # BLOCO DE DECISÕES INICIAIS DAS FIRMAS
        # ==============================================================
        # Esta é a primeira decisão produtiva do período. Nada que venha da
        # demanda corrente, da TRU legada ou da CEI abaixo pode alterá-la em t.
    
        for firma in firmas.values():
    
            firma.calcular_demanda_esperada(beta_investimento_nf)
    
    
            firma.decidir_producao(
                parametro_estoque_desejado=config_abm["parametro_estoque_desejado"],
                velocidade_ajuste_estoques=velocidade_ajuste_estoques_firmas,
            )
    
            firma.calcular_demanda_intermediaria()
    
            firma.calcular_demanda_trabalho()
    
            if firma.setor in setores_nf:
                firma.decidir_investimento(
                    v=v_investimento_nf,
                    depreciacao=depreciacao_capital_nf,
                    gamma_retorno=float(
                        config_abm.get("gamma_investimento_retorno", 0.5)
                    ),
                    gamma_investimento_capacidade=float(
                        config_abm.get(
                            "gamma_investimento_capacidade",
                            config_abm.get("gamma_investimento_retorno", 0.5),
                        )
                    ),
                )
            else:
                firma.investimento_liquido = 0.0
                firma.investimento_reposicao = 0.0
                firma.investimento_bruto = 0.0
    
            firma.atualizar_custo_e_preco(
                precos_insumos=pc_anterior,
                indice_salarios=indice_salarios,
            )
    
            firma.calcular_eob_recorrente_esperado()
    
            firma.calcular_dividendos()
      
    
    
    
    
        # G. Agrega as decisões microeconômicas por setor sem substituir os objetos.
    
        agregados = agregar_firmas(firmas, setores)
    
        agregados_firmas_periodos[t] = agregados
    
        demanda_intermediaria_real = (
            agregados["demanda_intermediaria_real"]
        )
    
        investimento_nf_total = (
            agregados["investimento_bruto"]
            .reindex(setores_nf)
            .fillna(0.0)
            .sum()
        )
    
        demanda_investimento_real = (
            pesos_bens_capital_nf
            .reindex(setores)
            .fillna(0.0)
            * investimento_nf_total
        ).rename("demanda_investimento_real")
        investimento_liquido_nf = agregados["investimento_liquido"].loc[setores_nf]
        investimento_reposicao_nf = agregados["investimento_reposicao"].loc[
            setores_nf
        ]
        investimento_nf_por_investidor = agregados["investimento_bruto"].loc[
            setores_nf
        ]
        investimento_nf_sem_piso = (
            investimento_liquido_nf + investimento_reposicao_nf
        ).rename("investimento_nf_sem_piso")
    
    
        # Converte os agregados das firmas, organizados por setor produtivo,
        # para a estrutura institucional usada pela CEI. Em particular, separa
        # o setor financeiro das demais firmas e reorganiza remunerações, VA e
        # demais fluxos necessários para alimentar as contas institucionais.
        # Esta etapa funciona apenas como uma ponte contábil entre o ABM e a CEI:
        # não redefine as decisões nem os resultados já determinados pelas firmas.
    
    
    
        # H. A FBCF familiar do período t depende da poupança observada em t-1.
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
        
    
        # A FBCF real decidida a partir da poupança de t-1 é paga ao preço da
        # Construção vigente em t. Assim, inflação não reduz artificialmente o
        # investimento real das famílias.
    
        fbcf_familias = investimento_familias["fbcf_familias_nominal"]
    
        # ==========================================================
        # DEMANDAS AUTÔNOMAS
        # ==========================================================
    
        # Atualiza os gastos nominais do governo, a FBCF autônoma e as exportações,
        # corrigindo seus valores pelos preços correntes e aplicando, quando previsto,
        # os choques exógenos definidos para o período.
    
        # ==========================================================
        # DEMANDAS AUTÔNOMAS
        # ==========================================================
    
        fator_governo = 1.0
        fator_investimento = 1.0
        fator_exportacoes = 1.0
    
    
        if choque_permanente:
    
            if t >= periodo_choque:
                fator_governo = cfg["multiplicador_governo"]
                fator_investimento = cfg["multiplicador_investimento"]
                fator_exportacoes = cfg["multiplicador_exportacoes"]
    
        else:
    
            if t == periodo_choque:
                fator_governo = cfg["multiplicador_governo"]
                fator_investimento = cfg["multiplicador_investimento"]
                fator_exportacoes = cfg["multiplicador_exportacoes"]
    
        governo_nominal = ci["governo_base"]* pc * fator_governo
    
        fbcf_fixa_nominal = fbcf_fixa_base * pc * fator_investimento
        
        exportacoes_nominais = ci["exportacoes_base"]* pc * fator_exportacoes
    
        # ==========================================================
        # INFLAÇÃO E ÍNDICE DE PREÇOS
        # ==========================================================
    
        # Calcula o índice agregado de preços do período a partir da cesta-base de
        # consumo e dos preços correntes. A comparação com o índice do período
        # anterior determina a inflação agregada usada no restante da simulação.
    
        inflacao_periodo = calcular_inflacao_periodo(
            ci["consumo_base"], pc, indice_precos_anterior
        )
        indice_precos = inflacao_periodo["indice_precos"]
    
        # ==========================================================
        # POLÍTICA MONETÁRIA e FLUXOS DE JUROS 
        # ==========================================================
    
        # Corrige os estoques financeiros pela inflação do período e atualiza a
        # taxa nominal de juros a partir da taxa real parametrizada, da inflação
        # corrente e da inércia da política monetária. Em seguida, calcula os juros
        # recebidos, pagos e líquidos sobre os ativos e passivos corrigidos.
    
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
    
    
        # Enquanto os mercados ainda não foram ligados à contabilidade, estes
        # dois fluxos usam a trajetória nominal calibrada na TRU-base.
    
        # ==========================================================
        # IMPOSTOS SOBRE PRODUTOS
        # ==========================================================
    
        impostos_produtos = (
            ci["taxa_impostos"]
            .reindex(setores)
            .fillna(0.0)
            * agregados["producao_nominal"]
            .reindex(setores)
            .fillna(0.0)
        ).rename("impostos_produtos")
    
        
        i_ff = cfg["setor_financeiro"]
    
        dados_firmas_cei_pre_mercado = {
            "ff": {
                "valor_adicionado": float(
                    agregados["valor_adicionado"].iloc[i_ff]
                ),
                "salarios": float(
                    agregados["salarios"].iloc[i_ff]
                ),
                "contribuicoes_efetivas": float(
                    agregados["contribuicoes"].iloc[i_ff]
                ),
                "dividendos": float(
                    agregados["dividendos"].iloc[i_ff]
                ),
                "outros_va": float(
                    agregados["outros_va"].iloc[i_ff]
                ),
            },
    
            "nf": {
                "valor_adicionado": float(
                    agregados["valor_adicionado"].sum()
                    - agregados["valor_adicionado"].iloc[i_ff]
                ),
                "salarios": float(
                    agregados["salarios"].sum()
                    - agregados["salarios"].iloc[i_ff]
                ),
                "contribuicoes_efetivas": float(
                    agregados["contribuicoes"].sum()
                    - agregados["contribuicoes"].iloc[i_ff]
                ),
                "dividendos": float(
                    agregados["dividendos"].sum()
                    - agregados["dividendos"].iloc[i_ff]
                ),
                "outros_va": float(
                    agregados["outros_va"].sum()
                    - agregados["outros_va"].iloc[i_ff]
                ),
            },
    
            "ocupacoes": float(
                agregados["ocupacoes"].sum()
            ),
        }
    
    
        distribuicao_pre_mercado = calcular_distribuicao_pre_mercado_abm(
            p=p,
            dados_firmas=dados_firmas_cei_pre_mercado,
            impostos_produtos=impostos_produtos,
            juros_recebidos=juros_recebidos,
            juros_pagos=juros_pagos,
            indice_salarios=indice_salarios,
            indice_precos=indice_precos_pre_mercado,
            setor_financeiro=cfg["setor_financeiro"],
            outras_transferencias_base=outras_transferencias_base,
        )
    
    
    
        consumo_nominal = distribuicao_pre_mercado["consumo_nominal"]
    
    
    
        # ==========================================================
        # J. Monta a demanda real PB e a demanda final nominal PM para o mercado.
        # DEMANDA SETORIAL PARA O MERCADO
        # ==========================================================
    
        demanda_real_setorial = (
            demanda_intermediaria_real
            + demanda_investimento_real
        ).rename("demanda_real_setorial")
    
            # ==========================================================
            # DEMANDA FINAL NOMINAL A PREÇOS DE MERCADO
            # ==========================================================
    
        demanda_final_pm_nominal = (
            consumo_nominal * pesos_consumo
            + governo_nominal
            + exportacoes_nominais
            + fbcf_familias * pesos_investimento_familias
            + fbcf_fixa_nominal
        ).rename("demanda_final_pm_nominal")
    
    
            # ==========================================================
            # CONVERSÃO PARA PREÇOS BÁSICOS
            # ==========================================================
    
        demanda_nominal_setorial = (
                ci["conversao_de_pm_pb"] @ demanda_final_pm_nominal
            ).rename("demanda_nominal_setorial")
    
        # ==========================================================
        # PREÇOS DOS IMPORTADOS
        # ==========================================================
    
        for importado in importados.values():
            importado.atualizar_preco(
                indice_cambio=indice_cambio,
            )
    
    
    
        # ==========================================================
        # K. O mercado aloca quantidades reais e orçamentos entre domésticos e importados.
        # MERCADOS
        # ==========================================================
    
        mercados = executar_mercados_periodo(
            setores=setores,
            firmas=firmas,
            importados=importados,
            demanda_real_setorial=demanda_real_setorial,
            demanda_nominal_setorial=demanda_nominal_setorial,
        )
    
    
       # ==========================================================
        # L. Agrega os preços de transação realizados em PB, PM e Pc.
        # PREÇOS REALIZADOS
        # ==========================================================
    
        precos_realizados = (
            calcular_precos_realizados_abm
            .calcular_precos_realizados_abm(
                setores=setores,
                firmas=firmas,
                importados=importados,
                G=ci["G"],
            )
        )
    
        pb_realizado = precos_realizados["pb"]
        pm_realizado = precos_realizados["pm"]
        preco_produto_realizado = precos_realizados["preco_produto"]
        pc_realizado = precos_realizados["pc"]
    
    
        # ==========================================================
        # M--O. Valoriza CI e FBCF NF pelas transações efetivamente realizadas.
        # VALORAÇÃO REALIZADA DO BLOCO REAL: CI + FBCF NF
        # ==========================================================
    
        conversao_pm_pb = (
            ci["conversao_de_pm_pb"]
            .reindex(
                index=setores,
                columns=setores,
            )
        )
    
        matriz_pm_pb = conversao_pm_pb.to_numpy(
            dtype=float
        )
    
        vendas_domesticas_setoriais = pd.Series(
            {
                setor: float(
                    sum(
                        firma.vendas_nominal
                        for firma in firmas.values()
                        if firma.setor == setor
                    )
                )
                for setor in setores
            },
            index=setores,
            dtype=float,
        )
    
        importacoes_basicas_nominais_pre = pd.Series(
            {
                setor: float(
                    importados[setor].vendas_nominal
                )
                for setor in setores
            },
            index=setores,
            dtype=float,
        )
    
        vendas_totais_pb = (
            vendas_domesticas_setoriais
            + importacoes_basicas_nominais_pre
        )
    
        # A demanda nominal já entrou no mercado convertida para PB.
        # Portanto, retirando-a das vendas totais obtemos o valor
        # efetivamente vendido para atender CI + FBCF NF.
    
        valor_bloco_real_pb = (
            vendas_totais_pb
            - demanda_nominal_setorial
        )
    
        preco_efetivo_bloco_real = (
            valor_bloco_real_pb
            .div(
                demanda_real_setorial.replace(
                    0.0,
                    np.nan,
                )
            )
            .fillna(0.0)
        )
    
        # ==========================================================
        # P. Reconstrói o custo intermediário realizado de cada firma a PM.
        # RESULTADOS REALIZADOS DAS FIRMAS
        # ==========================================================
    
        for firma in firmas.values():
    
            ci_firma_pb = (
                firma.demanda_intermediaria_real
                .reindex(setores)
                .fillna(0.0)
                * preco_efetivo_bloco_real
            )
    
            ci_firma_pm = pd.Series(
                np.linalg.solve(
                    matriz_pm_pb,
                    ci_firma_pb.to_numpy(
                        dtype=float
                    ),
                ),
                index=setores,
            )
    
            consumo_intermediario_firma = float(
                ci_firma_pm.sum()
            )
    
            if firma.producao_real > 0.0:
                firma.custo_intermediario_unitario_realizado = (
                    consumo_intermediario_firma
                    / firma.producao_real
                )
            else:
                firma.custo_intermediario_unitario_realizado = 0.0
    
            firma.calcular_resultado_realizado()
    
            firma.estoque_final()
    
    
        agregados_realizados = (
            agregar_resultados_realizados_firmas(
                firmas=firmas,
                setores=setores,
            )
        )
    
    
        # ==========================================================
        # Q. Registra produção menos vendas sem impor equilíbrio instantâneo.
        # VARIAÇÃO DE ESTOQUES
        # ==========================================================
    
        variacao_estoques_real = float(
            sum(
                firma.variacao_estoque_real
                for firma in firmas.values()
            )
        )
    
        variacao_estoques_nominal = float(
            sum(
                firma.variacao_estoque_real
                * firma.preco_transacao
                for firma in firmas.values()
            )
        )
    
    
        # ==========================================================
        # R. Consolida as vendas nominais PB dos fornecedores externos.
        # IMPORTAÇÕES REALIZADAS
        # ==========================================================
    
        importacoes_basicas_nominais = pd.Series(
            {
                setor: float(
                    importados[setor].vendas_nominal
                )
                for setor in setores
            },
            index=setores,
            dtype=float,
            name="importacoes_basicas_nominais",
        )
    
        importacoes_nominais = float(
            importacoes_basicas_nominais.sum()
        )
    
        # ==========================================================
        # O. Converte a cesta realizada de capital, PB, para o valor nominal PM.
        # FBCF NF REALIZADA
        # ==========================================================
    
        # O investimento NF é decidido em quantidade real.
        # O valor nominal aparece apenas ex post.
    
        investimento_nf_pb_realizado = (
            demanda_investimento_real
            .reindex(setores)
            .fillna(0.0)
            * preco_efetivo_bloco_real
        )
    
        investimento_nf_nominal_realizado = pd.Series(
            np.linalg.solve(
                matriz_pm_pb,
                investimento_nf_pb_realizado.to_numpy(
                    dtype=float
                ),
            ),
            index=setores,
            name="investimento_nf_nominal_realizado",
        )
    
        fbcf_nf_nominal_realizada = float(
            investimento_nf_nominal_realizado.sum()
        )
    
    
        # S. A base tributável vem diretamente dos usos realizados a PM.
        # USOS REALIZADOS A PREÇOS DE COMPRADOR
        # ==========================================================
    
        ci_pb_realizado = (
            demanda_intermediaria_real
            .reindex(setores)
            .fillna(0.0)
            * preco_efetivo_bloco_real
        )
    
        ci_pm_realizado = pd.Series(
            np.linalg.solve(
                matriz_pm_pb,
                ci_pb_realizado.to_numpy(
                    dtype=float
                ),
            ),
            index=setores,
            name="ci_pm_realizado",
        )
    
        demanda_total_pm_nominal_realizada = (
            ci_pm_realizado
            + demanda_final_pm_nominal
            .reindex(setores)
            .fillna(0.0)
            + investimento_nf_nominal_realizado
        ).rename(
            "demanda_total_pm_nominal_realizada"
        )
    
        # ==========================================================
        # IMPOSTOS SOBRE PRODUTOS REALIZADOS
        # ==========================================================
    
        impostos_produtos_realizados = (
            ci["taxa_impostos"]
            .reindex(setores)
            .fillna(0.0)
            * demanda_total_pm_nominal_realizada
        ).rename(
            "impostos_produtos_realizados"
        )
    
        impostos_produtos_total_realizado = float(
            impostos_produtos_realizados.sum()
        )
    
        impostos_produtos_ff_realizado = (
            p["parcela_impostos_produtos_ff"]
            * impostos_produtos_total_realizado
        )
    
        impostos_produtos_nf_realizado = (
            impostos_produtos_total_realizado
            - impostos_produtos_ff_realizado
        )
    
    
        # ==========================================================
        # DISTRIBUIÇÃO PARA A CEI PÓS-MERCADO
        # ==========================================================
    
        # A distribuição de renda foi decidida antes do mercado.
        # Não recalculamos consumo, dividendos, IR, benefícios etc.
        #
        # Apenas substituímos o VA e os impostos sobre produtos
        # pelos valores realizados.
    
        distribuicao_cei_periodo = (
            distribuicao_pre_mercado.copy()
        )
    
        i_ff = cfg["setor_financeiro"]
    
        va_realizado_ff = float(
            agregados_realizados[
                "valor_adicionado"
            ].iloc[i_ff]
        )
    
        va_realizado_nf = float(
            agregados_realizados[
                "valor_adicionado"
            ].sum()
            - va_realizado_ff
        )
    
        distribuicao_cei_periodo[
            "va_planejado_ff"
        ] = va_realizado_ff
    
        distribuicao_cei_periodo[
            "va_planejado_nf"
        ] = va_realizado_nf
    
        distribuicao_cei_periodo[
            "impostos_produtos_ff"
        ] = impostos_produtos_ff_realizado
    
        distribuicao_cei_periodo[
            "impostos_produtos_nf"
        ] = impostos_produtos_nf_realizado
     
        
    
    
        # ==========================================================
        # FBCF FIXA INSTITUCIONAL
        # ==========================================================
    
        # A FBCF fixa já é uma trajetória nominal.
        # Apenas repartimos seu total entre as instituições
        # segundo a composição calibrada no ano-base.
    
        fbcf_fixa_total_base = float(
            fbcf_fixa_base.sum()
        )
    
        fbcf_fixa_total = float(
            fbcf_fixa_nominal.sum()
        )
    
        if np.isclose(
            fbcf_fixa_total_base,
            0.0,
        ):
            fator_fbcf_fixa = 1.0
        else:
            fator_fbcf_fixa = (
                fbcf_fixa_total
                / fbcf_fixa_total_base
            )
    
        fbcf_cei_periodo = {
    
            "familias": float(
                fbcf_familias
            ),
    
            "governo": float(
                fbcf_fixa_cei_base["governo"]
                * fator_fbcf_fixa
            ),
    
            "firmas_financeiras": float(
                fbcf_fixa_cei_base[
                    "firmas_financeiras"
                ]
                * fator_fbcf_fixa
            ),
    
            "firmas_nao_financeiras": float(
                fbcf_nf_nominal_realizada
            ),
    
            "setor_externo": float(
                fbcf_fixa_cei_base[
                    "setor_externo"
                ]
                * fator_fbcf_fixa
            ),
        }
    
    
        # ==========================================================
        # T. Fecha a CEI com os fluxos realizados e mede capacidades institucionais.
        # CEI DO PERÍODO
        # ==========================================================
    
        resultado_cei = (
            montar_cei_abm.montar_cei_abm(
                estrutura_cei=ci["cei_original"],
    
                distribuicao=distribuicao_cei_periodo,
    
                importacoes_nominais=(
                    importacoes_nominais
                ),
    
                exportacoes_nominais=float(
                    exportacoes_nominais.sum()
                ),
    
                consumo_governo=float(
                    governo_nominal.sum()
                ),
    
                fbcf=fbcf_cei_periodo,
    
                estoques={
                    "familias": 0.0,
                    "governo": 0.0,
                    "firmas_financeiras": 0.0,
                    "firmas_nao_financeiras":
                        variacao_estoques_nominal,
                    "setor_externo": 0.0,
                },
                teste_flag=CONFIG_BASE["executar_testes"]
            )
        )
    
        cei_periodo = resultado_cei["cei"]
    
    
        capacidade = (
            resultado_cei[
                "capacidade_financiamento"
            ]
        )
    
        poupanca_familias = float(
            distribuicao_pre_mercado[
                "poupanca_familias"
            ]
        )
    
    
        # Emprego corrente determina o salário usado no próximo período.
        mercado_trabalho = calcular_mercado_trabalho(
            agregados["ocupacoes"].sum(),
            p["pea"],
            cfg["taxa_desemprego_base"],
            cfg["a0"],
            cfg["a1"],
            cfg["a3"],
        )
    
        emprego = mercado_trabalho["emprego"]
        taxa_desemprego = mercado_trabalho["taxa_desemprego"]
    
        variacao_salarios = max(
            0.0,
            mercado_trabalho["variacao_salarios"],
        )
    
    
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
    
        ##########################################################
        # U--W. Calcula agregados macro realizados e armazena as séries do período.
    
        # ==========================================================
        # INFLAÇÃO REALIZADA
        # ==========================================================
    
        inflacao_realizada_periodo = calcular_inflacao_periodo(
            ci["consumo_base"],
            pc_realizado,
            indice_precos_anterior,
        )
    
        indice_precos_realizado = (
            inflacao_realizada_periodo[
                "indice_precos"
            ]
        )
    
        inflacao_realizada = (
            inflacao_realizada_periodo[
                "inflacao"
            ]
        )
    
    
        # ==========================================================
        # PIB NOMINAL
        # ==========================================================
    
        pib_nominal = (
            float(
                agregados_realizados[
                    "valor_adicionado"
                ].sum()
            )
            + impostos_produtos_total_realizado
        )
    
    
        # ==========================================================
        # AGREGADOS REAIS SIMPLES
        # ==========================================================
    
        producao_real_total = float(
            sum(
                firma.producao_real
                for firma in firmas.values()
            )
        )
    
        vendas_real_total = float(
            sum(
                firma.vendas_real
                for firma in firmas.values()
            )
        )
    
        # ==========================================================
        # DEMANDAS AUTÔNOMAS REAIS
        # ==========================================================
    
        governo_real = float(
            (
                governo_nominal
                / pc_realizado
            ).sum()
        )
    
        fbcf_fixa_real = float(
            (
                fbcf_fixa_nominal
                / pc_realizado
            ).sum()
        )
    
        exportacoes_real = float(
            (
                exportacoes_nominais
                / pc_realizado
            ).sum()
        )
    
    
        # ==========================================================
        # DIAGNÓSTICO DA DEMANDA REAL
        # ==========================================================
    
        consumo_real = float(
            (
                consumo_nominal
                * pesos_consumo
                / pc_realizado
            ).sum()
        )
    
        renda_disponivel_real = float(
            distribuicao_pre_mercado[
                "renda_disponivel_familias"
            ]
            / indice_precos_realizado
        )
    
        consumo_intermediario_real_total = float(
            demanda_intermediaria_real.sum()
        )
    
        investimento_nf_real_total = float(
            demanda_investimento_real.sum()
        )
    
        demanda_esperada_total = float(
            sum(
                firma.demanda_esperada
                for firma in firmas.values()
            )
        )
    
        vendas_real_total = float(
            sum(
                firma.vendas_real
                for firma in firmas.values()
            )
        )
    
        # ==========================================================
        # HISTÓRICO
        # ==========================================================
    
        historico.append(
            {
                "periodo": t,
                "ano": cfg["ano"] + t,
    
                "indice_precos":
                    indice_precos_realizado,
    
                "inflacao":
                    inflacao_realizada,
    
                "indice_salarios":
                    indice_salarios,
    
                "indice_cambio":
                    indice_cambio,
    
                "taxa_juros_nominal":
                    taxa_juros_nominal,
    
                "pib_nominal":
                    pib_nominal,
    
                "producao_real":
                    producao_real_total,
    
                "vendas_real":
                    vendas_real_total,
    
                "emprego":
                    emprego,
    
                "taxa_desemprego":
                    taxa_desemprego,
    
                "consumo_nominal":
                    float(consumo_nominal),
    
                "poupanca_familias_nominal":
                    float(poupanca_familias),
    
                "fbcf_familias_nominal":
                    float(fbcf_familias),
    
                "fbcf_nf_real":
                    float(
                        demanda_investimento_real.sum()
                    ),
    
                "fbcf_nf_nominal":
                    fbcf_nf_nominal_realizada,
    
                "fbcf_fixa_nominal":
                    float(
                        fbcf_fixa_nominal.sum()
                    ),
    
                "variacao_estoques_real":
                    variacao_estoques_real,
    
                "variacao_estoques_nominal":
                    variacao_estoques_nominal,
    
                "importacoes_nominais":
                    importacoes_nominais,
    
                "exportacoes_nominais":
                    float(
                        exportacoes_nominais.sum()
                    ),
    
                "discrepancia_cei":
                    float(
                        resultado_cei["discrepancia"]
                    ),
    
                "governo_real":
                    governo_real,
    
                "fbcf_fixa_real":
                    fbcf_fixa_real,
    
                "exportacoes_real":
                    exportacoes_real,
    
                "consumo_real":
                    consumo_real,
    
                "renda_disponivel_real":
                    renda_disponivel_real,
    
                "consumo_intermediario_real":
                    consumo_intermediario_real_total,
    
                "investimento_nf_real":
                    investimento_nf_real_total,
    
                "demanda_esperada_total":
                    demanda_esperada_total,
    
                "vendas_real_total":
                    vendas_real_total,
    
                        }
    
        )
    
    
        # ==========================================================
        # RESULTADOS POR PERÍODO
        # ==========================================================
    
        pc_periodos[t] = (
            pc_realizado.copy()
        )
    
        pb_periodos[t] = (
            pb_realizado.copy()
        )
    
        pm_periodos[t] = (
            pm_realizado.copy()
        )
    
        pc_esperado_periodos[t] = (
            pc_esperado.copy()
        )
    
        inflacao_pc_setorial_periodos[t] = (
            pc_realizado
            / pc_anterior
            - 1.0
        )
    
        cei_periodos[t] = (
            cei_periodo.copy(
                deep=True
            )
        )
    
        capacidades[t] = (
            capacidade.copy()
        )
    
    
        # ----------------------------------------------------------
        # Financeiro
        # ----------------------------------------------------------
    
        ativos_financeiros_periodos[t] = (
            ativos_financeiros_periodo.copy()
        )
    
        passivos_financeiros_periodos[t] = (
            passivos_financeiros_periodo.copy()
        )
    
        estoque_financeiro_periodos[t] = (
            estoque_financeiro_periodo.copy()
        )
    
        aquisicao_ativos_periodos[t] = (
            aquisicao_ativos.copy()
        )
    
        emissao_passivos_periodos[t] = (
            emissao_passivos.copy()
        )
    
        juros_liquidos_periodos[t] = (
            juros_liquidos.copy()
        )
    
        juros_recebidos_periodos[t] = (
            juros_recebidos.copy()
        )
    
        juros_pagos_periodos[t] = (
            juros_pagos.copy()
        )
    
        reavaliacao_financeira_periodos[t] = (
            reavaliacao_financeira.copy()
        )
    
    
        # ----------------------------------------------------------
        # Investimento
        # ----------------------------------------------------------
    
        investimento_nf_real_periodos[t] = (
            demanda_investimento_real.copy()
        )
    
        investimento_nf_nominal_periodos[t] = (
            investimento_nf_nominal_realizado.copy()
        )
    
        fbcf_fixa_nominal_periodos[t] = (
            fbcf_fixa_nominal.copy()
        )
    
        variacao_estoques_real_periodos[t] = (
            variacao_estoques_real
        )
    
        variacao_estoques_nominal_periodos[t] = (
            variacao_estoques_nominal
        )
    
    
        # ==========================================================
        # X. Atualiza estados micro e macro que serão herdados por t + 1.
        # ESTADO DAS FIRMAS PARA t + 1
        # ==========================================================
    
        for firma in firmas.values():
    
            firma.atualizar_estado(
                depreciacao_capital_nf
            )
    
    
        # ==========================================================
        # ESTADO MACRO PARA t + 1
        # ==========================================================
    
        # Salário do próximo período.
    
        indice_salarios = (
            indice_salarios
            * (1.0 + variacao_salarios)
        )
    
    
        # Câmbio do próximo período.
    
        indice_cambio = (
            indice_cambio
            * (
                1.0
                + cfg[
                    "repasse_inflacao_cambio"
                ]
                * inflacao_realizada
            )
        )
    
    
        # Preços realizados passam a ser a informação herdada.
    
        pc_anterior_2 = (
            pc_anterior.copy()
        )
    
        pc_anterior = (
            pc_realizado.copy()
        )
    
        indice_precos_anterior = (
            indice_precos_realizado
        )
    
    
        # Poupança corrente determina a FBCF familiar
        # do próximo período.
    
        poupanca_familias_anterior = float(
            poupanca_familias
        )
    
    
        # Estoques financeiros herdados.
    
        ativos_financeiros = (
            ativos_financeiros_periodo.copy()
        )
    
        passivos_financeiros = (
            passivos_financeiros_periodo.copy()
        )
    
        estoque_financeiro = (
            estoque_financeiro_periodo.copy()
        )
    
    

    # A tabela é uma visão derivada: o estado corrente permanece nos agentes.
    historico_df = pd.DataFrame(historico).set_index("periodo")
    return {
        "nome": nome_cenario,
        "config": config_execucao,
        "config_abm": config_abm_execucao,
        "historico": historico_df,
        "resultados": resultados,
        "firmas": firmas,
        "importados": importados,
        "cei": cei_periodos,
        "pc": pc_periodos,
        "pb": pb_periodos,
        "pm": pm_periodos,
        "capacidade_financiamento": capacidades,
        "investimento_nf_real": investimento_nf_real_periodos,
        "investimento_nf_nominal": investimento_nf_nominal_periodos,
        "fbcf_fixa_nominal": fbcf_fixa_nominal_periodos,
        "variacao_estoques_real": variacao_estoques_real_periodos,
        "variacao_estoques_nominal": variacao_estoques_nominal_periodos,
    }


def comparar_cenarios(
    baseline: dict,
    cenario: dict,
    variaveis: list[str],
    percentual: bool = True,
) -> pd.DataFrame:
    """Compara variáveis dos históricos no índice de período."""
    base = baseline["historico"].loc[:, variaveis]
    alternativo = cenario["historico"].loc[:, variaveis]
    base, alternativo = base.align(alternativo, join="inner", axis=0)
    if percentual:
        return 100.0 * (alternativo / base - 1.0)
    return alternativo - base


def painel_cenarios(
    cenarios: dict[str, dict],
    variaveis: list[str],
) -> pd.DataFrame:
    """Organiza cenários em colunas MultiIndex cenário/variável."""
    painel = pd.concat(
        {
            nome: resultado["historico"].loc[:, variaveis]
            for nome, resultado in cenarios.items()
        },
        axis=1,
    )
    painel.columns = painel.columns.set_names(["cenario", "variavel"])
    return painel


if __name__ == "__main__":

    periodo_choque = 20

    # ==============================================================
    # CENÁRIOS
    # ==============================================================

    baseline = simular_modelo(
        nome_cenario="Baseline"
    )

    cenario_governo = simular_modelo(
        nome_cenario="Governo +5%",
        config={
            "periodo_choque": periodo_choque,
            "choque_permanente": False,
            "multiplicador_governo": 1.05,
        },
    )

    cenario_investimento = simular_modelo(
        nome_cenario="Investimento +5%",
        config={
            "periodo_choque": periodo_choque,
            "choque_permanente": False,
            "multiplicador_investimento": 1.05,
        },
    )

    cenario_exportacoes = simular_modelo(
        nome_cenario="Exportações +5%",
        config={
            "periodo_choque": periodo_choque,
            "choque_permanente": False,
            "multiplicador_exportacoes": 1.05,
        },
    )

    cenarios = {
        "Governo +5%": cenario_governo,
        "Investimento +5%": cenario_investimento,
        "Exportações +5%": cenario_exportacoes,
    }


    # ==============================================================
    # FUNÇÃO PARA RESPOSTAS AO CHOQUE
    # ==============================================================

    def resposta_percentual(cenario, variavel):
        """Desvio percentual do cenário em relação ao baseline."""

        base = baseline["historico"][variavel]
        alt = cenario["historico"][variavel]

        base, alt = base.align(alt, join="inner")

        resposta = 100 * (
            alt / base.replace(0.0, np.nan) - 1
        )

        resposta.index = resposta.index - periodo_choque

        return resposta


    def resposta_pontos_percentuais(cenario, variavel):
        """Diferença em pontos percentuais contra o baseline."""

        base = baseline["historico"][variavel]
        alt = cenario["historico"][variavel]

        base, alt = base.align(alt, join="inner")

        resposta = 100 * (alt - base)

        resposta.index = resposta.index - periodo_choque

        return resposta


    # ==============================================================
    # VARIÁVEIS DO PAINEL
    # ==============================================================

    variaveis = {
        "producao_real": (
            "Produção real",
            resposta_percentual,
            "% em relação ao baseline",
        ),
        "consumo_real": (
            "Consumo real",
            resposta_percentual,
            "% em relação ao baseline",
        ),
        "fbcf_nf_real": (
            "Investimento das firmas",
            resposta_percentual,
            "% em relação ao baseline",
        ),
        "emprego": (
            "Emprego",
            resposta_percentual,
            "% em relação ao baseline",
        ),
        "taxa_desemprego": (
            "Taxa de desemprego",
            resposta_pontos_percentuais,
            "p.p. em relação ao baseline",
        ),
        "inflacao": (
            "Inflação",
            resposta_pontos_percentuais,
            "p.p. em relação ao baseline",
        ),
    }


    # ==============================================================
    # PAINEL TIPO IMPULSO-RESPOSTA
    # ==============================================================

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 11),
        sharex=True,
    )

    axes = axes.flatten()

    horizonte_anterior = 5
    horizonte_posterior = 20


    for ax, (
        variavel,
        (titulo, funcao_resposta, unidade),
    ) in zip(axes, variaveis.items()):

        for nome, cenario in cenarios.items():

            resposta = funcao_resposta(
                cenario,
                variavel,
            )

            resposta = resposta.loc[
                (resposta.index >= -horizonte_anterior)
                & (resposta.index <= horizonte_posterior)
            ]

            ax.plot(
                resposta.index,
                resposta.values,
                label=nome,
                linewidth=2,
            )

        # Linha de equilíbrio
        ax.axhline(
            0,
            linewidth=0.8,
            linestyle="--",
        )

        # Momento do choque
        ax.axvline(
            0,
            linewidth=0.8,
            linestyle=":",
        )

        ax.set_title(titulo)
        ax.set_ylabel(unidade)
        ax.grid(alpha=0.25)


    # Eixo horizontal
    axes[-1].set_xlabel("Períodos após o choque")
    axes[-2].set_xlabel("Períodos após o choque")


    # Uma legenda para o painel inteiro
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(cenarios),
        frameon=False,
    )

    fig.suptitle(
        "Respostas da economia aos choques exógenos",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0.06, 1, 0.96]
    )

    plt.show()
