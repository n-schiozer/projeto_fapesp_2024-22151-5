"""Ponto de entrada organizado da versão SFC--IO--ABM.

O arquivo não reimplementa a economia: ele apenas configura os dados, prepara
o ano-base e chama o ciclo temporal já implementado nos módulos específicos.
Assim, pode ser executado diretamente ou usado célula a célula em um notebook.
"""

from configuracao_projeto import ARQUIVO_CEI, DATA_DIR
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais
from macro.simulacao_cei_2 import simul_
from tests.unit.testar_mercado_leilao_etapa11 import testar_mercado_leilao_etapa11
from tests.unit.testar_firmas_vs_legado import testar_firmas_vs_legado

import matplotlib.pyplot as plt

"""Três blocos auditáveis: TRU, CEI e ciclo temporal."""

import numpy as np
import pandas as pd


import contabilidade.montar_cei_abm as montar_cei_abm

import mercados.calcular_precos_realizados_abm as calcular_precos_realizados_abm

from agentes.agregar_firmas import agregar_firmas, separar_agregados_firmas_cei,agregar_resultados_realizados_firmas
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
    calcular_distribuicao_pre_mercado_abm
)
from financeiro.financeiro_abm import inicializar_financeiro_abm
from mercados.atendimento_categorial_abm import ratear_atendimento_proporcional 
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

from agentes.fornecedor_importado_abm import inicializar_importados_abm

from investimento.investimento_abm import (
    atualizar_demandas_autonomas,
    calcular_estoques_legado_periodo,
    calcular_fbcf_familias,
    calcular_investimento_nf_periodo,
    montar_investimento_e_cei_legado,
)


from resultados.resultados_abm_legado import (
    armazenar_resultados_periodo,
    finalizar_resultados,
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

CONFIG = {
    "ano": 2020,
    "nivel": 20,
    "aba_cei": "Python",
    "periodos": 50,
    "multiplicador_governo": 1.0,
    "multiplicador_investimento": 1.0,
    "multiplicador_exportacoes": 1.0,
    # Mesmo no cenário sem choque, o período precisa ser válido para simul_().
    "periodo_choque": 1,
    "choque_permanente": True,
    "taxa_desemprego_base": 0.138,
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
    "utilizacao_capacidade_inicial_padrao": 0.8,
    "inicializacao_investimento_nf": "estacionaria",
    "razao_estoque_producao": 1.0 / 12.0,
    "velocidade_ajuste_estoques": 0.25,
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

CONFIG_ABM = {
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
        "parametro_markup": 0.5,
        "markup_min": 0.0,
        "markup_max": 10.0,
        "epsilon_market_share": 1e-12,
    },
    # Capacidade real máxima do importado nos setores de leilão, como múltiplo
    # da quantidade importada observada no ano-base.
    "multiplicador_capacidade_importada": 1.5,
    "velocidade_ajuste_estoques_firmas": 0.25,
    "lambda_expectativa_precos": 1.0,
    # Mude para True somente quando quiser imprimir toda a bateria de regressões.
    "executar_testes": False,
}

periodos = CONFIG["periodos"]
config_abm = CONFIG_ABM

condicoes_iniciais = preparar_condicoes_iniciais(
        CONFIG,
        DATA_DIR,
        ARQUIVO_CEI,
)



# Auxiliar:


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


print("""Executa o ciclo ABM, preservando temporariamente a TRU/CEI legadas.""")

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

importados = inicializar_importados_abm(
    condicoes_iniciais=ci,
    firmas=firmas,
)



# TESTE

variacao_estoque_firmas_base = sum(
    firma.producao_base_real - firma.producao_vendida_base_real
    for firma in firmas.values()
)

variacao_estoque_cei_base = float(
    ci["valores_cei"].iat[
        L["estoques"],
        C["nf_s"]
    ]
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
fbcf_nf_pb = calibracao_investimento_nf["fbcf_nf_pb"].copy()
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


# ==========================================================
# INVESTIMENTO AUTÔNOMO
# ==========================================================

investimento_autonomo_base_pm = (
    fbcf_fixa_base.copy()
)

investimento_autonomo_base_real = (
    ci["conversao_de_pm_pb"]
    @ investimento_autonomo_base_pm
).rename(
    "investimento_autonomo_base_real"
)


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
    teste_flag=CONFIG["executar_testes"]
)

cei_periodo_zero = resultado_cei_base["cei"]

capacidade_abm_base = (
    resultado_cei_base[
        "capacidade_financiamento"
    ]
)



# ==========================================================
# PARÂMETRO DE ESTOQUE DESEJADO ESTACIONÁRIO
# ==========================================================

estoque_inicial_total = sum(
    firma.estoque
    for firma in firmas.values()
    if firma.forma_estoque
)

demanda_esperada_t1_total = sum(
    (
        firma.demanda_esperada
        + beta_investimento_nf
        * (
            firma.demanda_realizada
            - firma.demanda_esperada
        )
    )
    for firma in firmas.values()
    if firma.forma_estoque
)

parametro_estoque_desejado_estacionario = (
    estoque_inicial_total
    / demanda_esperada_t1_total
)

print(
    "θ estoque atual:",
    config_abm["parametro_estoque_desejado"],
)

print(
    "θ estoque estacionário:",
    parametro_estoque_desejado_estacionario,
)



# ==================================================================
# FOR TEMPORAL
# ==================================================================







# ==========================================================
# TESTE DA PONTE PB -> PM DA FBCF NF NO ANO-BASE
# ==========================================================

B = (
    ci["conversao_de_pm_pb"]
    .reindex(
        index=setores,
        columns=setores,
    )
    .to_numpy(dtype=float)
)

fbcf_nf_pm_reconstruida_base = pd.Series(
    np.linalg.solve(
        B,
        investimento_nf_base.to_numpy(dtype=float),
    ),
    index=setores,
)

print(
    "\n================ FBCF NF BASE: PB -> PM ================\n"
)

print(
    "FBCF NF PB:",
    investimento_nf_base.sum(),
)

print(
    "FBCF NF PM reconstruída:",
    fbcf_nf_pm_reconstruida_base.sum(),
)

print(
    "FBCF NF PM observada:",
    fbcf_nf_pm.sum(),
)

print(
    "Erro:",
    fbcf_nf_pm_reconstruida_base.sum()
    - fbcf_nf_pm.sum(),
)


diagnostico_nominal = []



for t in range(1, periodos + 1):

    print(f"\n{'=' * 50}")
    print(f"PERÍODO {t}")
    print(f"{'=' * 50}\n")


    # Cálculo de preços esperados:


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

        #firma.calcular_demanda_esperada(0)

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
  


    # ==========================================================
    # TESTE DO EOB RECORRENTE E DIVIDENDOS ANTES DA AGREGAÇÃO
    # ==========================================================

    setor_financeiro = setores[
        cfg["setor_financeiro"]
    ]

    eob_recorrente_manual_nf = 0.0
    eob_recorrente_objeto_nf = 0.0

    dividendos_manual_nf = 0.0
    dividendos_objeto_nf = 0.0

    for firma in firmas.values():

        if firma.setor == setor_financeiro:
            continue

        crescimento_producao = (
            firma.producao_planejada_real
            / firma.producao_anterior
        )

        crescimento_custo = (
            firma.custo_unitario
            / firma.custo_unitario_anterior
        )

        eob_manual = (
            firma.eob_misto_realizado_anterior
            * crescimento_producao
            * crescimento_custo
        )

        dividendo_manual = (
            firma.parametro_dividendos
            * max(0.0, eob_manual)
        )

        eob_recorrente_manual_nf += eob_manual

        eob_recorrente_objeto_nf += (
            firma.eob_misto_recorrente_esperado
        )

        dividendos_manual_nf += dividendo_manual

        dividendos_objeto_nf += firma.dividendos


    print("\nTESTE DIRETO DOS OBJETOS - NF")

    print(
        f"EOB recorrente manual : "
        f"{eob_recorrente_manual_nf:,.4f}"
    )

    print(
        f"EOB recorrente objeto : "
        f"{eob_recorrente_objeto_nf:,.4f}"
    )

    print(
        f"Dividendos manual     : "
        f"{dividendos_manual_nf:,.4f}"
    )

    print(
        f"Dividendos objeto     : "
        f"{dividendos_objeto_nf:,.4f}"
    )


    # CÁLCULO DOS AGREGADOS MACROECONÔMICOS A PARTIR DAS FIRMAS:

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
    investimento_nf_real_total = float(investimento_nf_total)

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
    investimento_nf_real = demanda_investimento_real.copy().rename(
        "investimento_nf_real"
    )
    investimento_nf_nominal = (investimento_nf_real * pc).rename(
        "investimento_nf_nominal"
    )
    fbcf_nf_nominal = float(investimento_nf_nominal.sum())
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


    # Converte os agregados das firmas, organizados por setor produtivo,
    # para a estrutura institucional usada pela CEI. Em particular, separa
    # o setor financeiro das demais firmas e reorganiza remunerações, VA e
    # demais fluxos necessários para alimentar as contas institucionais.
    # Esta etapa funciona apenas como uma ponte contábil entre o ABM e a CEI:
    # não redefine as decisões nem os resultados já determinados pelas firmas.

    dados_firmas_cei_periodos[t] = separar_agregados_firmas_cei(
        agregados_firmas_periodos[t],
        setores,
        ci["razoes_va"],
        cfg["setor_financeiro"],
    )


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
    
    if config_abm["executar_testes"]:

        # Calcula temporariamente o investimento pelo mecanismo agregado legado
        # apenas para comparar seus resultados com as decisões das firmas.
        # Este bloco é exclusivamente diagnóstico e não altera o estado do modelo.

        investimento_nf = calcular_investimento_nf_periodo(
            producao_nf_corrente,
            producao_nf_anterior,
            estoque_capital_nf,
            pesos_bens_capital_nf,
            pc,
            beta_investimento_nf,
            v_investimento_nf,
            depreciacao_capital_nf,
        )

        testes_periodo = testar_firmas_vs_legado(
            firmas=firmas,
            setores_nf=setores_nf,
            producao_nf_corrente=producao_nf_corrente,
            estoque_capital_nf=estoque_capital_nf,
            investimento_liquido_nf=investimento_nf["investimento_liquido"],
            investimento_reposicao_nf=investimento_nf["investimento_reposicao"],
            investimento_nf_por_investidor=investimento_nf["por_investidor"],
            parametro_estoque_desejado=config_abm["parametro_estoque_desejado"],
            velocidade_ajuste_estoques=velocidade_ajuste_estoques_firmas,
        )

    # A FBCF real decidida a partir da poupança de t-1 é paga ao preço da
    # Construção vigente em t. Assim, inflação não reduz artificialmente o
    # investimento real das famílias.

    fbcf_familias = investimento_familias["fbcf_familias_nominal"]

    # A equação do capital produz quantidades reais. Cada bem de capital é
    # valorizado pelo preço comprador do setor que o fornece.

    #investimento_nf_nominal = investimento_nf["nominal"]

    #fbcf_nf_nominal = float(investimento_nf_nominal.sum())

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

    investimento_autonomo_real = (
    investimento_autonomo_base_real
        * fator_investimento
    ).rename(
        "investimento_autonomo_real"
    )

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


    if config_abm["executar_testes"] and t == 1:

        # ==========================================================
        # TESTE DOS DIVIDENDOS
        # ==========================================================

        i_ff = cfg["setor_financeiro"]

        # Dividendos pagos pelas firmas no ABM
        dividendos_ff_abm = float(
            agregados["dividendos"].iloc[i_ff]
        )

        dividendos_nf_abm = float(
            agregados["dividendos"].sum()
            - agregados["dividendos"].iloc[i_ff]
        )

        dividendos_totais_abm = (
            dividendos_ff_abm
            + dividendos_nf_abm
        )

        # Dividendos recebidos após a distribuição institucional
        dividendos_familias_abm = float(
            distribuicao_pre_mercado["dividendos_familias"]
        )

        dividendos_exterior_abm = float(
            distribuicao_pre_mercado["dividendos_exterior"]
        )

        dividendos_recebidos_abm = (
            dividendos_familias_abm
            + dividendos_exterior_abm
        )

        # ==========================================================
        # REFERÊNCIA DA CEI
        # ==========================================================

        dividendos_ff_base = float(
            ci["valores_cei"].iat[
                L["dividendos"],
                C["ff_s"],
            ]
        )

        dividendos_nf_base = float(
            ci["valores_cei"].iat[
                L["dividendos"],
                C["nf_s"],
            ]
        )

        dividendos_familias_base = float(
            ci["valores_cei"].iat[
                L["dividendos"],
                C["familias_e"],
            ]
        )

        dividendos_exterior_base = float(
            ci["valores_cei"].iat[
                L["dividendos"],
                C["externo_e"],
            ]
        )

        # Referência nominal simples para t=1.
        dividendos_ff_referencia = (
            dividendos_ff_base
            * indice_precos_pre_mercado
        )

        dividendos_nf_referencia = (
            dividendos_nf_base
            * indice_precos_pre_mercado
        )

        dividendos_familias_referencia = (
            dividendos_familias_base
            * indice_precos_pre_mercado
        )

        dividendos_exterior_referencia = (
            dividendos_exterior_base
            * indice_precos_pre_mercado
        )

        # ==========================================================
        # DIFERENÇAS
        # ==========================================================

        print("\nTESTE DIVIDENDOS - t=1")

        print(
            f"FF  | referência: {dividendos_ff_referencia:,.4f}"
            f" | ABM: {dividendos_ff_abm:,.4f}"
            f" | diferença: "
            f"{100 * (dividendos_ff_abm / dividendos_ff_referencia - 1):.4f}%"
        )

        print(
            f"NF  | referência: {dividendos_nf_referencia:,.4f}"
            f" | ABM: {dividendos_nf_abm:,.4f}"
            f" | diferença: "
            f"{100 * (dividendos_nf_abm / dividendos_nf_referencia - 1):.4f}%"
        )

        print(
            f"Famílias | referência: {dividendos_familias_referencia:,.4f}"
            f" | ABM: {dividendos_familias_abm:,.4f}"
            f" | diferença: "
            f"{100 * (dividendos_familias_abm / dividendos_familias_referencia - 1):.4f}%"
        )

        print(
            f"Exterior | referência: {dividendos_exterior_referencia:,.4f}"
            f" | ABM: {dividendos_exterior_abm:,.4f}"
            f" | diferença: "
            f"{100 * (dividendos_exterior_abm / dividendos_exterior_referencia - 1):.4f}%"
        )

        # ==========================================================
        # FECHAMENTO: PAGOS = RECEBIDOS
        # ==========================================================

        residuo_dividendos = (
            dividendos_totais_abm
            - dividendos_recebidos_abm
        )

        print(
            f"Dividendos pagos     : {dividendos_totais_abm:,.4f}"
        )

        print(
            f"Dividendos recebidos : {dividendos_recebidos_abm:,.4f}"
        )

        print(
            f"Resíduo              : {residuo_dividendos:,.10f}"
        )

        if not np.isclose(
            dividendos_totais_abm,
            dividendos_recebidos_abm,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Os dividendos pagos pelas firmas não coincidem "
                "com os dividendos recebidos por famílias e exterior."
            )


    if config_abm["executar_testes"] and t == 1:

        setor_financeiro = setores[cfg["setor_financeiro"]]

        firmas_nf = [
            firma
            for firma in firmas.values()
            if firma.setor != setor_financeiro
        ]

        dividendos_nf_base_regra = 0.0
        dividendos_nf_so_producao = 0.0
        dividendos_nf_completo = 0.0

        for firma in firmas_nf:

            crescimento_producao = (
                firma.producao_planejada_real
                / firma.producao_anterior
            )

            crescimento_custo = (
                firma.custo_unitario
                / firma.custo_unitario_anterior
            )

            eob_anterior = (
                firma.eob_misto_realizado_anterior
            )

            # Ano-base da própria regra
            dividendos_nf_base_regra += (
                firma.parametro_dividendos
                * max(0.0, eob_anterior)
            )

            # Contrafactual: muda apenas produção
            dividendos_nf_so_producao += (
                firma.parametro_dividendos
                * max(
                    0.0,
                    eob_anterior
                    * crescimento_producao,
                )
            )

            # Regra efetivamente utilizada
            dividendos_nf_completo += (
                firma.parametro_dividendos
                * max(
                    0.0,
                    eob_anterior
                    * crescimento_producao
                    * crescimento_custo,
                )
            )

        print("\nDECOMPOSIÇÃO DIVIDENDOS NF - t=1")

        print(
            f"Base da regra             : "
            f"{dividendos_nf_base_regra:,.4f}"
        )

        print(
            f"Somente efeito produção   : "
            f"{dividendos_nf_so_producao:,.4f}"
        )

        print(
            f"Produção + custo          : "
            f"{dividendos_nf_completo:,.4f}"
        )

        print(
            f"Efeito produção           : "
            f"{100 * (
                dividendos_nf_so_producao
                / dividendos_nf_base_regra
                - 1.0
            ):.4f}%"
        )

        print(
            f"Efeito custo após produção: "
            f"{100 * (
                dividendos_nf_completo
                / dividendos_nf_so_producao
                - 1.0
            ):.4f}%"
        )

        print(
            f"Crescimento total da regra: "
            f"{100 * (
                dividendos_nf_completo
                / dividendos_nf_base_regra
                - 1.0
            ):.4f}%"
        )

        print(
            f"Inflação benchmark        : "
            f"{100 * (
                indice_precos_pre_mercado - 1.0
            ):.4f}%"
        )


    consumo_nominal = distribuicao_pre_mercado["consumo_nominal"]


    # ==========================================================
    # TESTE: CONSUMO ABM VS. CEI-BASE
    # ==========================================================

    if config_abm["executar_testes"] and t == 1:

            consumo_cei_referencia = (
                consumo_nominal_base
                * indice_precos_pre_mercado
            )

            diferenca_consumo = (
                consumo_nominal
                - consumo_cei_referencia
            )

            diferenca_consumo_pct = (
                diferenca_consumo
                / consumo_cei_referencia
            )

            print("\nTESTE DO CONSUMO")
            print(
                f"Consumo CEI referência : {consumo_cei_referencia:,.4f}"
            )
            print(
                f"Consumo ABM            : {consumo_nominal:,.4f}"
            )
            print(
                f"Diferença              : {diferenca_consumo:,.4f}"
            )
            print(
                f"Diferença percentual   : {100 * diferenca_consumo_pct:.4f}%"
            )


            renda_disponivel_cei_base = (
                consumo_nominal_base
                / p["propensao_consumir"]
            )


            renda_disponivel_abm = (
                distribuicao_pre_mercado["renda_disponivel_familias"]
            )

            renda_disponivel_cei_referencia = (
                renda_disponivel_cei_base
                * indice_precos_pre_mercado
            )

            diferenca_yd_pct = (
                renda_disponivel_abm
                / renda_disponivel_cei_referencia
                - 1.0
            )

            print(
                f"YD CEI referência      : {renda_disponivel_cei_referencia:,.4f}"
            )
            print(
                f"YD ABM                 : {renda_disponivel_abm:,.4f}"
            )
            print(
                f"Diferença YD           : {100 * diferenca_yd_pct:.4f}%"
            )



    # ==========================================================
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


    print("\nPERÍODO:", t)

    print("\n================ PC ANTERIOR 2 ================\n")
    print(pc_anterior_2)

    print("\n================ PC ANTERIOR ==================\n")
    print(pc_anterior)

    print("\n================ PC ESPERADO ==================\n")
    print(pc_esperado)

    print("\n================ PREÇOS DAS FIRMAS ============\n")

    for setor in setores:

        precos_setor = [
            firma.preco_firma
            for firma in firmas.values()
            if (
                firma.setor == setor
                and firma.regime == "industrial"
            )
        ]

        if precos_setor:
            print(
                setor,
                "min =", min(precos_setor),
                "media =", np.mean(precos_setor),
                "max =", max(precos_setor),
            )
 

    # ==========================================================
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


    # ==========================================================
    # IMPOSTOS SOBRE PRODUTOS REALIZADOS
    # ==========================================================

    # A taxa de impostos foi calibrada sobre valores a preços
    # de mercado. A produção realizada está a preços básicos.
    #
    # Portanto:
    #
    # oferta PB realizada
    #       -> inversa da conversão PM/PB
    #       -> base tributável PM
    #       -> impostos

    # ==========================================================
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
            teste_flag=CONFIG["executar_testes"],
        )
    )

    cei_periodo = resultado_cei["cei"]

    resultado_cei["saldo_linhas"]

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

    emprego = float(
        distribuicao_pre_mercado[
            "emprego"
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
    variacao_salarios = mercado_trabalho["variacao_salarios"]   

    variacao_salarios = max(
        0.0,
        mercado_trabalho["variacao_salarios"],
    )


    # ==========================================================
    # COMPARAÇÃO CEI BASE x PERÍODO 1
    # ==========================================================

    if t == 1:

        linha_va = L["va"]
        linha_consumo = L["consumo"]
        linha_fbcf = L["fbcf"]
        linha_estoques = L["estoques"]

        colunas_entrada = [
            entrada
            for entrada, saida
            in COLUNAS_SETORES.values()
        ]

    colunas_saida = [
        saida
        for entrada, saida
        in COLUNAS_SETORES.values()
    ]

    # ======================================================
    # ANO-BASE
    # ======================================================

    importacoes_base = float(
        (
            ci["parcela_importada"]
            .reindex(setores)
            .fillna(0.0)
            * (
                ci["conversao_de_pm_pb"]
                @ ci["demanda_final_base"]
            )
        ).sum()
    )

    exportacoes_base = float(
        ci["exportacoes_base"].sum()
    )

    governo_base = float(
        ci["governo_base"].sum()
    )

    recursos_base = float(
        cei_periodo_zero.iloc[
            linha_va,
            colunas_entrada,
        ].sum()
    )

    produto_domestico_base = (
        recursos_base
        - importacoes_base
    )

    consumo_total_base = float(
        cei_periodo_zero.iloc[
            linha_consumo,
            colunas_saida,
        ].sum()
    )

    consumo_familias_base_cei = (
        consumo_total_base
        - governo_base
    )

    fbcf_total_base = float(
        cei_periodo_zero.iloc[
            linha_fbcf,
            colunas_saida,
        ].sum()
    )

    estoques_total_base = float(
        cei_periodo_zero.iloc[
            linha_estoques,
            colunas_saida,
        ].sum()
    )

    usos_base = (
        consumo_familias_base_cei
        + governo_base
        + fbcf_total_base
        + estoques_total_base
        + exportacoes_base
    )

    residuo_base = (
        recursos_base
        - usos_base
    )

    # ======================================================
    # PERÍODO 1
    # ======================================================

    recursos_t1 = float(
        cei_periodo.iloc[
            linha_va,
            colunas_entrada,
        ].sum()
    )

    produto_domestico_t1 = (
        recursos_t1
        - importacoes_nominais
    )

    consumo_total_t1 = float(
        cei_periodo.iloc[
            linha_consumo,
            colunas_saida,
        ].sum()
    )

    governo_t1 = float(
        governo_nominal.sum()
    )

    consumo_familias_t1 = (
        consumo_total_t1
        - governo_t1
    )

    fbcf_total_t1 = float(
        cei_periodo.iloc[
            linha_fbcf,
            colunas_saida,
        ].sum()
    )

    estoques_total_t1 = float(
        cei_periodo.iloc[
            linha_estoques,
            colunas_saida,
        ].sum()
    )

    exportacoes_t1 = float(
        exportacoes_nominais.sum()
    )

    usos_t1 = (
        consumo_familias_t1
        + governo_t1
        + fbcf_total_t1
        + estoques_total_t1
        + exportacoes_t1
    )

    residuo_t1 = (
        recursos_t1
        - usos_t1
    )

    # ======================================================
    # TABELA COMPARATIVA
    # ======================================================

    comparacao_cei = pd.DataFrame(
        {
            "base": {
                "Produto doméstico":
                    produto_domestico_base,

                "Importações":
                    importacoes_base,

                "RECURSOS":
                    recursos_base,

                "Consumo famílias":
                    consumo_familias_base_cei,

                "Gasto governo":
                    governo_base,

                "FBCF":
                    fbcf_total_base,

                "Δ Estoques":
                    estoques_total_base,

                "Exportações":
                    exportacoes_base,

                "USOS":
                    usos_base,

                "RESÍDUO":
                    residuo_base,
            },

            "t1": {
                "Produto doméstico":
                    produto_domestico_t1,

                "Importações":
                    importacoes_nominais,

                "RECURSOS":
                    recursos_t1,

                "Consumo famílias":
                    consumo_familias_t1,

                "Gasto governo":
                    governo_t1,

                "FBCF":
                    fbcf_total_t1,

                "Δ Estoques":
                    estoques_total_t1,

                "Exportações":
                    exportacoes_t1,

                "USOS":
                    usos_t1,

                "RESÍDUO":
                    residuo_t1,
            },
        }
    )

    comparacao_cei["diferença"] = (
        comparacao_cei["t1"]
        - comparacao_cei["base"]
    )

    comparacao_cei["var_%"] = np.where(
        comparacao_cei["base"].abs() > 1e-8,
        100.0
        * comparacao_cei["diferença"]
        / comparacao_cei["base"].abs(),
        np.nan,
    )

    print(
        "\n"
        "================ CEI BASE x PERÍODO 1 ================\n"
    )

    print(
        comparacao_cei.to_string(
            float_format=lambda x: f"{x:,.3f}"
        )
    )


    ##################

    #dados_setoriais_firmas = dados_firmas_cei_pre_mercado["setorial"]
   

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

    ### Cálculo do PIB

    print(
        "\n================ CEI ================\n"
    )

    print(
        "Discrepância:",
        resultado_cei["discrepancia"],
    )

    print(
        "\nCapacidade de financiamento:"
    )

    for nome, valor in capacidade.items():
        print(
            f"{nome:25s}: {valor:15,.3f}"
        )


    # ==========================================================
    # IDENTIDADE DE BENS E SERVIÇOS
    # ==========================================================

    produto_domestico = (
        float(
            cei_periodo.iloc[
                L["va"],
                colunas_entrada,
            ].sum()
        )
        - importacoes_nominais
    )

    consumo_total = float(
        cei_periodo.iloc[
            L["consumo"],
            colunas_saida,
        ].sum()
    )

    gasto_governo = float(
        governo_nominal.sum()
    )

    consumo_familias = (
        consumo_total
        - gasto_governo
    )

    fbcf_total = float(
        cei_periodo.iloc[
            L["fbcf"],
            colunas_saida,
        ].sum()
    )

    estoques_total = float(
        cei_periodo.iloc[
            L["estoques"],
            colunas_saida,
        ].sum()
    )

    exportacoes_total = float(
        exportacoes_nominais.sum()
    )

    recursos = (
        produto_domestico
        + importacoes_nominais
    )

    usos = (
        consumo_familias
        + gasto_governo
        + fbcf_total
        + estoques_total
        + exportacoes_total
    )

    print(
        "\n"
        "================ IDENTIDADE DE BENS E SERVIÇOS ================\n"
    )

    print(
        f"Produto doméstico : {produto_domestico:15,.3f}"
    )
    print(
        f"Importações       : {importacoes_nominais:15,.3f}"
    )
    print(
        f"RECURSOS          : {recursos:15,.3f}"
    )

    print()

    print(
        f"Consumo famílias  : {consumo_familias:15,.3f}"
    )
    print(
        f"Gasto governo     : {gasto_governo:15,.3f}"
    )
    print(
        f"FBCF              : {fbcf_total:15,.3f}"
    )
    print(
        f"Δ Estoques        : {estoques_total:15,.3f}"
    )
    print(
        f"Exportações       : {exportacoes_total:15,.3f}"
    )
    print(
        f"USOS              : {usos:15,.3f}"
    )

    print()

    print(
        f"RESÍDUO           : {recursos - usos:15,.3f}"
    )



    # ==========================================================
    # DEMANDA DESEJADA x VENDAS REALIZADAS
    # ==========================================================

    demanda_final_desejada = float(
        demanda_final_pm_nominal.sum()
    )

    demanda_real_desejada = float(
        demanda_real_setorial.sum()
    )

    vendas_domesticas_nominais = float(
        sum(
            firma.vendas_nominal
            for firma in firmas.values()
        )
    )

    importacoes_realizadas_nominais = float(
        sum(
            importado.vendas_nominal
            for importado in importados.values()
        )
    )

    print(
        "\n"
        "================ MERCADO x CEI ================\n"
    )

    print(
        "Demanda final nominal desejada :",
        f"{demanda_final_desejada:15,.3f}",
    )

    print(
        "Demanda real CI + I            :",
        f"{demanda_real_desejada:15,.3f}",
    )

    print(
        "Vendas domésticas nominais     :",
        f"{vendas_domesticas_nominais:15,.3f}",
    )

    print(
        "Importações nominais realizadas:",
        f"{importacoes_realizadas_nominais:15,.3f}",
    )

    print(
        "Demanda não atendida real      :",
        f"{sum(f.demanda_nao_atendida_real for f in firmas.values()):15,.3f}",
    )



    ##########################################################
    # ARMAZENAMENTO DOS DADOS:

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

    diagnostico_nominal.append({
        "VA": va_realizado_ff + va_realizado_nf,
        "impostos": impostos_produtos_total_realizado,
        "importacoes": importacoes_nominais,
        "consumo": consumo_nominal,
        "governo": governo_nominal.sum(),
        "exportacoes": exportacoes_nominais.sum(),
        "fbcf_familias": fbcf_familias,
        "fbcf_fixa": fbcf_fixa_nominal.sum(),
        "fbcf_nf": fbcf_nf_nominal_realizada,
        "estoques": variacao_estoques_nominal,
    })



diagnostico_nominal = pd.DataFrame(diagnostico_nominal)

print(
    100 * diagnostico_nominal.pct_change()
)

breakpoint()


# ==========================================================
# GRÁFICOS
# ==========================================================

historico_df = (
    pd.DataFrame(historico)
    .set_index("periodo")
)


# ==========================================================
# ATIVIDADE
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["producao_real"],
    marker="o",
    label="Produção",
)

ax.plot(
    historico_df.index,
    historico_df["vendas_real"],
    marker="o",
    label="Vendas",
)

ax.set_title("Produção e vendas reais")
ax.set_xlabel("Período")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# PIB NOMINAL
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["pib_nominal"],
    marker="o",
)

ax.set_title("PIB nominal")
ax.set_xlabel("Período")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# INFLAÇÃO E DESEMPREGO
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    100 * historico_df["inflacao"],
    marker="o",
    label="Inflação",
)

ax.plot(
    historico_df.index,
    100 * historico_df["taxa_desemprego"],
    marker="o",
    label="Desemprego",
)

ax.set_title("Inflação e desemprego")
ax.set_xlabel("Período")
ax.set_ylabel("%")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# INVESTIMENTO
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["fbcf_nf_real"],
    marker="o",
    label="Investimento NF",
)

ax.plot(
    historico_df.index,
    historico_df["variacao_estoques_real"],
    marker="o",
    label="Variação de estoques",
)

ax.set_title("Investimento e variação de estoques")
ax.set_xlabel("Período")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# PREÇOS
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["indice_precos"],
    marker="o",
    label="Preços",
)

ax.plot(
    historico_df.index,
    historico_df["indice_salarios"],
    marker="o",
    label="Salários",
)

ax.plot(
    historico_df.index,
    historico_df["indice_cambio"],
    marker="o",
    label="Câmbio",
)

ax.set_title("Índices de preços, salários e câmbio")
ax.set_xlabel("Período")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# SETOR EXTERNO
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["exportacoes_nominais"],
    marker="o",
    label="Exportações",
)

ax.plot(
    historico_df.index,
    historico_df["importacoes_nominais"],
    marker="o",
    label="Importações",
)

ax.set_title("Exportações e importações")
ax.set_xlabel("Período")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# DISCREPÂNCIA DA CEI
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["discrepancia_cei"],
    marker="o",
)

ax.axhline(
    0.0,
    linewidth=1,
)

ax.set_title("Discrepância da CEI")
ax.set_xlabel("Período")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# DEMANDAS AUTÔNOMAS REAIS
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["governo_real"],
    marker="o",
    label="Governo",
)

ax.plot(
    historico_df.index,
    historico_df["fbcf_fixa_real"],
    marker="o",
    label="Investimento autônomo",
)

ax.plot(
    historico_df.index,
    historico_df["exportacoes_real"],
    marker="o",
    label="Exportações",
)

ax.set_title("Demandas autônomas em termos reais")
ax.set_xlabel("Período")
ax.set_ylabel("Valor a preços do ano-base")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# COMPONENTES ENDÓGENOS DA DEMANDA REAL
# ==========================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["consumo_real"],
    marker="o",
    label="Consumo",
)

ax.plot(
    historico_df.index,
    historico_df["consumo_intermediario_real"],
    marker="o",
    label="Consumo intermediário",
)

ax.plot(
    historico_df.index,
    historico_df["investimento_nf_real"],
    marker="o",
    label="Investimento NF",
)

ax.set_title("Componentes endógenos da demanda real")
ax.set_xlabel("Período")
ax.set_ylabel("Quantidade real")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    historico_df.index,
    historico_df["demanda_esperada_total"],
    marker="o",
    label="Demanda esperada",
)

ax.plot(
    historico_df.index,
    historico_df["producao_real"],
    marker="o",
    label="Produção",
)

ax.plot(
    historico_df.index,
    historico_df["vendas_real_total"],
    marker="o",
    label="Vendas",
)

ax.set_title(
    "Expectativa, produção e vendas"
)

ax.set_xlabel("Período")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()





# ==========================================================
# DIAGNÓSTICO DOS ESTOQUES
# ==========================================================

estoque_total_real = sum(
    firma.estoque
    for firma in firmas.values()
)

variacao_estoques_real = sum(
    firma.variacao_estoque_real
    for firma in firmas.values()
)

estoque_por_setor = pd.Series(
    {
        setor: sum(
            firma.estoque
            for firma in firmas.values()
            if firma.setor == setor
        )
        for setor in setores
    },
    name=f"estoque_t{t}",
)

print(
    f"t={t:3d} | "
    f"Estoque total={estoque_total_real:,.3f} | "
    f"Δ Estoque={variacao_estoques_real:,.3f}"
)
