"""Laboratório SFC--IO--ABM limpo e auditável.

Preserva o ciclo econômico da referência e remove somente caminhos
legados, testes manuais, gráficos e diagnósticos temporários. As unidades
são explícitas: quantidade real, preço básico (PB) e preço de comprador
(PM/Pc) não são intercambiados.
"""

from configuracao_projeto import (
    ARQUIVO_CEI,
    DATA_DIR,
    validar_caminhos_dados,
)
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais

"""Três blocos auditáveis: TRU, CEI e ciclo temporal."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
)


from resultados.resultados_abm_legado import (
    inicializar_resultados_abm,
)

from mercados.executar_mercados_periodo_abm import executar_mercados_periodo
from mercados.regulacao_producao_abm import calcular_decisoes_regulador

from calibracao.calibracao_investimento_nf_abm import (
    calibrar_investimento_nf_abm,
)

# %% =====================================================================
# 1. ARQUIVOS DE DADOS
# ========================================================================
# A TRU de 2020 fica na pasta indicada por DATA_DIR. A CEI é a planilha que
# contém os fluxos institucionais iniciais da economia.

DATA_DIR, ARQUIVO_CEI = validar_caminhos_dados(DATA_DIR, ARQUIVO_CEI)

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
    "periodos": 25,
    "multiplicador_governo": 1,
    "multiplicador_investimento": 1.0,
    "multiplicador_exportacoes": 1.0,
    # Mesmo no cenário sem choque, o período precisa ser válido para simul_().
    "periodo_choque": 2,
    "choque_permanente": True,
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
    "a0": 0.03,
    "a1": 0.2,
    "a3": 0.2,
    "repasse_inflacao_cambio": 1.0,
    "taxa_juros_real": 0.06,
    "inertia_pm": 0.5,
    "fracao_reavaliacao_financeira": 1.0,
    "tolerancia_consumo": 1e-6,
    "max_iteracoes_consumo": 100,
    # Mude para True somente quando quiser imprimir toda a bateria de regressões.
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

SETORES_LEILAO =[]
SETORES_REGULADOS = []

SETORES_LEILAO = [
    "A - Agricultura, pecuária, produção florestal, pesca e aquicultura",
    "D - Eletricidade e gás",
]

SETORES_REGULADOS = ["D - Eletricidade e gás"]

CONFIG_ABM = {
    "numero_firmas_industria": 25,
    "numero_firmas_leilao": 25,
    # Mantém a configuração especial anterior: uma firma financeira agregada.
    "numero_firmas_por_setor": {SETOR_FINANCEIRO: 1},
    "setores_leilao": SETORES_LEILAO,
    # Regulação e regime de mercado são conceitos independentes.
    "setores_regulados": SETORES_REGULADOS,
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
    "adj_r_obs_inicial" : 1, # Para ser reduzido o componente de risco < 1
    # u* é tecnológico/operacional e independe do regime de preço.
    "utilizacao_capacidade_normal": 0.80,
    "gamma_investimento_retorno": 0.5,
    "gamma_investimento_capacidade": 0.5,
    "choques_climaticos": {
        "ativo": False,
        "setores": {
            "A - Agricultura, pecuária, produção florestal, pesca e aquicultura": {
                "periodo_choque": 5,
                "multiplicador_produtividade": 0.95,
                "choque_permanente": False,
            },

            "D - Eletricidade e gás": {
                "periodo_choque": 5,
                "multiplicador_produtividade": 0.95,
                "choque_permanente": False,
            },
        },
    },
}

periodos = CONFIG["periodos"]
config_abm = CONFIG_ABM

condicoes_iniciais = preparar_condicoes_iniciais(
        CONFIG,
        DATA_DIR,
        ARQUIVO_CEI,
)

# Auxiliar:

ci = condicoes_iniciais
cfg = ci["config"]
setores = list(ci["setores"])
setores_regulados_desconhecidos = set(
    config_abm["setores_regulados"]
).difference(setores)
if setores_regulados_desconhecidos:
    raise ValueError(
        "setores_regulados contém setores inexistentes: "
        f"{sorted(setores_regulados_desconhecidos)}"
    )
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
# homogêneo, isso faz Pc_1 = Pb_1 = Pm_1 = 1 + inflacao.

# a0 é simultaneamente o componente autônomo da variação salarial e a
# inflação herdada utilizada para iniciar a simulação.

inflacao = float(cfg["a0"])

if inflacao <= -1.0:
    raise ValueError("inflacao deve ser maior que -1.")

indice_salarios = 1.0 + inflacao
indice_cambio = 1.0 + inflacao
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
pc_anterior_2 = pc_anterior / (1.0 + inflacao)

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
    "inflacao": inflacao,
    "indice_salarios": 1.0,
    "indice_cambio": 1.0,
    "taxa_juros_nominal": ( 1+ taxa_juros_real) * ( 1 + inflacao) - 1,
    "pib_real": pib_base,
    "pib_nominal": pib_base,
    "emprego": float(ci["va_base"].loc["Fator trabalho (ocupações)"].sum()),
    "taxa_desemprego": cfg["taxa_desemprego_inicial"],
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
    teste_flag=CONFIG["executar_testes"],   
)

cei_periodo_zero = resultado_cei_base["cei"]

# PESOS DO ESTOQUE DE CAPITAL

pesos_preco_capital = (
    pesos_bens_capital_nf
    .reindex(setores)
    .fillna(0.0)
)

soma_pesos_preco_capital = float(
    pesos_preco_capital.sum()
)

if soma_pesos_preco_capital <= 0.0:
    raise RuntimeError(
        "Os pesos dos bens de capital devem somar valor positivo."
    )

pesos_preco_capital = (
    pesos_preco_capital
    / soma_pesos_preco_capital
)

# ==========================================================
# RENTABILIDADE NORMAL SETORIAL - ANO-BASE
# ==========================================================

preco_capital_base = 1.0

taxa_retorno_parametro_setorial = {}

for setor in setores_nf:

    firmas_setor = [
        firma
        for firma in firmas.values()
        if firma.setor == setor
    ]

    eob_base_setorial = sum(
        firma.eob_misto_realizado
        for firma in firmas_setor
    )

    capital_base_setorial = sum(
        firma.estoque_capital_real
        for firma in firmas_setor
    )

    if capital_base_setorial <= 0.0:
        continue

    taxa_retorno_bruta_base = (
        eob_base_setorial
        / (
            preco_capital_base
            * capital_base_setorial
        )
    )

    taxa_retorno_observada_base = (
        taxa_retorno_bruta_base
        - depreciacao_capital_nf
    )

    taxa_retorno_parametro_setorial[setor] = (
        taxa_retorno_observada_base
        - taxa_juros_real
    )


for firma in firmas.values():

    if firma.setor not in taxa_retorno_parametro_setorial:
        continue

    firma.taxa_retorno_parametro = (
        taxa_retorno_parametro_setorial[
            firma.setor
        ]
    ) * CONFIG_ABM["adj_r_obs_inicial"]

    # Inicializa r_obs e r_ajustado no ano-base.
    firma.calcular_taxa_retorno_observada(
        preco_capital=preco_capital_base,
        depreciacao=depreciacao_capital_nf,
        taxa_juros_real=taxa_juros_real,
    )
    firma.taxa_retorno_ajustada_anterior = (
        firma.taxa_retorno_ajustada
    )

# ============================================================================
# . FATOR CLIMÁTICO
# ============================================================================


def obter_fator_produtividade_climatica(
    setor: str,
    periodo: int,
    config_abm: dict,
) -> float:
    """Retorna o multiplicador climático da produtividade do capital."""

    config_clima = config_abm.get(
        "choques_climaticos",
        {},
    )

    if not config_clima.get("ativo", False):
        return 1.0

    choque = config_clima.get(
        "setores",
        {},
    ).get(setor)

    if choque is None:
        return 1.0

    periodo_choque = int(
        choque["periodo_choque"]
    )

    multiplicador = float(
        choque["multiplicador_produtividade"]
    )

    choque_permanente = bool(
        choque.get("choque_permanente", False)
    )

    if not 0.0 <= multiplicador <= 1.0:
        raise ValueError(
            "multiplicador_produtividade deve estar entre 0 e 1."
        )

    if choque_permanente:
        if periodo >= periodo_choque:
            return multiplicador
    else:
        if periodo == periodo_choque:
            return multiplicador

    return 1.0


# ==========================================================
# DIAGNÓSTICO DO CHOQUE CLIMÁTICO
# ==========================================================

teste_clima_periodos = []
teste_regulacao_periodos = []
taxas_retorno_observadas_periodos = []
diagnostico_capacidade_setorial_periodos = []


def registrar_diagnostico_capacidade_setorial(
    periodo: int,
    depreciacao: float | None,
) -> None:
    """Registra identidades de capacidade e investimento agregadas por setor."""

    for setor in setores:
        firmas_setor = [
            firma for firma in firmas.values() if firma.setor == setor
        ]
        if not firmas_setor:
            continue

        capital = float(sum(firma.estoque_capital_real for firma in firmas_setor))
        producao_normal = float(
            sum(firma.producao_normal_real for firma in firmas_setor)
        )
        capacidade_estrutural = float(
            sum(
                firma.capacidade_produtiva_estrutural_real
                for firma in firmas_setor
            )
        )
        capacidade_efetiva = float(
            sum(firma.capacidade_produtiva_real for firma in firmas_setor)
        )
        producao_planejada = float(
            sum(firma.producao_planejada_real for firma in firmas_setor)
        )
        producao_real = float(sum(firma.producao_real for firma in firmas_setor))
        investimento_liquido = float(
            sum(firma.investimento_liquido for firma in firmas_setor)
        )
        investimento_reposicao = float(
            sum(firma.investimento_reposicao for firma in firmas_setor)
        )
        investimento_bruto = float(
            sum(firma.investimento_bruto for firma in firmas_setor)
        )

        if depreciacao is None:
            capital_seguinte = capital
        else:
            capital_seguinte = float(
                sum(
                    (1.0 - depreciacao) * firma.estoque_capital_real
                    + firma.investimento_bruto
                    for firma in firmas_setor
                )
            )

        utilizacao_normal = (
            producao_normal / capacidade_estrutural
            if capacidade_estrutural > 0.0 else np.nan
        )
        fator_clima = (
            capacidade_efetiva / capacidade_estrutural
            if capacidade_estrutural > 0.0 else np.nan
        )
        capitais_desejados = [
            firma.capital_desejado
            for firma in firmas_setor
            if np.isfinite(firma.capital_desejado)
        ]
        gaps_capital = [
            firma.gap_capital
            for firma in firmas_setor
            if np.isfinite(firma.gap_capital)
        ]
        diagnostico_capacidade_setorial_periodos.append(
            {
                "periodo": periodo,
                "setor": setor,
                "produtividade_capital_normal": (
                    producao_normal / capital if capital > 0.0 else np.nan
                ),
                "utilizacao_capacidade_normal": utilizacao_normal,
                "producao_normal": producao_normal,
                "capacidade_estrutural": capacidade_estrutural,
                "capacidade_efetiva": capacidade_efetiva,
                "fator_clima": fator_clima,
                "producao_planejada": producao_planejada,
                "producao_real": producao_real,
                "utilizacao_planejada": (
                    producao_planejada / capacidade_efetiva
                    if capacidade_efetiva > 0.0 else np.nan
                ),
                "demanda_esperada": float(
                    sum(firma.demanda_esperada for firma in firmas_setor)
                ),
                "capital": capital,
                "capital_desejado": (
                    float(sum(capitais_desejados))
                    if capitais_desejados else np.nan
                ),
                "gap_capital": (
                    float(sum(gaps_capital)) if gaps_capital else np.nan
                ),
                "investimento_liquido": investimento_liquido,
                "investimento_reposicao": investimento_reposicao,
                "investimento_bruto": investimento_bruto,
                "capital_periodo_seguinte": capital_seguinte,
            }
        )

for firma in firmas.values():

    taxas_retorno_observadas_periodos.append(
        {
            "periodo": 0,
            "firma": firma.id,
            "setor": firma.setor,
            "regime": firma.regime,
            "preco_capital": preco_capital_base,
            "capital_real": firma.estoque_capital_real,
            "eob_realizado": firma.eob_misto_realizado,
            "r_obs_bruto":
                firma.taxa_retorno_bruta_observada,
            "r_obs":
                firma.taxa_retorno_observada,
        }
    )

registrar_diagnostico_capacidade_setorial(
    periodo=0,
    depreciacao=None,
)

# ============================================================================
# 7. CICLO TEMPORAL: decisões, mercados, realização e estado herdado
# ============================================================================

# Cada período preserva a causalidade: expectativa, decisão, mercado,
# realização, CEI e atualização do estado para o período seguinte.
for t in range(1, periodos + 1):


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

    decisoes_producao = {}
    diagnosticos_clima_iniciais = {}

    # Primeiro laço: cada firma decide descentralizadamente.
    for firma in firmas.values():

        fator_clima = obter_fator_produtividade_climatica(
            setor=firma.setor,
            periodo=t,
            config_abm=config_abm,
        )

        # Capital antes da aplicação do fator climático.
        # O choque climático NÃO deve alterar K.
        capital_antes_clima = firma.estoque_capital_real

        firma.atualizar_capacidade_produtiva(
            fator_produtividade_climatica=fator_clima,
        )

        # ==========================================================
        # TESTES DA CAPACIDADE CLIMÁTICA
        # ==========================================================

        capital_depois_clima = firma.estoque_capital_real

        if np.isfinite(
            firma.produtividade_capital_capacidade
        ):

            capacidade_estrutural_esperada = (
                firma.produtividade_capital_capacidade
                * firma.estoque_capital_real
            )

            capacidade_climatica_esperada = (
                fator_clima
                * capacidade_estrutural_esperada
            )

            erro_capital_clima = (
                capital_depois_clima
                - capital_antes_clima
            )

            erro_capacidade_estrutural = (
                firma.capacidade_produtiva_estrutural_real
                - capacidade_estrutural_esperada
            )

            erro_capacidade_climatica = (
                firma.capacidade_produtiva_real
                - capacidade_climatica_esperada
            )

        else:

            capacidade_estrutural_esperada = np.inf
            capacidade_climatica_esperada = np.inf

            erro_capital_clima = (
                capital_depois_clima
                - capital_antes_clima
            )

            erro_capacidade_estrutural = 0.0
            erro_capacidade_climatica = 0.0

        diagnosticos_clima_iniciais[firma.id] = {
            "fator_clima": fator_clima,
            "capital_antes_clima": capital_antes_clima,
            "erro_capital_clima": erro_capital_clima,
            "capacidade_efetiva_esperada": capacidade_climatica_esperada,
            "erro_capacidade_estrutural": erro_capacidade_estrutural,
            "erro_capacidade_climatica": erro_capacidade_climatica,
        }

        firma.calcular_demanda_esperada(
            beta=beta_investimento_nf,
        )

        decisoes_producao[firma.id] = firma.calcular_producao_desejada(
            parametro_estoque_desejado=(
                config_abm["parametro_estoque_desejado"]
            ),
            velocidade_ajuste_estoques=(
                config_abm["velocidade_ajuste_estoques_firmas"]
            ),
        )
    # Etapa intermediária: só setores configurados sofrem coordenação.
    decisoes_regulador = calcular_decisoes_regulador(
        firmas=firmas,
        decisoes_producao=decisoes_producao,
        setores_regulados=config_abm["setores_regulados"],
    )

    # Segundo laço: realiza a decisão final e preserva o restante do fluxo.
    for firma in firmas.values():
        quantidade_final = decisoes_regulador.get(
            firma.id,
            decisoes_producao[firma.id],
        )
        firma.realizar_producao(quantidade_final)

        diagnostico_clima = diagnosticos_clima_iniciais[firma.id]
        producao_esperada = min(
            quantidade_final,
            firma.capacidade_produtiva_real,
        )
        restricao_esperada = max(
            0.0,
            quantidade_final - firma.capacidade_produtiva_real,
        )
        if not np.isfinite(firma.capacidade_produtiva_real):
            producao_esperada = quantidade_final
            restricao_esperada = 0.0

        teste_clima_periodos.append(
            {
                "periodo": t,
                "firma": firma.id,
                "setor": firma.setor,
                "fator_clima": diagnostico_clima["fator_clima"],
                "capital": firma.estoque_capital_real,
                "capacidade_estrutural": firma.capacidade_produtiva_estrutural_real,
                "capacidade_efetiva": firma.capacidade_produtiva_real,
                "capacidade_efetiva_esperada": diagnostico_clima["capacidade_efetiva_esperada"],
                "producao_desejada": firma.producao_desejada_real,
                "producao_antes_regulacao": decisoes_producao[firma.id],
                "redistribuicao_regulador": quantidade_final - decisoes_producao[firma.id],
                "producao_planejada": firma.producao_planejada_real,
                "producao_real": firma.producao_real,
                "restricao_capacidade": firma.producao_restringida_capacidade_real,
                "erro_capital_clima": diagnostico_clima["erro_capital_clima"],
                "erro_capacidade_estrutural": diagnostico_clima["erro_capacidade_estrutural"],
                "erro_capacidade_climatica": diagnostico_clima["erro_capacidade_climatica"],
                "erro_producao": firma.producao_real - producao_esperada,
                "erro_restricao": firma.producao_restringida_capacidade_real - restricao_esperada,
            }
        )
        teste_regulacao_periodos.append(
            {
                "periodo": t,
                "setor": firma.setor,
                "firma": firma.id,
                "regulado": firma.setor in set(config_abm["setores_regulados"]),
                "producao_desejada": firma.producao_desejada_real,
                "producao_final": quantidade_final,
                "capacidade_efetiva": firma.capacidade_produtiva_real,
            }
        )

        firma.calcular_demanda_intermediaria()

        firma.calcular_demanda_trabalho()

        # Decisão de investimento. É um for pois alguns setores não investem:
        if firma.setor in setores_nf:

            firma.decidir_investimento(
                v=v_investimento_nf,
                depreciacao=depreciacao_capital_nf,
                gamma_retorno=config_abm[
                    "gamma_investimento_retorno"
                ],
                gamma_investimento_capacidade=config_abm[
                    "gamma_investimento_capacidade"
                ],
            )

        else:
            firma.investimento_liquido = 0.0
            firma.investimento_reposicao = 0.0
            firma.investimento_bruto = 0.0


        firma.atualizar_custo_e_preco(
            precos_insumos=pc_anterior,
            indice_salarios=indice_salarios,
            inflacao=inflacao,
        )

        firma.calcular_eob_recorrente_esperado()

        firma.calcular_dividendos()
  

    registrar_diagnostico_capacidade_setorial(
        periodo=t,
        depreciacao=depreciacao_capital_nf,
    )

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
    # PREÇO REALIZADO DO CAPITAL
    # ==========================================================

    pesos_capital = (
        pesos_bens_capital_nf
        .reindex(setores)
        .fillna(0.0)
    )

    pesos_capital = (
        pesos_capital
        / pesos_capital.sum()
    )

    preco_capital_realizado = float(
        (
            pesos_capital
            * pc_realizado.reindex(setores)
        ).sum()
    )


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

        firma.calcular_taxa_retorno_observada(
                preco_capital=preco_capital_realizado,
                depreciacao=depreciacao_capital_nf,
                taxa_juros_real=float(cfg["taxa_juros_real"]),
            )


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

            teste_flag=CONFIG["executar_testes"],
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


    taxas_retorno_observadas_periodos.append(
        {
            "periodo": t,
            "firma": firma.id,
            "setor": firma.setor,
            "regime": firma.regime,
            "preco_capital": preco_capital_realizado,
            "capital_real": firma.estoque_capital_real,
            "eob_realizado": firma.eob_misto_realizado,
            "r_obs_bruto":
                firma.taxa_retorno_bruta_observada,
            "r_obs":
                firma.taxa_retorno_observada,
        }
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


# A tabela final é uma visão derivada do histórico; os estados permanecem
# nos objetos das firmas e nos dicionários de resultados por período.
historico_df = pd.DataFrame(historico).set_index("periodo")



# ==========================================================
# RESULTADO DO TESTE CLIMÁTICO
# ==========================================================

teste_clima_df = pd.DataFrame(
    teste_clima_periodos
)
diagnostico_capacidade_setorial_df = pd.DataFrame(
    diagnostico_capacidade_setorial_periodos
)

# A tabela é o diagnóstico reproduzível da nova arquitetura: uma linha por
# setor em t=0 e em cada período da simulação. Ela fica disponível no ambiente
# do laboratório para inspeção, exportação ou gráficos sob demanda.

diagnostico_capacidade_base = diagnostico_capacidade_setorial_df.query(
    "periodo == 0"
)
diagnostico_capacidade_finito = diagnostico_capacidade_setorial_df[
    np.isfinite(
        diagnostico_capacidade_setorial_df[
            [
                "produtividade_capital_normal",
                "utilizacao_capacidade_normal",
                "producao_normal",
                "capacidade_estrutural",
                "capacidade_efetiva",
            ]
        ]
    ).all(axis=1)
]
diagnostico_capacidade_base_finito = diagnostico_capacidade_base[
    np.isfinite(diagnostico_capacidade_base["producao_normal"])
]
assert np.allclose(
    diagnostico_capacidade_base_finito["producao_normal"],
    diagnostico_capacidade_base_finito["producao_planejada"],
    atol=1e-8,
    rtol=0.0,
)
assert np.allclose(
    diagnostico_capacidade_finito["capacidade_estrutural"]
    * diagnostico_capacidade_finito["utilizacao_capacidade_normal"],
    diagnostico_capacidade_finito["producao_normal"],
    atol=1e-8,
    rtol=0.0,
)
assert np.allclose(
    diagnostico_capacidade_finito["capacidade_efetiva"],
    diagnostico_capacidade_finito["fator_clima"]
    * diagnostico_capacidade_finito["capacidade_estrutural"],
    atol=1e-8,
    rtol=0.0,
)

if not config_abm["choques_climaticos"].get("ativo", False):
    tres_periodos_sem_choque = diagnostico_capacidade_finito[
        diagnostico_capacidade_finito["periodo"].between(
            0,
            min(3, periodos),
        )
    ]
    assert np.allclose(
        tres_periodos_sem_choque["fator_clima"],
        1.0,
        atol=1e-10,
        rtol=0.0,
    )

print("✓ Diagnóstico setorial de capacidade disponível em diagnostico_capacidade_setorial_df.")

print(
    "\n================ TESTE CHOQUE CLIMÁTICO ================\n"
)


# ----------------------------------------------------------
# 1. Identidades mecânicas
# ----------------------------------------------------------

assert np.allclose(
    teste_clima_df["erro_capital_clima"],
    0.0,
    atol=1e-10,
    rtol=0.0,
)

assert np.allclose(
    teste_clima_df["erro_capacidade_estrutural"],
    0.0,
    atol=1e-8,
    rtol=0.0,
)

assert np.allclose(
    teste_clima_df["erro_capacidade_climatica"],
    0.0,
    atol=1e-8,
    rtol=0.0,
)

assert np.allclose(
    teste_clima_df["erro_producao"],
    0.0,
    atol=1e-8,
    rtol=0.0,
)

assert np.allclose(
    teste_clima_df["erro_restricao"],
    0.0,
    atol=1e-8,
    rtol=0.0,
)


print(
    "✓ Clima não altera diretamente o estoque de capital."
)

print(
    "✓ Capacidade estrutural = produtividade estrutural × K."
)

print(
    "✓ Capacidade efetiva = fator climático × capacidade estrutural."
)

print(
    "✓ Produção respeita a capacidade efetiva."
)


# ----------------------------------------------------------
# 2. Ver se algum choque foi efetivamente aplicado
# ----------------------------------------------------------

choques_aplicados = teste_clima_df[
    teste_clima_df["fator_clima"] < 1.0 - 1e-12
].copy()


print(
    "\n================ CHOQUES EFETIVAMENTE APLICADOS ================\n"
)

if choques_aplicados.empty:

    print(
        "ATENÇÃO: nenhum fator climático menor que 1 foi aplicado."
    )

else:

    print(
        choques_aplicados[
            [
                "periodo",
                "firma",
                "setor",
                "fator_clima",
                "capital",
                "capacidade_estrutural",
                "capacidade_efetiva",
                "producao_planejada",
                "producao_real",
                "restricao_capacidade",
            ]
        ].round(4)
    )


# ----------------------------------------------------------
# 3. Verificar magnitude da queda de capacidade
# ----------------------------------------------------------

if not choques_aplicados.empty:

    choques_aplicados[
        "razao_capacidade"
    ] = (
        choques_aplicados["capacidade_efetiva"]
        / choques_aplicados["capacidade_estrutural"]
    )

    erro_fator = (
        choques_aplicados["razao_capacidade"]
        - choques_aplicados["fator_clima"]
    )

    assert np.allclose(
        erro_fator,
        0.0,
        atol=1e-10,
        rtol=0.0,
    )

    print(
        "\n✓ A redução da capacidade coincide exatamente "
        "com o multiplicador climático."
    )


print(
    "\nTodos os testes do mecanismo climático passaram."
)





# ==========================================================
# PREPARAÇÃO DAS SÉRIES
# ==========================================================

# ----------------------------------------------------------
# 1. Capacidade produtiva dos setores afetados pelo clima
# ----------------------------------------------------------

setores_climaticos = list(
    config_abm[
        "choques_climaticos"
    ]["setores"].keys()
)

capacidade_setorial = (
    teste_clima_df[
        teste_clima_df["setor"].isin(setores_climaticos)
        & np.isfinite(teste_clima_df["capacidade_efetiva"])
    ]
    .groupby(["periodo", "setor"])["capacidade_efetiva"]
    .sum()
    .unstack("setor")
)

# Índice: capacidade do primeiro período = 100.
capacidade_indice = (
    capacidade_setorial
    / capacidade_setorial.iloc[0]
    * 100
)


# ----------------------------------------------------------
# 2. Crescimento do PIB real
# ----------------------------------------------------------

crescimento_producao_real = (
    historico_df["producao_real"].pct_change(fill_method=None) * 100
)


# ----------------------------------------------------------
# 3. Inflação
# ----------------------------------------------------------

inflacao_percentual = (
    historico_df["inflacao"]
    * 100
)


# ----------------------------------------------------------
# 4. Taxa de desemprego
# ----------------------------------------------------------

desemprego_percentual = (
    historico_df["taxa_desemprego"]
    * 100
)


# ==========================================================
# GRÁFICOS
# ==========================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(18, 20),
    sharex=True,
)


# ----------------------------------------------------------
# CAPACIDADE PRODUTIVA
# ----------------------------------------------------------

for setor in capacidade_indice.columns:

    nome_curto = setor.split(" - ", 1)[-1]

    axes[0].plot(
        capacidade_indice.index,
        capacidade_indice[setor],
        marker="o",
        label=nome_curto,
    )

axes[0].axhline(
    100,
    linestyle="--",
    linewidth=1,
)

axes[0].set_title(
    "Capacidade produtiva dos setores afetados pelo clima"
)

axes[0].set_ylabel(
    "Índice (primeiro período = 100)"
)

axes[0].legend()

axes[0].grid(
    alpha=0.3
)


# ----------------------------------------------------------
# CRESCIMENTO DO PIB REAL
# ----------------------------------------------------------

axes[1].plot(
    crescimento_producao_real.index,
    crescimento_producao_real,
    marker="o",
)

axes[1].axhline(
    0,
    linestyle="--",
    linewidth=1,
)

axes[1].set_title(
    "Taxa de crescimento do PIB real"
)

axes[1].set_ylabel(
    "%"
)

axes[1].grid(
    alpha=0.3
)


# ----------------------------------------------------------
# INFLAÇÃO
# ----------------------------------------------------------

axes[2].plot(
    inflacao_percentual.index,
    inflacao_percentual,
    marker="o",
)

axes[2].axhline(
    0,
    linestyle="--",
    linewidth=1,
)

axes[2].set_title(
    "Inflação"
)

axes[2].set_ylabel(
    "%"
)

axes[2].grid(
    alpha=0.3
)


# ----------------------------------------------------------
# DESEMPREGO
# ----------------------------------------------------------

axes[3].plot(
    desemprego_percentual.index,
    desemprego_percentual,
    marker="o",
)

axes[3].set_title(
    "Taxa de desemprego"
)

axes[3].set_xlabel(
    "Período"
)

axes[3].set_ylabel(
    "%"
)

axes[3].grid(
    alpha=0.3
)


# ==========================================================
# MARCAR PERÍODOS DOS CHOQUES CLIMÁTICOS
# ==========================================================

periodos_choque = sorted(
    {
        parametros["periodo_choque"]
        for parametros in config_abm[
            "choques_climaticos"
        ]["setores"].values()
    }
)

for ax in axes:

    for periodo_choque in periodos_choque:

        ax.axvline(
            periodo_choque,
            linestyle=":",
            linewidth=1.5,
        )


plt.tight_layout()

plt.show()
