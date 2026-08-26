"""Laboratório SFC--IO--ABM limpo e auditável.

Preserva o ciclo econômico da referência e remove somente caminhos
legados, testes manuais, gráficos e diagnósticos temporários. As unidades
são explícitas: quantidade real, preço básico (PB) e preço de comprador
(PM/Pc) não são intercambiados.
"""

from configuracao_projeto import (
    ARQUIVO_CEI,
    DATA_DIR,
    DEMOGRAFIA_RAW_DIR,
    validar_caminhos_dados,
)
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais

"""Três blocos auditáveis: TRU, CEI e ciclo temporal."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inicializacao.inicializar_agentes import inicializar_agentes
from inicializacao.inicializar_estado import inicializar_estado
from macro.construir_saida_periodo_zero import construir_saida_periodo_zero
from macro.executar_periodo import executar_periodo

from resultados.resultados_abm import (
    concatenar_diagnostico,
    concatenar_resultados_firmas,
    construir_historico_macro,
    inicializar_resultados_abm,
    registrar_resultados_periodo,
)

from calibracao.calibrar_modelo import calibrar_modelo
from diagnosticos.imutabilidade_fontes import (
    capturar_referencias,
    verificar_fontes_inalteradas,
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
    "taxa_crescimento_populacional": 0.0,
    "taxa_crescimento_demanda_autonoma": 0.0,
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
    # False: abertura simples, sem Demografia. True: coortes demográficas.
    "usar_demografia_empresas": True,
    "numero_firmas_industria": 20,
    "numero_firmas_leilao": 20,
    "numero_firmas_por_setor": {SETOR_FINANCEIRO: 1},
    "setores_leilao": SETORES_LEILAO,
    # Regulação e regime de mercado são conceitos independentes.
    "setores_regulados": SETORES_REGULADOS,
    # Em cenários climáticos, distribui o alvo setorial por qualidade e preço;
    # a capacidade ociosa permanece disponível como reserva operacional.
    "eta_preco_padrao": -1.2,
    "eta_qualidade_padrao": 2.0,
    "eta_atendimento_padrao": 1.0,
    "parametro_estoque_desejado": 0.0978561253333731,
    "ajustes_setoriais": {},
    "market_shares_domesticos": {},
    "precos_relativos_iniciais": {},
    "rho_qualidade": 0.90,
    "sigma_qualidade": 0.02,
    "rho_produtividade_idiossincratica": 0.90,
    "sigma_produtividade_idiossincratica": 0.02,
    "semente_qualidade": 42,
    "semente_exposicao_climatica": 202604,
    "usar_heterogeneidade_tecnologica": True,
    "peso_relativo_ci_eletricidade_exposta": 0.1,
    "probabilidades_exposicao_climatica": {
        SETORES_LEILAO[0]: 0.90,
        SETORES_LEILAO[1]: 0.50,
    },
    "demografia_empresas": {
        "arquivo": DEMOGRAFIA_RAW_DIR / "Demografia_Empresas.xlsx",
        "aba": "Planilha1",
        "distribuicao": "pareto",
        "semente": 42,
        "tamanho_coorte": 10000,
    },
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
    "velocidade_ajuste_expectativa_demanda": 0.50,
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

referencias_immutabilidade = capturar_referencias(
    {
        "CONFIG": CONFIG,
        "CONFIG_ABM": CONFIG_ABM,
    }
)

condicoes_iniciais = preparar_condicoes_iniciais(
    CONFIG,
    DATA_DIR,
    ARQUIVO_CEI,
)

referencias_immutabilidade["condicoes_iniciais"] = capturar_referencias(
    {"condicoes_iniciais": condicoes_iniciais}
)["condicoes_iniciais"]

calibracoes = calibrar_modelo(
    condicoes_iniciais=condicoes_iniciais,
    CONFIG=CONFIG,
    CONFIG_ABM=CONFIG_ABM,
)
referencias_immutabilidade["calibracoes"] = capturar_referencias(
    {"calibracoes": calibracoes}
)["calibracoes"]

setores = list(condicoes_iniciais["setores"])
setores_regulados_desconhecidos = set(
    CONFIG_ABM["setores_regulados"]
).difference(setores)
if setores_regulados_desconhecidos:
    raise ValueError(
        "setores_regulados contém setores inexistentes: "
        f"{sorted(setores_regulados_desconhecidos)}"
    )
velocidade_ajuste_estoques_firmas = float(CONFIG_ABM.get("velocidade_ajuste_estoques_firmas", CONFIG["velocidade_ajuste_estoques"]))
if velocidade_ajuste_estoques_firmas < 0.0:
    raise ValueError("velocidade_ajuste_estoques_firmas não pode ser negativa.")
# Abertura única antes do FOR temporal: os mesmos objetos sobreviverão a
# todos os períodos. Produção, preço e capital são definidos pelos objetos.

firmas, importados = inicializar_agentes(
    condicoes_iniciais=condicoes_iniciais,
    calibracoes=calibracoes,
    CONFIG=CONFIG,
    CONFIG_ABM=CONFIG_ABM,
)

estado = inicializar_estado(
    condicoes_iniciais=condicoes_iniciais,
    calibracoes=calibracoes,
    firmas=firmas,
    CONFIG=CONFIG,
    CONFIG_ABM=CONFIG_ABM,
)


resultados = inicializar_resultados_abm()

saida_zero = construir_saida_periodo_zero(
    firmas=firmas,
    importados=importados,
    estado=estado,
    condicoes_iniciais=condicoes_iniciais,
    calibracoes=calibracoes,
    CONFIG=CONFIG,
    CONFIG_ABM=CONFIG_ABM,
)

registrar_resultados_periodo(
    periodo=0,
    firmas=firmas,
    estado=estado,
    resultados=resultados,
    **saida_zero,
)

# ============================================================================
# 7. CICLO TEMPORAL: decisões, mercados, realização e estado herdado
# ============================================================================

# Cada período preserva a causalidade: expectativa, decisão, mercado,
# realização, CEI e atualização do estado para o período seguinte.

for t in range(1, CONFIG["periodos"] + 1):
    saida_periodo = executar_periodo(
        periodo=t,
        firmas=firmas,
        importados=importados,
        estado=estado,
        condicoes_iniciais=condicoes_iniciais,
        calibracoes=calibracoes,
        CONFIG=CONFIG,
        CONFIG_ABM=CONFIG_ABM,
    )
    registrar_resultados_periodo(
        periodo=t,
        firmas=firmas,
        estado=estado,
        resultados=resultados,
        **saida_periodo,
    )

resultado_teste_immutabilidade_fontes = verificar_fontes_inalteradas(
    referencias_immutabilidade,
    {
        "CONFIG": CONFIG,
        "CONFIG_ABM": CONFIG_ABM,
        "condicoes_iniciais": condicoes_iniciais,
        "calibracoes": calibracoes,
    },
)
print(
    "Fontes e calibrações permaneceram profundamente inalteradas: "
    f"{resultado_teste_immutabilidade_fontes}."
)


# As tabelas abaixo são somente views finais. A fonte única é ``resultados``.
historico_df = construir_historico_macro(resultados)

resultados_firmas_df = concatenar_resultados_firmas(resultados)
qualidades_firmas_df = resultados_firmas_df.loc[
    :,
    [
        "periodo",
        "firma",
        "setor",
        "qualidade_base",
        "desvio_qualidade",
        "qualidade",
    ],
].copy()

taxas_retorno_observadas_df = pd.concat(
    [
        dados if periodo == 0 else dados.tail(1)
        for periodo, dados in resultados_firmas_df.groupby(
            "periodo",
            sort=False,
        )
    ],
    ignore_index=True,
).loc[
    :,
    [
        "periodo",
        "firma",
        "setor",
        "regime",
        "preco_capital",
        "capital_real",
        "eob_realizado",
        "r_obs_bruto",
        "r_obs",
    ],
]

# ==========================================================
# RESULTADO DO TESTE CLIMÁTICO
# ==========================================================

teste_clima_df = concatenar_diagnostico(resultados, "clima")
teste_regulacao_df = concatenar_diagnostico(resultados, "regulacao")
diagnostico_capacidade_setorial_df = concatenar_diagnostico(
    resultados,
    "capacidade_setorial",
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
    diagnostico_capacidade_finito["fator_capacidade_total"]
    * diagnostico_capacidade_finito["capacidade_estrutural"],
    atol=1e-8,
    rtol=0.0,
)

if not CONFIG_ABM["choques_climaticos"].get("ativo", False):
    tres_periodos_sem_choque = diagnostico_capacidade_finito[
        diagnostico_capacidade_finito["periodo"].between(
            0,
            min(3, CONFIG["periodos"]),
        )
    ]
    assert np.allclose(
        tres_periodos_sem_choque["fator_clima"],
        1.0,
        atol=1e-10,
        rtol=0.0,
    )

print("Diagnóstico setorial de capacidade disponível em diagnostico_capacidade_setorial_df.")

print(
    "\n================ TESTE CHOQUE CLIMÁTICO ================\n"
)


# As identidades de implementação são verificadas a partir dos resultados
# econômicos, sem carregar colunas de erro no runtime de executar_periodo().
clima_finito = teste_clima_df[
    np.isfinite(teste_clima_df["capacidade_efetiva"])
]
assert np.allclose(
    clima_finito["capacidade_efetiva"],
    clima_finito["fator_clima"]
    * clima_finito["fator_produtividade_idiossincratica"]
    * clima_finito["capacidade_estrutural"],
    atol=1e-8,
    rtol=0.0,
)
producao_esperada = np.minimum(
    clima_finito["producao_planejada"],
    clima_finito["capacidade_efetiva"],
)
assert np.allclose(
    clima_finito["producao_real"],
    producao_esperada,
    atol=1e-8,
    rtol=0.0,
)
assert np.allclose(
    clima_finito["restricao_capacidade"],
    np.maximum(
        0.0,
        clima_finito["producao_planejada"]
        - clima_finito["capacidade_efetiva"],
    ),
    atol=1e-8,
    rtol=0.0,
)


print(
    "Clima não altera diretamente o estoque de capital."
)

print(
    "Capacidade estrutural = produtividade estrutural × K."
)

print(
    "Capacidade efetiva = fator climático × capacidade estrutural."
)

print(
    "Produção respeita a capacidade efetiva."
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
        * choques_aplicados["fator_produtividade_idiossincratica"]
    )

    assert np.allclose(
        erro_fator,
        0.0,
        atol=1e-10,
        rtol=0.0,
    )

    print(
        "\nA redução da capacidade coincide exatamente "
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
    CONFIG_ABM[
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
        for parametros in CONFIG_ABM[
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
