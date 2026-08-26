"""Ponto de entrada organizado da versão SFC--IO--ABM.

O arquivo não reimplementa a economia: ele apenas configura os dados, prepara
o ano-base e chama o ciclo temporal já implementado nos módulos específicos.
Assim, pode ser executado diretamente ou usado célula a célula em um notebook.
"""

from configuracao_projeto import (
    ARQUIVO_CEI,
    DATA_DIR,
    validar_caminhos_dados,
)
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais
from macro.simulacao_cei_2 import simul_
from tests.unit.testar_mercado_leilao_etapa11 import testar_mercado_leilao_etapa11


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
    "periodos": 1,
    "multiplicador_governo": 1.0,
    "multiplicador_investimento": 1.0,
    "multiplicador_exportacoes": 1.0,
    # Mesmo no cenário sem choque, o período precisa ser válido para simul_().
    "periodo_choque": 1,
    "choque_permanente": True,
    "taxa_desemprego_base": 0.138,
    "taxa_desemprego_inicial": 0.138,
    "taxa_crescimento_populacional": 0.0,
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
    "numero_firmas_industria": 100,
    "numero_firmas_leilao": 100,
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
    "semente_exposicao_climatica": 202604,
    "usar_heterogeneidade_tecnologica": True,
    "peso_relativo_ci_eletricidade_exposta": 0.1,
    "probabilidades_exposicao_climatica": {
        SETORES_LEILAO[0]: 0.90,
        SETORES_LEILAO[1]: 0.50,
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
    "velocidade_ajuste_estoques_firmas": 0.25,
    "lambda_expectativa_precos": 1.0,
    "gamma_investimento_retorno": 0.5,
    "gamma_investimento_capacidade": 0.5,
    # Mude para True somente quando quiser imprimir toda a bateria de regressões.
    "executar_testes": False,
}


# %% =====================================================================
# 4. FUNÇÕES DE EXECUÇÃO
# ========================================================================

def executar_simulacao() -> tuple[dict, dict]:
    """Prepara os dados-base e executa o ciclo temporal do ABM.

    ``preparar_condicoes_iniciais`` lê TRU e CEI, calcula a tecnologia, as
    razões de valor adicionado, preços-base, parâmetros institucionais e a
    calibração do investimento. Não realiza uma simulação temporal.

    ``simul_`` inicializa as firmas uma única vez e executa os períodos. Em
    cada período ela fixa produção ex ante, atualiza preços, constrói a CEI e
    realiza os mercados industriais e de leilão.
    """

    data_dir, arquivo_cei = validar_caminhos_dados(DATA_DIR, ARQUIVO_CEI)
    condicoes_iniciais = preparar_condicoes_iniciais(
        CONFIG,
        data_dir,
        arquivo_cei,
    )

    resultado = simul_(CONFIG["periodos"], condicoes_iniciais, CONFIG_ABM)

    return condicoes_iniciais, resultado


def imprimir_resumo(resultado: dict) -> None:
    """Mostra os principais indicadores já calculados por ``simul_``."""

    colunas = [
        "indice_precos",
        "inflacao",
        "pib_real",
        "taxa_desemprego",
        "deficit_governo",
        "saldo_setor_externo",
        "discrepancia_cei",
    ]
    print(resultado["historico"][colunas])


def executar_testes(condicoes_iniciais: dict) -> None:
    """Executa as identidades do ano-base e regressões encadeadas até a Etapa 11.

    ``testar_mercado_leilao_etapa11`` também chama as regressões das etapas
    anteriores, verificando TRU-base, CEI, preços, multilogit e leilões.
    """

    print("\nTESTES DA ETAPA 11")
    print(testar_mercado_leilao_etapa11(condicoes_iniciais).to_string())


# %% =====================================================================
# 5. EXECUÇÃO DIRETA
# ========================================================================
# Em notebook, execute manualmente:
#
# condicoes_iniciais, resultado = executar_simulacao()
# imprimir_resumo(resultado)
#
# Ao rodar este .py diretamente, o bloco abaixo faz exatamente isso e deixa
# ``condicoes_iniciais`` e ``resultado`` disponíveis no escopo do script.

if __name__ == "__main__":
    condicoes_iniciais, resultado = executar_simulacao()
    imprimir_resumo(resultado)
    if CONFIG_ABM["executar_testes"]:
        executar_testes(condicoes_iniciais)
