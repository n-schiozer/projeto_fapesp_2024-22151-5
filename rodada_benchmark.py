"""Modelo-base editável: benchmark SFC--IO--ABM Brasil.

Edite ``CONFIG`` e ``CONFIG_ABM`` abaixo e execute este arquivo. Com
``USAR_PARAMETROS_CALIBRADOS=True``, os parâmetros estimados são aplicados
sobre essas hipóteses. Defina-o como ``False`` para usar literalmente todos
os valores declarados.
"""

from pathlib import Path
import sys


# Em IDEs que mantêm outros projetos no ``sys.path`` (por exemplo, Spyder e
# Positron), prioriza explicitamente esta cópia do repositório.
PASTA_PROJETO = Path(__file__).resolve().parent
if str(PASTA_PROJETO) in sys.path:
    sys.path.remove(str(PASTA_PROJETO))
sys.path.insert(0, str(PASTA_PROJETO))

# O caminho acima resolve importações futuras. Esta limpeza complementar evita
# reutilizar módulos de outro checkout que já tenham sido importados pelo
# kernel atual da IDE.
PACOTES_DO_MODELO = (
    "agentes",
    "calibracao",
    "contabilidade",
    "demografia",
    "diagnosticos",
    "experimentos",
    "financeiro",
    "inicializacao",
    "investimento",
    "macro",
    "mercados",
    "modelos",
    "resultados",
    "simulacao",
)
for nome, modulo in tuple(sys.modules.items()):
    if (
        nome.split(".", 1)[0] not in PACOTES_DO_MODELO
        and nome != "configuracao_projeto"
    ):
        continue
    arquivo_modulo = getattr(modulo, "__file__", None)
    if arquivo_modulo and not Path(arquivo_modulo).resolve().is_relative_to(PASTA_PROJETO):
        del sys.modules[nome]

import matplotlib.pyplot as plt
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

from configuracao_projeto import DEMOGRAFIA_RAW_DIR
from experimentos.laboratorio_benchmark_calibrado import executar_laboratorio


# =============================================================================
# CONTROLES DA RODADA — edite livremente
# =============================================================================

SEMENTE = 42
USAR_PARAMETROS_CALIBRADOS = True
FORCAR_BENCHMARK_SEM_CHOQUE = True


# =============================================================================
# HIPÓTESES MACROECONÔMICAS — edite livremente
# =============================================================================

CONFIG = {
    "ano": 2020,
    "nivel": 20,
    "aba_cei": "Python",
    "periodos": 25,
    "multiplicador_governo": 1.0,
    "multiplicador_investimento": 1.0,
    "multiplicador_exportacoes": 1.0,
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
    "a0": 0.03,
    "a1": 0.2,
    "a3": 0.2,
    "repasse_inflacao_cambio": 1.0,
    "taxa_juros_real": 0.06,
    "inertia_pm": 0.5,
    "fracao_reavaliacao_financeira": 1.0,
    "tolerancia_consumo": 1e-6,
    "max_iteracoes_consumo": 100,
    "executar_testes": False,
}


# =============================================================================
# FIRMAS, MERCADOS, DEMOGRAFIA E CHOQUES — edite livremente
# =============================================================================

SETOR_FINANCEIRO = "K - Atividades financeiras, de seguros e serviços relacionados"
SETORES_LEILAO = [
    "A - Agricultura, pecuária, produção florestal, pesca e aquicultura",
    "D - Eletricidade e gás",
]
SETORES_REGULADOS = ["D - Eletricidade e gás"]

CONFIG_ABM = {
    "usar_demografia_empresas": False,
    "numero_firmas_industria": 20,
    "numero_firmas_leilao": 20,
    "numero_firmas_por_setor": {SETOR_FINANCEIRO: 1},
    "setores_leilao": SETORES_LEILAO,
    "setores_regulados": SETORES_REGULADOS,
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
        "tamanho_coorte": 1000,
    },
    "parametros_markup": {
        "parametro_markup": 0.1,
        "markup_min": 0.0,
        "markup_max": 10.0,
        "epsilon_market_share": 1e-12,
    },
    "multiplicador_capacidade_importada": 1.5,
    "velocidade_ajuste_expectativa_demanda": 0.50,
    "velocidade_ajuste_estoques_firmas": 0.25,
    "lambda_expectativa_precos": 1.0,
    "adj_r_obs_inicial": 1,
    "utilizacao_capacidade_normal": 0.80,
    "gamma_investimento_retorno": 0.5,
    "gamma_investimento_capacidade": 0.5,
    "choques_climaticos": {
        "ativo": False,
        "setores": {
            SETORES_LEILAO[0]: {
                "periodo_choque": 5,
                "multiplicador_produtividade": 0.95,
                "choque_permanente": False,
            },
            SETORES_LEILAO[1]: {
                "periodo_choque": 5,
                "multiplicador_produtividade": 0.95,
                "choque_permanente": False,
            },
        },
    },
}


print("Preparando a rodada benchmark. Aguarde...", flush=True)
resultado = executar_laboratorio(
    seed=SEMENTE,
    config=CONFIG,
    config_abm=CONFIG_ABM,
    usar_parametros_calibrados=USAR_PARAMETROS_CALIBRADOS,
    forcar_benchmark_sem_choque=FORCAR_BENCHMARK_SEM_CHOQUE,
    mostrar_graficos=False,
)
historico = resultado["historico"]
resultados = resultado["resultados"]
figuras = resultado["figuras"]

# Tabelas disponíveis no painel Variables/Data Explorer do Positron.
tabela_macro = historico[[
    "periodo",
    "pib_real",
    "taxa_desemprego",
    "inflacao",
    "deficit_governo",
    "deficit_externo",
]].copy()
periodo_final = max(resultados)
tabela_setores = pd.DataFrame.from_dict(
    resultados[periodo_final]["setores"],
    orient="index",
)
tabela_setores.index.name = "setor"

print("\nTABELA MACROECONÔMICA")
print(tabela_macro.to_string(index=False))
print("\nTABELA SETORIAL — ÚLTIMO PERÍODO")
print(
    "Tabela completa disponível em `tabela_setores` no painel Variables/Data Explorer."
)
print(
    tabela_setores.iloc[:, :8].to_string(
        max_rows=25,
        max_cols=8,
    )
)
print("\nGráficos disponíveis na aba Plots.")

# Em uma sessão interativa, o Positron também renderiza as tabelas abaixo.
display(tabela_macro)
display(tabela_setores)

plt.show()
