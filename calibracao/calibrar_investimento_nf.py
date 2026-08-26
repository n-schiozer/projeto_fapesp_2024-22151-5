# %%
"""Execução auditável da calibração do investimento das firmas NF."""

import pandas as pd

from calibracao.calibracao_investimento_nf import calibrar_investimento_nf
from configuracao_projeto import ARQUIVO_CEI, DATA_DIR


# %% ======================================================================
# 1. ARQUIVOS E HIPÓTESES
# =========================================================================

ANO_BASE = 2020
NIVEL_TRU = 20
ABA_CEI = "Python"
VIDA_UTIL_CAPITAL = 20.0
V_REFERENCIA = 4.5
ANO_INICIAL_BETA = 2010
ANO_FINAL_BETA = 2019  # último ano antes da pandemia


# %% ======================================================================
# 2. CALIBRAÇÃO
# =========================================================================

resultado_calibracao_investimento_nf = calibrar_investimento_nf(
    DATA_DIR,
    ARQUIVO_CEI,
    ano_base=ANO_BASE,
    nivel=NIVEL_TRU,
    aba_cei=ABA_CEI,
    vida_util_capital=VIDA_UTIL_CAPITAL,
    ano_inicial_beta=ANO_INICIAL_BETA,
    ano_final_beta=ANO_FINAL_BETA,
)

beta_calibrado = resultado_calibracao_investimento_nf["beta"]
v_calibrado = resultado_calibracao_investimento_nf["v"]
depreciacao = resultado_calibracao_investimento_nf["depreciacao"]
producao_real_2020 = resultado_calibracao_investimento_nf["producao_real"]
estoque_capital_nf_2020 = resultado_calibracao_investimento_nf[
    "estoque_capital_nf_base"
]
investimento_nf_por_setor_investidor_2020 = (
    resultado_calibracao_investimento_nf[
        "investimento_nf_base_por_investidor"
    ]
)
pesos_bens_capital_nf = resultado_calibracao_investimento_nf[
    "pesos_bens_capital_nf"
]
fbcf_nf_por_setor_fornecedor_2020 = resultado_calibracao_investimento_nf[
    "fbcf_nf_base_fornecedor"
]
fbcf_outros_autonoma_2020 = resultado_calibracao_investimento_nf[
    "fbcf_outros_base"
]


# %% ======================================================================
# 3. RESULTADOS
# =========================================================================

diagnosticos = resultado_calibracao_investimento_nf["diagnosticos"].copy()
diagnosticos.loc["v_referencia"] = V_REFERENCIA
diagnosticos.loc["fbcf_prevista_com_v_4_5"] = (
    diagnosticos.loc["fbcf_nf_cei"] * V_REFERENCIA / v_calibrado
)

print("\nCALIBRAÇÃO DO INVESTIMENTO DAS FIRMAS NÃO FINANCEIRAS")
print(diagnosticos.to_string())

print("\nESTOQUE E INVESTIMENTO POR SETOR INVESTIDOR")
print(
    pd.concat(
        [
            estoque_capital_nf_2020,
            investimento_nf_por_setor_investidor_2020,
        ],
        axis="columns",
    ).to_string()
)

print("\nDISTRIBUIÇÃO DA FBCF NF ENTRE OS SETORES FORNECEDORES")
print(
    pd.concat(
        [pesos_bens_capital_nf, fbcf_nf_por_setor_fornecedor_2020],
        axis="columns",
    ).to_string()
)
