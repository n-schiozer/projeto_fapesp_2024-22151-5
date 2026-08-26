"""Roda o benchmark calibrado e mostra gráficos e tabelas no Positron."""

import matplotlib.pyplot as plt
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

from experimentos.laboratorio_benchmark_calibrado import executar_laboratorio


print("Preparando a rodada benchmark calibrada. Aguarde...", flush=True)
resultado = executar_laboratorio(mostrar_graficos=False)
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
print(tabela_setores.to_string())
print("\nGráficos disponíveis na aba Plots.")

# Em uma sessão interativa, o Positron também renderiza as tabelas abaixo.
display(tabela_macro)
display(tabela_setores)

plt.show()
