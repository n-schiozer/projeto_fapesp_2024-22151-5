"""API programática de execução do modelo SFC--IO--ABM."""

from simulacao.simular_trajetoria import (
    gerar_historico_df,
    gerar_resultados_firmas_df,
    simular_trajetoria,
    simular_trajetorias,
)

__all__ = (
    "simular_trajetoria",
    "simular_trajetorias",
    "gerar_historico_df",
    "gerar_resultados_firmas_df",
)
