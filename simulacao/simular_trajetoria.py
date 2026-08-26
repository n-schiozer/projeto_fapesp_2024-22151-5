"""Orquestração programática de uma ou mais trajetórias SFC--IO--ABM."""

from __future__ import annotations

import pandas as pd

from inicializacao.inicializar_agentes import inicializar_agentes
from inicializacao.inicializar_estado import inicializar_estado
from macro.construir_saida_periodo_zero import construir_saida_periodo_zero
from macro.executar_periodo import executar_periodo
from resultados.resultados_abm import (
    concatenar_resultados_firmas,
    construir_historico_macro,
    inicializar_resultados_abm,
    registrar_resultados_periodo,
)


def simular_trajetoria(
    *,
    condicoes_iniciais,
    calibracoes,
    CONFIG,
    CONFIG_ABM,
    seed=None,
) -> dict:
    """Executa uma realização completa sem preparar ou recalibrar a base."""

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
        seed=seed,
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

    for periodo in range(1, CONFIG["periodos"] + 1):

        print(f"Período {periodo}/{CONFIG['periodos']}")

        saida_periodo = executar_periodo(
            periodo=periodo,
            firmas=firmas,
            importados=importados,
            estado=estado,
            condicoes_iniciais=condicoes_iniciais,
            calibracoes=calibracoes,
            CONFIG=CONFIG,
            CONFIG_ABM=CONFIG_ABM,
        )
        registrar_resultados_periodo(
            periodo=periodo,
            firmas=firmas,
            estado=estado,
            resultados=resultados,
            **saida_periodo,
        )

    return resultados


def simular_trajetorias(
    *,
    m,
    condicoes_iniciais,
    calibracoes,
    CONFIG,
    CONFIG_ABM,
    seeds=None,
) -> dict:
    """Executa sequencialmente ``m`` trajetórias reprodutíveis e independentes."""

    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError("m deve ser um inteiro positivo.")

    if seeds is None:
        seed_base = CONFIG_ABM["semente_qualidade"]
        seeds_trajetorias = [
            seed_base + indice
            for indice in range(m)
        ]
    else:
        seeds_trajetorias = list(seeds)

        if len(seeds_trajetorias) != m:
            raise ValueError(
                "seeds deve possuir exatamente m elementos."
            )

    simulacoes = {}

    for indice, seed in enumerate(seeds_trajetorias):

        numero_simulacao = indice + 1

        print(
            f"\n=== Simulação {numero_simulacao}/{m} "
            f"| seed={seed} ==="
        )

        resultados = simular_trajetoria(
            condicoes_iniciais=condicoes_iniciais,
            calibracoes=calibracoes,
            CONFIG=CONFIG,
            CONFIG_ABM=CONFIG_ABM,
            seed=seed,
        )

        simulacoes[numero_simulacao] = {
            "seed": seed,
            "resultados": resultados,
        }

    return simulacoes


def gerar_historico_df(simulacoes) -> pd.DataFrame:
    """Concatena as visões macro existentes, identificando cada trajetória."""

    tabelas: list[pd.DataFrame] = []
    for simulacao, item in simulacoes.items():
        tabela = construir_historico_macro(item["resultados"]).copy()
        tabela.insert(0, "simulacao", simulacao)
        tabela.insert(1, "seed", item["seed"])
        tabelas.append(tabela)
    return pd.concat(tabelas) if tabelas else pd.DataFrame()


def gerar_resultados_firmas_df(simulacoes) -> pd.DataFrame:
    """Concatena as visões existentes das firmas por trajetória."""

    tabelas: list[pd.DataFrame] = []
    for simulacao, item in simulacoes.items():
        tabela = concatenar_resultados_firmas(item["resultados"]).copy()
        tabela.insert(0, "simulacao", simulacao)
        tabela.insert(1, "seed", item["seed"])
        tabelas.append(tabela)
    return pd.concat(tabelas, ignore_index=True) if tabelas else pd.DataFrame()
