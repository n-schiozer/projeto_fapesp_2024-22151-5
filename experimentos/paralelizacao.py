"""Execução paralela e compacta das trajetórias de um cenário."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from multiprocessing import get_context
import os

import numpy as np
import pandas as pd

from simulacao.simular_trajetoria import simular_trajetoria


LIMITE_PROCESSOS_AUTOMATICO = 4
COLUNAS_RESULTADO = (
    "pib_real",
    "taxa_desemprego",
    "inflacao",
    "deficit_governo",
    "deficit_externo",
)

_CONTEXTO_TRABALHADOR: dict | None = None


def resolver_numero_processos(
    numero_processos: int,
    numero_simulacoes: int,
) -> int:
    """Resolve 0 como modo automático e nunca cria processos ociosos."""

    if (
        isinstance(numero_processos, bool)
        or not isinstance(numero_processos, int)
        or numero_processos < 0
    ):
        raise ValueError("numero_processos deve ser um inteiro não negativo.")
    if (
        isinstance(numero_simulacoes, bool)
        or not isinstance(numero_simulacoes, int)
        or numero_simulacoes < 1
    ):
        raise ValueError("numero_simulacoes deve ser um inteiro positivo.")
    if numero_processos == 0:
        disponiveis = os.cpu_count() or 1
        numero_processos = min(disponiveis, LIMITE_PROCESSOS_AUTOMATICO)
    return min(numero_processos, numero_simulacoes)


def _metadados_cenario(cenario: dict) -> dict:
    return {
        "cenario": cenario["nome"],
        "tipo_choque": cenario["tipo"],
        "multiplicador_produtividade": cenario[
            "multiplicador_produtividade"
        ],
        "perda_produtividade": cenario["perda_produtividade"],
        "choque_permanente": cenario["choque_permanente"],
    }


def _extrair_trajetoria(
    resultados: dict,
    *,
    simulacao: int,
    seed: int,
    metadados: dict,
) -> list[dict]:
    linhas = []
    for periodo, snapshot in resultados.items():
        macro = snapshot["macro"]
        capacidade = snapshot["financeiro"]["capacidade_financiamento"]
        pib_nominal = float(macro["pib_nominal"])
        if not np.isfinite(pib_nominal) or pib_nominal <= 0.0:
            raise ValueError(
                f"PIB nominal inválido em {metadados['cenario']}, t={periodo}."
            )
        linhas.append({
            **metadados,
            "simulacao": simulacao,
            "seed": seed,
            "periodo": periodo,
            "pib_real": float(macro["pib_real"]),
            "taxa_desemprego": float(macro["taxa_desemprego"]),
            "inflacao": float(macro["inflacao"]),
            # Necessidade de financiamento do governo: positivo é déficit.
            "deficit_governo": -float(capacidade["governo"]) / pib_nominal,
            # Superávit do resto do mundo é o déficit externo brasileiro.
            "deficit_externo": (
                float(capacidade["setor_externo"]) / pib_nominal
            ),
        })
    return linhas


def extrair_historico_macro(
    simulacoes: dict,
    *,
    cenario: dict,
) -> pd.DataFrame:
    """Extrai os cinco resultados de um lote já simulado."""

    metadados = _metadados_cenario(cenario)
    linhas = []
    for simulacao, item in simulacoes.items():
        linhas.extend(
            _extrair_trajetoria(
                item["resultados"],
                simulacao=simulacao,
                seed=item["seed"],
                metadados=metadados,
            )
        )
    historico = pd.DataFrame(linhas)
    if not np.isfinite(historico[list(COLUNAS_RESULTADO)].to_numpy()).all():
        raise ValueError(f"O cenário {cenario['nome']} produziu valor não finito.")
    return historico


def _inicializar_trabalhador(
    condicoes_iniciais: dict,
    calibracoes: dict,
    config: dict,
    config_abm: dict,
    metadados: dict,
) -> None:
    global _CONTEXTO_TRABALHADOR
    _CONTEXTO_TRABALHADOR = {
        "condicoes_iniciais": condicoes_iniciais,
        "calibracoes": calibracoes,
        "CONFIG": config,
        "CONFIG_ABM": config_abm,
        "metadados": metadados,
    }


def _executar_trajetoria_trabalhador(tarefa: tuple[int, int]) -> tuple:
    if _CONTEXTO_TRABALHADOR is None:
        raise RuntimeError("O processo trabalhador não recebeu o contexto.")
    simulacao, seed = tarefa
    # Evita milhares de mensagens de período misturadas entre processos.
    with open(os.devnull, "w", encoding="utf-8") as saida_nula:
        with redirect_stdout(saida_nula):
            resultados = simular_trajetoria(
                condicoes_iniciais=_CONTEXTO_TRABALHADOR["condicoes_iniciais"],
                calibracoes=_CONTEXTO_TRABALHADOR["calibracoes"],
                CONFIG=_CONTEXTO_TRABALHADOR["CONFIG"],
                CONFIG_ABM=_CONTEXTO_TRABALHADOR["CONFIG_ABM"],
                seed=seed,
            )
    linhas = _extrair_trajetoria(
        resultados,
        simulacao=simulacao,
        seed=seed,
        metadados=_CONTEXTO_TRABALHADOR["metadados"],
    )
    return simulacao, seed, linhas


def executar_cenario_paralelo(
    *,
    cenario: dict,
    numero_simulacoes: int,
    numero_processos: int,
    seeds: tuple[int, ...],
    condicoes_iniciais: dict,
    calibracoes: dict,
    config: dict,
) -> pd.DataFrame:
    """Executa trajetórias em processos e devolve apenas a saída macro pedida."""

    if len(seeds) != numero_simulacoes:
        raise ValueError("seeds deve possuir exatamente numero_simulacoes elementos.")
    processos = resolver_numero_processos(numero_processos, numero_simulacoes)
    metadados = _metadados_cenario(cenario)
    tarefas = tuple(enumerate(seeds, start=1))

    if processos == 1:
        linhas_ordenadas = []
        for simulacao, seed in tarefas:
            print(
                f"\n=== Simulação {simulacao}/{numero_simulacoes} "
                f"| seed={seed} ==="
            )
            resultados = simular_trajetoria(
                condicoes_iniciais=condicoes_iniciais,
                calibracoes=calibracoes,
                CONFIG=config,
                CONFIG_ABM=cenario["CONFIG_ABM"],
                seed=seed,
            )
            linhas_ordenadas.extend(
                _extrair_trajetoria(
                    resultados,
                    simulacao=simulacao,
                    seed=seed,
                    metadados=metadados,
                )
            )
        historico = pd.DataFrame(linhas_ordenadas)
        if not np.isfinite(
            historico[list(COLUNAS_RESULTADO)].to_numpy()
        ).all():
            raise ValueError(
                f"O cenário {cenario['nome']} produziu valor não finito."
            )
        return historico

    print(f"Execução paralela com {processos} processos.")
    linhas_por_simulacao = {}
    with ProcessPoolExecutor(
        max_workers=processos,
        mp_context=get_context("spawn"),
        initializer=_inicializar_trabalhador,
        initargs=(
            condicoes_iniciais,
            calibracoes,
            config,
            cenario["CONFIG_ABM"],
            metadados,
        ),
    ) as executor:
        futuros = {
            executor.submit(_executar_trajetoria_trabalhador, tarefa): tarefa
            for tarefa in tarefas
        }
        for futuro in as_completed(futuros):
            simulacao_esperada, seed_esperada = futuros[futuro]
            try:
                simulacao, seed, linhas = futuro.result()
            except Exception as erro:
                for pendente in futuros:
                    pendente.cancel()
                raise RuntimeError(
                    f"Falha na simulação {simulacao_esperada}, "
                    f"seed={seed_esperada}."
                ) from erro
            linhas_por_simulacao[simulacao] = linhas
            print(
                f"Simulação {simulacao}/{numero_simulacoes} "
                f"concluída | seed={seed}"
            )

    linhas_ordenadas = []
    for simulacao in range(1, numero_simulacoes + 1):
        linhas_ordenadas.extend(linhas_por_simulacao[simulacao])
    historico = pd.DataFrame(linhas_ordenadas)
    if not np.isfinite(historico[list(COLUNAS_RESULTADO)].to_numpy()).all():
        raise ValueError(f"O cenário {cenario['nome']} produziu valor não finito.")
    return historico.sort_values(["simulacao", "periodo"]).reset_index(drop=True)
