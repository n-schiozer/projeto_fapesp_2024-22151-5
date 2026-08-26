"""Ligação entre theta, o simulador completo, Monte Carlo e os momentos."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from black_it.loss_functions.base import BaseLoss

from calibracao.blackit.parametros import aplicar_theta
from calibracao.empirica.moments import calculate_moments, cross_correlation
from resultados.resultados_abm import construir_historico_macro
from simulacao.simular_trajetoria import simular_trajetorias


ORDEM_MOMENTOS = (
    "inflacao__mean",
    "inflacao__std",
    "inflacao__acf1",
    "crescimento_pib_real__mean",
    "crescimento_pib_real__std",
    "crescimento_pib_real__acf1",
    "taxa_desemprego__mean",
    "taxa_desemprego__std",
    "taxa_desemprego__acf1",
    "crescimento_populacao__mean",
    "inflacao_desemprego_defasado__corr",
)


class PerdaQuadraticaNormalizada(BaseLoss):
    """J(theta): soma dos quadrados dos erros já normalizados."""

    def compute_loss_1d(self, sim_data_ensemble, real_data) -> float:
        error = sim_data_ensemble.mean(axis=0) - real_data
        return float(np.square(error).sum())


def momentos_trajetoria(resultados: dict) -> dict[str, float]:
    historico = construir_historico_macro(resultados).sort_index()
    if list(historico.index) != list(range(16)):
        raise ValueError("Cada trajetória da calibração deve conter t=0,...,15.")
    pib = pd.to_numeric(historico["pib_real"], errors="coerce")
    if not np.isfinite(pib).all():
        raise ValueError("pib_real contém valor não finito em t=0,...,15.")

    inflacao = pd.to_numeric(historico.loc[1:15, "inflacao"], errors="coerce")
    crescimento_pib = pib.pct_change(fill_method=None).loc[1:15]
    desemprego = pd.to_numeric(
        historico.loc[1:15, "taxa_desemprego"], errors="coerce"
    )
    crescimento_populacao = pd.to_numeric(
        historico["indice_populacao"], errors="coerce"
    ).pct_change(fill_method=None).loc[1:15]

    def tres(prefixo: str, serie: pd.Series) -> dict[str, float]:
        calculados = calculate_moments(serie)
        return {
            f"{prefixo}__mean": float(calculados["mean"]),
            f"{prefixo}__std": float(calculados["std"]),
            f"{prefixo}__acf1": float(calculados["autocorrelation_lag1"]),
        }

    valores = {
        **tres("inflacao", inflacao),
        **tres("crescimento_pib_real", crescimento_pib),
        **tres("taxa_desemprego", desemprego),
        "crescimento_populacao__mean": float(crescimento_populacao.mean()),
        "inflacao_desemprego_defasado__corr": cross_correlation(
            inflacao,
            desemprego,
            lag_second=1,
        ),
    }
    if not np.isfinite(list(valores.values())).all():
        raise ValueError("A trajetória produziu momento não finito.")
    return valores


@dataclass
class ModeloCalibracao:
    condicoes_iniciais: dict
    calibracoes: dict
    CONFIG: dict
    CONFIG_ABM: dict
    targets: pd.DataFrame
    seeds: tuple[int, ...]

    def avaliar(self, theta) -> tuple[np.ndarray, list[dict[str, float]]]:
        config, config_abm = aplicar_theta(
            theta,
            self.CONFIG,
            self.CONFIG_ABM,
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            simulacoes = simular_trajetorias(
                m=len(self.seeds),
                condicoes_iniciais=self.condicoes_iniciais,
                calibracoes=self.calibracoes,
                CONFIG=config,
                CONFIG_ABM=config_abm,
                seeds=self.seeds,
            )
        individuais = [
            momentos_trajetoria(item["resultados"])
            for item in simulacoes.values()
        ]
        medias = {
            nome: float(np.mean([item[nome] for item in individuais]))
            for nome in ORDEM_MOMENTOS
        }
        alvo = self.targets.set_index("moment_id").loc[list(ORDEM_MOMENTOS)]
        erros = np.array(
            [
                (medias[nome] - float(alvo.loc[nome, "value"]))
                / float(alvo.loc[nome, "scale"])
                for nome in ORDEM_MOMENTOS
            ],
            dtype=float,
        )
        return erros, individuais

    def __call__(self, theta, N, seed):
        del seed
        if N != 1:
            raise ValueError("O adaptador Black-it usa N=1 vetor de momentos.")
        erros, _ = self.avaliar(theta)
        return erros.reshape(1, -1)


def validar_monte_carlo(
    individuais: list[dict[str, float]],
    numero_esperado: int,
) -> None:
    """Confirma que os momentos foram calculados antes da média entre runs."""
    if len(individuais) != numero_esperado:
        raise AssertionError(
            f"Esperadas {numero_esperado} trajetórias; recebidas {len(individuais)}."
        )
    for nome in ORDEM_MOMENTOS:
        media = np.mean([trajetoria[nome] for trajetoria in individuais])
        if not np.isfinite(media):
            raise AssertionError(f"Momento MC não finito: {nome}")
