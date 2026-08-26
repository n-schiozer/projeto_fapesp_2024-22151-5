"""Verifica imutabilidade profunda das fontes e calibrações do laboratório."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd


def capturar_referencias(fontes: Mapping[str, Any]) -> dict[str, Any]:
    """Cria um retrato profundo independente das estruturas informadas."""

    return deepcopy(dict(fontes))


def _comparar_exato(esperado: Any, observado: Any, caminho: str) -> None:
    if type(esperado) is not type(observado):
        raise AssertionError(
            f"{caminho}: tipo alterado de {type(esperado).__name__} "
            f"para {type(observado).__name__}."
        )

    if isinstance(esperado, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(esperado, observado, check_exact=True)
        except AssertionError as erro:
            raise AssertionError(f"{caminho}: DataFrame alterado. {erro}") from erro
        return

    if isinstance(esperado, pd.Series):
        try:
            pd.testing.assert_series_equal(esperado, observado, check_exact=True)
        except AssertionError as erro:
            raise AssertionError(f"{caminho}: Series alterada. {erro}") from erro
        return

    if isinstance(esperado, pd.Index):
        try:
            pd.testing.assert_index_equal(esperado, observado, exact=True)
        except AssertionError as erro:
            raise AssertionError(f"{caminho}: Index alterado. {erro}") from erro
        return

    if isinstance(esperado, np.ndarray):
        try:
            np.testing.assert_array_equal(esperado, observado, strict=True)
        except AssertionError as erro:
            raise AssertionError(f"{caminho}: ndarray alterado. {erro}") from erro
        return

    if isinstance(esperado, dict):
        if list(esperado) != list(observado):
            raise AssertionError(f"{caminho}: chaves ou ordem das chaves alteradas.")
        for chave in esperado:
            _comparar_exato(
                esperado[chave],
                observado[chave],
                f"{caminho}[{chave!r}]",
            )
        return

    if isinstance(esperado, (list, tuple)):
        if len(esperado) != len(observado):
            raise AssertionError(f"{caminho}: tamanho alterado.")
        for indice, (item_esperado, item_observado) in enumerate(
            zip(esperado, observado, strict=True)
        ):
            _comparar_exato(
                item_esperado,
                item_observado,
                f"{caminho}[{indice}]",
            )
        return

    if isinstance(esperado, (set, frozenset)):
        if esperado != observado:
            raise AssertionError(f"{caminho}: conjunto alterado.")
        return

    if hasattr(esperado, "__dict__"):
        _comparar_exato(vars(esperado), vars(observado), f"{caminho}.__dict__")
        return

    if isinstance(esperado, (float, np.floating)):
        if np.isnan(esperado) and np.isnan(observado):
            return

    igualdade = esperado == observado
    if isinstance(igualdade, (np.ndarray, pd.Series, pd.DataFrame)):
        igualdade = bool(np.asarray(igualdade).all())
    if not bool(igualdade):
        raise AssertionError(
            f"{caminho}: valor alterado de {esperado!r} para {observado!r}."
        )


def verificar_fontes_inalteradas(
    referencias: Mapping[str, Any],
    fontes_atuais: Mapping[str, Any],
) -> dict[str, str]:
    """Compara todas as fontes profundamente e devolve um status auditável."""

    if list(referencias) != list(fontes_atuais):
        raise AssertionError("O conjunto de fontes submetidas ao teste foi alterado.")

    resultado = {}
    for nome in referencias:
        _comparar_exato(referencias[nome], fontes_atuais[nome], nome)
        resultado[nome] = "PASS"
    return resultado
