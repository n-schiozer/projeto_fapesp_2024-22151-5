"""Snapshots históricos organizados diretamente pelo período econômico."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd


CATEGORIAS_RESULTADOS = (
    "macro", "setores", "firmas", "cei", "financeiro", "diagnosticos"
)


def inicializar_resultados_abm() -> dict:
    """Cria o histórico vazio; cada chave futura será um período."""

    return {}


_ATRIBUTOS_FIRMAS = (
    "demanda_esperada", "expectativa_vendas_real",
    "producao_desejada_real", "producao_planejada_real",
    "producao_real", "vendas_real", "vendas_nominal",
    "demanda_recebida_real", "demanda_nao_atendida_real",
    "producao_nao_vendida_real", "excedente_nao_estocavel_real",
    "estoque", "variacao_estoque_real", "estoque_capital_real",
    "capital_desejado", "gap_capital", "investimento_liquido",
    "investimento_reposicao", "investimento_bruto", "preco_firma",
    "preco_transacao", "market_share_desejado", "market_share_realizado",
    "qualidade_base", "desvio_qualidade", "qualidade",
    "eob_misto_realizado", "taxa_retorno_bruta_observada",
    "taxa_retorno_observada", "taxa_retorno_ajustada",
    "demanda_trabalho", "exposicao_climatica", "peso_relativo_ci",
    "custo_intermediario_unitario", "custo_unitario", "markup",
    "preco_oferta_leilao",
    "fator_produtividade_climatica",
    "fator_produtividade_idiossincratica",
    "desvio_produtividade_idiossincratica",
    "producao_normal_real", "capacidade_produtiva_estrutural_real",
    "capacidade_produtiva_real", "taxa_utilizacao_capacidade",
    "valor_producao_nominal_realizado",
    "consumo_intermediario_nominal_realizado",
    "valor_adicionado_realizado",
)

# Aliases mantidos somente na view DataFrame pública para compatibilidade.
_ALIASES_VIEW_FIRMAS = {
    "estoque_capital_real": "capital_real",
    "eob_misto_realizado": "eob_realizado",
    "taxa_retorno_bruta_observada": "r_obs_bruto",
    "taxa_retorno_observada": "r_obs",
    "taxa_retorno_ajustada": "r_ajustado",
    "fator_produtividade_climatica": "fator_clima",
    "capacidade_produtiva_estrutural_real": "capacidade_estrutural_real",
    "capacidade_produtiva_real": "capacidade_efetiva_real",
    "valor_producao_nominal_realizado": "valor_producao_nominal",
    "consumo_intermediario_nominal_realizado": "consumo_intermediario_nominal",
    "valor_adicionado_realizado": "valor_adicionado_nominal",
}


def _snapshot_firmas(
    firmas: dict[str, Any],
    preco_capital: float,
) -> dict[str, dict[str, Any]]:
    """Copia atributos observáveis usando nomes econômicos canônicos."""

    snapshot = {}
    for firma in firmas.values():
        dados = {
            "setor": firma.setor,
            "regime": firma.regime,
            "preco_capital": float(preco_capital),
        }
        for atributo in _ATRIBUTOS_FIRMAS:
            valor = getattr(firma, atributo, np.nan)
            dados[atributo] = (
                valor.item()
                if isinstance(valor, np.generic)
                else deepcopy(valor)
            )
        snapshot[firma.id] = dados
    return snapshot


def registrar_resultados_periodo(
    periodo: int,
    *,
    firmas: dict[str, Any],
    estado: dict[str, Any],
    resultados: dict,
    macro: dict[str, Any],
    setores: dict[str, Any],
    cei: dict[str, Any],
    financeiro: dict[str, Any],
    diagnosticos: dict[str, Any] | None = None,
    preco_capital: float = np.nan,
) -> None:
    """Registra a fotografia completa de ``periodo`` pela única porta."""

    if periodo in resultados:
        raise ValueError(f"O período {periodo} já foi registrado.")

    snapshot_financeiro = deepcopy(financeiro)
    snapshot_financeiro["ativos_financeiros"] = deepcopy(
        estado["financeiro"]["ativos_financeiros"]
    )
    snapshot_financeiro["passivos_financeiros"] = deepcopy(
        estado["financeiro"]["passivos_financeiros"]
    )
    resultados[periodo] = {
        "macro": deepcopy(macro),
        "setores": deepcopy(setores),
        "firmas": _snapshot_firmas(firmas, preco_capital),
        "cei": deepcopy(cei),
        "financeiro": snapshot_financeiro,
        "diagnosticos": deepcopy(diagnosticos or {}),
    }


def construir_historico_macro(resultados: dict) -> pd.DataFrame:
    """Cria a mesma view macro pública percorrendo os períodos."""

    if not resultados:
        return pd.DataFrame()
    return pd.DataFrame(
        snapshot["macro"] for snapshot in resultados.values()
    ).set_index("periodo")


def _dataframe_firmas_periodo(
    periodo: int,
    firmas: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    linhas = []
    for id_firma, dados in firmas.items():
        linha = {
            "periodo": periodo,
            "firma": id_firma,
            "setor": dados["setor"],
            "regime": dados["regime"],
            "preco_capital": dados["preco_capital"],
        }
        for atributo in _ATRIBUTOS_FIRMAS:
            coluna = _ALIASES_VIEW_FIRMAS.get(atributo, atributo)
            linha[coluna] = dados[atributo]
        linhas.append(linha)
    return pd.DataFrame(linhas)


def concatenar_resultados_firmas(resultados: dict) -> pd.DataFrame:
    """Materializa a view das firmas somente para análise final."""

    tabelas = [
        _dataframe_firmas_periodo(periodo, snapshot["firmas"])
        for periodo, snapshot in resultados.items()
    ]
    return pd.concat(tabelas, ignore_index=True) if tabelas else pd.DataFrame()


def concatenar_diagnostico(
    resultados: dict,
    nome: str,
) -> pd.DataFrame:
    """Concatena um diagnóstico percorrendo diretamente os períodos."""

    valores = [
        snapshot["diagnosticos"][nome]
        for snapshot in resultados.values()
        if nome in snapshot["diagnosticos"]
    ]
    tabelas = [
        valor if isinstance(valor, pd.DataFrame) else pd.DataFrame(valor)
        for valor in valores
    ]
    return pd.concat(tabelas, ignore_index=True) if tabelas else pd.DataFrame()
