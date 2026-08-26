"""Criação determinística da população microeconômica inicial do modelo."""

from __future__ import annotations

from agentes.firma import Firma
from agentes.fornecedor_importado_abm import (
    FornecedorImportadoABM,
    inicializar_importados_abm,
)
from inicializacao.inicializar_firmas import inicializar_firmas


def inicializar_agentes(
    *,
    condicoes_iniciais: dict,
    calibracoes: dict,
    CONFIG: dict,
    CONFIG_ABM: dict,
) -> tuple[dict[str, Firma], dict[str, FornecedorImportadoABM]]:
    """Retorna novas firmas e novos importados prontos para o período zero."""

    firmas = inicializar_firmas(
        condicoes_iniciais,
        CONFIG_ABM,
        config=CONFIG,
        calibracao_investimento_nf_abm=calibracoes["investimento"],
        calibracao_investimento_nf_legada=calibracoes["investimento"]["legado"],
        parametros_cei=calibracoes["cei"]["parametros"],
        coortes_demografia=(
            calibracoes["demografia"]["coortes"]
            if "demografia" in calibracoes
            else None
        ),
    )

    importados = inicializar_importados_abm(
        condicoes_iniciais=condicoes_iniciais,
        firmas=firmas,
    )

    multiplicador_capacidade_importada = float(
        CONFIG_ABM.get("multiplicador_capacidade_importada", 1.5)
    )
    if multiplicador_capacidade_importada < 0.0:
        raise ValueError("multiplicador_capacidade_importada não pode ser negativo.")

    taxas_setoriais = calibracoes["rentabilidade"][
        "taxa_retorno_parametro_setorial"
    ]
    for firma in firmas.values():
        if firma.setor not in taxas_setoriais:
            continue

        firma.taxa_retorno_parametro = (
            taxas_setoriais[firma.setor] * CONFIG_ABM["adj_r_obs_inicial"]
        )
        firma.calcular_taxa_retorno_observada(
            preco_capital=1.0,
            depreciacao=calibracoes["investimento"]["depreciacao"],
            taxa_juros_real=CONFIG["taxa_juros_real"],
        )
        firma.taxa_retorno_ajustada_anterior = firma.taxa_retorno_ajustada

    return firmas, importados
