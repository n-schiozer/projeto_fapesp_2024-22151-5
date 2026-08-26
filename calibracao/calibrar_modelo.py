"""Coordena todas as calibrações pré-agentes do laboratório principal."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from calibracao.calibracao_investimento_nf_abm import (
    calibrar_investimento_nf_abm,
)
from contabilidade.estrutura_cei import C, L, VA
from demografia.calibrar_firmas_demografia import (
    coorte_setor_t,
    diagnosticar_coortes_demografia,
    gerar_firmas_sinteticas,
    ler_demografia,
    normalizar_coortes_para_abm,
)


def _codigo_setor_modelo(setor: str) -> str:
    codigo = str(setor).split(" - ", 1)[0].strip()
    if len(codigo) != 1 or not codigo.isalpha():
        raise ValueError(f"Rótulo setorial inválido: {setor!r}.")
    return codigo


def _calibrar_investimento(
    *,
    condicoes_iniciais: dict,
    CONFIG: dict,
) -> dict:
    setores = list(condicoes_iniciais["setores"])
    resultado = {
        **calibrar_investimento_nf_abm(
            condicoes_iniciais,
            config=CONFIG,
        ),
        "legado": deepcopy(condicoes_iniciais["investimento_nf"]),
    }
    resultado["pesos_bens_capital_nf"] = (
        resultado["pesos_bens_capital_nf"].reindex(setores).fillna(0.0)
    )
    resultado["investimento_nf_base"] = (
        resultado["pesos_bens_capital_nf"] * resultado["fbcf_nf_total_pb"]
    ).rename("investimento_nf_base")
    resultado["pesos_preco_capital"] = (
        resultado["pesos_bens_capital_nf"].reindex(setores).fillna(0.0)
    )
    soma_pesos_preco_capital = float(resultado["pesos_preco_capital"].sum())
    if soma_pesos_preco_capital <= 0.0:
        raise RuntimeError(
            "Os pesos dos bens de capital devem somar valor positivo."
        )
    resultado["pesos_preco_capital"] = (
        resultado["pesos_preco_capital"] / soma_pesos_preco_capital
    )
    return resultado


def _calibrar_consumo(*, condicoes_iniciais: dict) -> dict:
    consumo_base = condicoes_iniciais["consumo_base"]
    consumo_total = float(consumo_base.sum())
    if consumo_total <= 0.0:
        raise RuntimeError("O consumo do ano-base deve ser positivo.")
    return {
        "pesos_consumo": (
            consumo_base / consumo_total
        ).rename("peso_consumo")
    }


def _calibrar_cei(*, condicoes_iniciais: dict) -> dict:
    return {
        "parametros": {
            **deepcopy(condicoes_iniciais["parametros_cei"]),
            "parcela_impostos_produtos_ff": (
                float(
                    condicoes_iniciais["valores_cei"].iat[
                        L["impostos_produtos"],
                        C["ff_s"],
                    ]
                )
                / float(
                    condicoes_iniciais["valores_cei"].iat[
                        L["impostos_produtos"],
                        C["governo_e"],
                    ]
                )
            ),
        }
    }


def _calibrar_demografia(
    *,
    condicoes_iniciais: dict,
    CONFIG: dict,
    CONFIG_ABM: dict,
) -> dict:
    setores = list(condicoes_iniciais["setores"])
    setor_financeiro = setores[int(CONFIG["setor_financeiro"])]
    codigo_financeiro = _codigo_setor_modelo(setor_financeiro)
    arquivo_demografia = Path(CONFIG_ABM["demografia_empresas"]["arquivo"])
    if not arquivo_demografia.is_file():
        raise FileNotFoundError(
            f"Arquivo de Demografia não encontrado: {arquivo_demografia}"
        )

    ocupacoes_por_codigo = pd.Series(
        {
            _codigo_setor_modelo(setor): int(
                round(
                    float(
                        condicoes_iniciais["va_base"].at[
                            VA["ocupacoes"],
                            setor,
                        ]
                    )
                )
            )
            for setor in setores
            if _codigo_setor_modelo(setor) != codigo_financeiro
        },
        name="ocupacoes_tru",
    )
    dados_demografia = ler_demografia(
        arquivo_demografia,
        CONFIG_ABM["demografia_empresas"]["aba"],
    )
    dados_demografia = dados_demografia.loc[
        dados_demografia["setor"].isin(ocupacoes_por_codigo.index)
        & ~dados_demografia["setor"].isin([codigo_financeiro, "T"])
    ].copy()

    resumo, coortes = gerar_firmas_sinteticas(
        dados=dados_demografia,
        distribuicao=CONFIG_ABM["demografia_empresas"]["distribuicao"],
        semente=int(CONFIG_ABM["demografia_empresas"]["semente"]),
        ocupacoes_tru=ocupacoes_por_codigo.drop(index="T", errors="ignore"),
        tamanho_coorte=int(
            CONFIG_ABM["demografia_empresas"]["tamanho_coorte"]
        ),
    )
    if "T" in ocupacoes_por_codigo.index:
        coortes = pd.concat(
            [coortes, coorte_setor_t(int(ocupacoes_por_codigo.at["T"]))],
            ignore_index=True,
        )

    coortes = normalizar_coortes_para_abm(coortes)
    diagnostico = diagnosticar_coortes_demografia(resumo, coortes)
    coortes_por_codigo = {
        codigo: grupo.reset_index(drop=True)
        for codigo, grupo in coortes.groupby("setor", sort=False)
    }
    return {
        "resumo": resumo,
        "coortes": coortes,
        "diagnostico": diagnostico,
        "numero_firmas_por_setor": {
            setor: (
                1
                if setor == setor_financeiro
                else len(coortes_por_codigo[_codigo_setor_modelo(setor)])
            )
            for setor in setores
        },
        "market_shares_domesticos": {
            setor: (
                np.ones(1)
                if setor == setor_financeiro
                else coortes_por_codigo[_codigo_setor_modelo(setor)][
                    "market_share_domestico"
                ].to_numpy(dtype=float)
            )
            for setor in setores
        },
        "precos_relativos_iniciais": {
            setor: (
                np.ones(1)
                if setor == setor_financeiro
                else coortes_por_codigo[_codigo_setor_modelo(setor)][
                    "preco_relativo"
                ].to_numpy(dtype=float)
            )
            for setor in setores
        },
    }


def _calibrar_rentabilidade(
    *,
    condicoes_iniciais: dict,
    calibracao_investimento: dict,
    CONFIG: dict,
) -> dict:
    taxas_setoriais = {}
    for setor in calibracao_investimento["setores_nf"]:
        capital_base_setorial = float(
            calibracao_investimento["estoque_capital_inicial"].at[setor]
        )
        if capital_base_setorial <= 0.0:
            continue
        # Na abertura das firmas, cada parcela do EOB e do capital setorial é
        # multiplicada pelo mesmo share doméstico. Como os shares somam um, a
        # agregação pré-agentes abaixo é a identidade exata da soma das firmas.
        eob_base_setorial = float(
            condicoes_iniciais["razoes_va"].at[
                VA["eob_mais_misto"],
                setor,
            ]
            * calibracao_investimento["producao_anterior"].at[setor]
        )
        taxas_setoriais[setor] = (
            eob_base_setorial / capital_base_setorial
            - float(calibracao_investimento["depreciacao"])
            - float(CONFIG["taxa_juros_real"])
        )
    return {"taxa_retorno_parametro_setorial": taxas_setoriais}


def calibrar_modelo(
    *,
    condicoes_iniciais: dict,
    CONFIG: dict,
    CONFIG_ABM: dict,
) -> dict:
    """Retorna calibrações determinísticas sem alterar nenhuma fonte de entrada."""

    calibracoes = {
        "investimento": _calibrar_investimento(
            condicoes_iniciais=condicoes_iniciais,
            CONFIG=CONFIG,
        ),
        "consumo": _calibrar_consumo(
            condicoes_iniciais=condicoes_iniciais,
        ),
        "cei": _calibrar_cei(condicoes_iniciais=condicoes_iniciais),
    }
    if bool(CONFIG_ABM.get("usar_demografia_empresas", False)):
        calibracoes["demografia"] = _calibrar_demografia(
            condicoes_iniciais=condicoes_iniciais,
            CONFIG=CONFIG,
            CONFIG_ABM=CONFIG_ABM,
        )
    calibracoes["rentabilidade"] = _calibrar_rentabilidade(
        condicoes_iniciais=condicoes_iniciais,
        calibracao_investimento=calibracoes["investimento"],
        CONFIG=CONFIG,
    )
    return calibracoes
