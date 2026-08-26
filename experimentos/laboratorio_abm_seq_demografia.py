"""Laboratório ABM com população inicial de firmas calibrada pela Demografia.

O laboratório de referência permanece intacto. Este arquivo o executa com uma
inicialização alternativa, baseada em coortes, e produz os diagnósticos do
ano-base antes do ciclo temporal usual.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import VA
from inicializacao.inicializar_firmas import codigo_setor_modelo, inicializar_firmas


PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_DEMOGRAFIA = PASTA_PROJETO / "demografia_empresas"
sys.path.insert(0, str(PASTA_DEMOGRAFIA))

from demografia.calibrar_firmas_demografia import (  # noqa: E402
    coorte_setor_t,
    gerar_firmas_sinteticas,
    ler_demografia,
    normalizar_coortes_para_abm,
)


CONFIG_ABM_DEMOGRAFIA = {
    "usar_demografia_empresas": True,
    "arquivo_demografia": PASTA_DEMOGRAFIA / "Demografia_Empresas.xlsx",
    "aba_demografia": "Planilha1",
    "distribuicao_tamanho_firmas": "lognormal",
    "semente_demografia": 42,
    # 100 reproduz a granularidade máxima, mas gera aproximadamente 57 mil
    # agentes. O padrão operacional mantém o laboratório reproduzível e leve.
    "tamanho_coorte": 10_000,
}


def construir_coortes_demografia(ci: dict) -> pd.DataFrame:
    """Gera coortes, normaliza preços e recalibra qualidade antes do ABM."""

    configuracao = CONFIG_ABM_DEMOGRAFIA
    caminho = Path(configuracao["arquivo_demografia"])
    if not configuracao["usar_demografia_empresas"]:
        raise ValueError("Este laboratório requer usar_demografia_empresas=True.")
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo de Demografia não encontrado: {caminho}")

    setores = list(ci["setores"])
    setor_financeiro = setores[ci["config"]["setor_financeiro"]]
    codigo_financeiro = codigo_setor_modelo(setor_financeiro)
    ocupacoes_tru = pd.Series(
        {
            codigo_setor_modelo(setor): int(
                round(float(ci["va_base"].at[VA["ocupacoes"], setor]))
            )
            for setor in setores
            if codigo_setor_modelo(setor) != codigo_financeiro
        },
        name="ocupacoes_tru",
    )
    dados = ler_demografia(caminho, configuracao["aba_demografia"])
    dados = dados.loc[
        dados["setor"].isin(ocupacoes_tru.index)
        & ~dados["setor"].isin([codigo_financeiro, "T"])
    ].copy()
    _, coortes = gerar_firmas_sinteticas(
        dados=dados,
        distribuicao=configuracao["distribuicao_tamanho_firmas"],
        semente=configuracao["semente_demografia"],
        ocupacoes_tru=ocupacoes_tru.drop(index="T", errors="ignore"),
        tamanho_coorte=int(configuracao["tamanho_coorte"]),
    )
    if "T" in ocupacoes_tru.index:
        coortes = pd.concat(
            [coortes, coorte_setor_t(int(ocupacoes_tru.at["T"]))],
            ignore_index=True,
        )
    return normalizar_coortes_para_abm(coortes)


def diagnosticar_ano_base(
    firmas: dict,
    ci: dict,
    config_abm: dict,
    calibracao_investimento_nf_abm: dict,
) -> pd.DataFrame:
    """Confere a reprodução setorial antes de iniciar a dinâmica."""

    linhas = []
    for setor in ci["setores"]:
        firmas_setor = [firma for firma in firmas.values() if firma.setor == setor]
        shares = np.asarray(
            [firma.share_domestico_inicial for firma in firmas_setor], dtype=float
        )
        precos = np.asarray(
            [firma.preco_relativo for firma in firmas_setor], dtype=float
        )
        atratividades = np.asarray(
            [
                firma.qualidade ** firma.eta_qualidade
                * firma.preco_relativo ** firma.eta_preco
                for firma in firmas_setor
            ],
            dtype=float,
        )
        shares_multilogit = atratividades / atratividades.sum()
        for firma in firmas_setor:
            firma.calcular_demanda_trabalho()

        producao_tru = float(
            (ci["conversao_domestica"] @ ci["demanda_final_base"]).at[setor]
        )
        capital_setorial = float(
            calibracao_investimento_nf_abm["estoque_capital_inicial"].get(
                setor,
                sum(
                    firma.estoque_capital_inicial_real
                    for firma in firmas_setor
                ),
            )
        )
        variacao_estoque_setorial = float(
            ci["investimento_nf"]["estoques_base"].at[setor]
        )
        forma_estoque = not np.isclose(variacao_estoque_setorial, 0.0)
        estoque_setorial = (
            float(config_abm["parametro_estoque_desejado"])
            * producao_tru
            * float(forma_estoque)
            + variacao_estoque_setorial
        )
        estoque_firmas = float(sum(firma.estoque for firma in firmas_setor))
        variacao_firmas = float(
            sum(
                firma.producao_base_real - firma.producao_vendida_base_real
                for firma in firmas_setor
            )
        )
        ocupacoes_tru = float(ci["va_base"].at[VA["ocupacoes"], setor])
        emprego_firmas = float(sum(firma.demanda_trabalho for firma in firmas_setor))
        linhas.append(
            {
                "setor": setor,
                "numero_coortes": len(firmas_setor),
                "numero_firmas_representadas": int(
                    sum(firma.numero_firmas_representadas for firma in firmas_setor)
                ),
                "ocupacoes_demografia": float(
                    sum(
                        firma.pessoal_ocupado_firma
                        for firma in firmas_setor
                        if np.isfinite(firma.pessoal_ocupado_firma)
                    )
                ),
                "ocupacoes_tru": ocupacoes_tru,
                "soma_market_share": float(shares.sum()),
                "preco_relativo_min": float(precos.min()),
                "preco_relativo_medio": float(precos.mean()),
                "preco_relativo_max": float(precos.max()),
                "preco_medio_ponderado": float(shares @ precos),
                "producao_firmas": float(
                    sum(firma.producao_base_real for firma in firmas_setor)
                ),
                "producao_tru": producao_tru,
                "capital_firmas": float(
                    sum(
                        firma.estoque_capital_inicial_real
                        for firma in firmas_setor
                    )
                ),
                "capital_setorial": capital_setorial,
                "estoque_firmas": estoque_firmas,
                "estoque_setorial": estoque_setorial,
                "variacao_estoque_firmas": variacao_firmas,
                "variacao_estoque_setorial": variacao_estoque_setorial,
                "emprego_firmas": emprego_firmas,
                "emprego_tru": ocupacoes_tru,
                "max_abs_error_share": float(
                    np.abs(shares_multilogit - shares).max()
                ),
            }
        )
    diagnostico = pd.DataFrame(linhas).set_index("setor")
    diagnostico["erro_producao"] = (
        diagnostico["producao_firmas"] - diagnostico["producao_tru"]
    )
    diagnostico["erro_capital"] = (
        diagnostico["capital_firmas"] - diagnostico["capital_setorial"]
    )
    diagnostico["erro_estoque"] = (
        diagnostico["estoque_firmas"] - diagnostico["estoque_setorial"]
    )
    diagnostico["erro_variacao_estoque"] = (
        diagnostico["variacao_estoque_firmas"]
        - diagnostico["variacao_estoque_setorial"]
    )
    diagnostico["erro_emprego"] = (
        diagnostico["emprego_firmas"] - diagnostico["emprego_tru"]
    )
    return diagnostico


coortes_demografia = None
diagnostico_demografia_base = None


def inicializar_firmas_demografia(
    ci: dict,
    config_abm: dict,
    config: dict | None = None,
    calibracao_investimento_nf_abm: dict | None = None,
) -> dict:
    """Injeta as coortes somente nesta execução do laboratório novo."""

    global coortes_demografia, diagnostico_demografia_base
    coortes_demografia = construir_coortes_demografia(ci)
    firmas = inicializar_firmas(
        ci,
        config_abm,
        config=config,
        calibracao_investimento_nf_abm=calibracao_investimento_nf_abm,
        coortes_demografia=coortes_demografia,
    )
    diagnostico_demografia_base = diagnosticar_ano_base(
        firmas,
        ci,
        config_abm,
        calibracao_investimento_nf_abm,
    )
    return firmas


import inicializacao.inicializar_firmas as modulo_inicializacao  # noqa: E402

inicializador_original = modulo_inicializacao.inicializar_firmas
modulo_inicializacao.inicializar_firmas = inicializar_firmas_demografia
try:
    resultado_laboratorio = runpy.run_path(
        str(PASTA_PROJETO / "laboratorio_abm_regulacao_preco_medio.py")
    )
finally:
    modulo_inicializacao.inicializar_firmas = inicializador_original


assert np.allclose(diagnostico_demografia_base["soma_market_share"], 1.0)
assert np.allclose(diagnostico_demografia_base["preco_medio_ponderado"], 1.0)
assert np.allclose(diagnostico_demografia_base["erro_producao"], 0.0, atol=1e-8)
assert np.allclose(diagnostico_demografia_base["erro_capital"], 0.0, atol=1e-8)
assert np.allclose(diagnostico_demografia_base["erro_estoque"], 0.0, atol=1e-8)
assert np.allclose(
    diagnostico_demografia_base["erro_variacao_estoque"], 0.0, atol=1e-8
)
assert np.allclose(diagnostico_demografia_base["erro_emprego"], 0.0, atol=1e-8)
assert np.allclose(diagnostico_demografia_base["max_abs_error_share"], 0.0, atol=1e-10)

print("Demografia: reprodução do ano-base aprovada.")
print(diagnostico_demografia_base.to_string())
print(
    resultado_laboratorio["historico_df"].loc[
        1:3,
        ["pib_real", "inflacao", "taxa_desemprego"],
    ].to_string()
)
