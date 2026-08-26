"""Executa uma unica trajetoria benchmark com os parametros calibrados."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

import matplotlib.pyplot as plt
import pandas as pd

from calibracao.calibrar_modelo import calibrar_modelo
from configuracao_projeto import (
    ARQUIVO_CEI,
    DATA_DIR,
    validar_caminhos_dados,
)
from experimentos.monte_carlo_100 import (
    ARQUIVO_PARAMETROS_CALIBRADOS,
    carregar_configuracao_calibrada,
)
from experimentos.paralelizacao import extrair_historico_macro
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais
from simulacao.simular_trajetoria import simular_trajetoria


VARIAVEIS_GRAFICOS = {
    "pib_real": {
        "titulo": "PIB real - benchmark calibrado",
        "ylabel": "Nivel real",
        "escala": 1.0,
    },
    "taxa_desemprego": {
        "titulo": "Taxa de desemprego - benchmark calibrado",
        "ylabel": "Taxa de desemprego (%)",
        "escala": 100.0,
    },
    "inflacao": {
        "titulo": "Inflacao - benchmark calibrado",
        "ylabel": "Inflacao (%)",
        "escala": 100.0,
    },
    "deficit_governo": {
        "titulo": "Deficit do governo - benchmark calibrado",
        "ylabel": "Deficit do governo (% do PIB)",
        "escala": 100.0,
    },
    "deficit_externo": {
        "titulo": "Deficit externo - benchmark calibrado",
        "ylabel": "Deficit externo (% do PIB)",
        "escala": 100.0,
    },
}


def construir_historico(resultados: dict, *, seed: int) -> pd.DataFrame:
    """Converte a unica trajetoria para a tabela macro padronizada."""

    simulacoes = {1: {"seed": seed, "resultados": resultados}}
    cenario = {
        "nome": "benchmark_calibrado",
        "tipo": "benchmark",
        "multiplicador_produtividade": 1.0,
        "perda_produtividade": 0.0,
        "choque_permanente": False,
    }
    return extrair_historico_macro(simulacoes, cenario=cenario)


def criar_graficos_individuais(
    historico: pd.DataFrame,
    *,
    mostrar: bool,
) -> dict[str, plt.Figure]:
    """Cria uma figura por serie e as mostra no Positron."""

    figuras = {}
    for variavel, definicao in VARIAVEIS_GRAFICOS.items():
        figura, eixo = plt.subplots(figsize=(11, 6))
        eixo.plot(
            historico["periodo"],
            definicao["escala"] * historico[variavel],
            color="#2f5597",
            linewidth=2.2,
        )
        eixo.set(
            title=definicao["titulo"],
            xlabel="Periodo",
            ylabel=definicao["ylabel"],
        )
        eixo.grid(alpha=0.25)
        figura.tight_layout()
        figuras[variavel] = figura

    if mostrar:
        plt.show()
    return figuras


def executar_laboratorio(
    *,
    seed: int = 42,
    periodos: int | None = None,
    arquivo_parametros: Path = ARQUIVO_PARAMETROS_CALIBRADOS,
    mostrar_graficos: bool = True,
) -> dict:
    """Roda somente o benchmark calibrado e devolve objetos para inspecao."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed deve ser um inteiro.")
    if periodos is not None and (
        isinstance(periodos, bool)
        or not isinstance(periodos, int)
        or periodos < 1
    ):
        raise ValueError("periodos deve ser um inteiro positivo.")

    config, config_abm, payload = carregar_configuracao_calibrada(
        arquivo_parametros
    )
    if periodos is not None:
        config["periodos"] = periodos
        config["periodo_choque"] = min(config["periodo_choque"], periodos)
    config_abm["choques_climaticos"]["ativo"] = False

    data_dir, arquivo_cei = validar_caminhos_dados(DATA_DIR, ARQUIVO_CEI)
    condicoes_iniciais = preparar_condicoes_iniciais(
        config,
        data_dir,
        arquivo_cei,
    )
    calibracoes = calibrar_modelo(
        condicoes_iniciais=condicoes_iniciais,
        CONFIG=config,
        CONFIG_ABM=config_abm,
    )
    resultados = simular_trajetoria(
        condicoes_iniciais=condicoes_iniciais,
        calibracoes=calibracoes,
        CONFIG=config,
        CONFIG_ABM=config_abm,
        seed=seed,
    )
    historico = construir_historico(resultados, seed=seed)
    figuras = criar_graficos_individuais(
        historico,
        mostrar=mostrar_graficos,
    )
    return {
        "historico": historico,
        "resultados": resultados,
        "CONFIG": config,
        "CONFIG_ABM": config_abm,
        "condicoes_iniciais": condicoes_iniciais,
        "calibracoes": calibracoes,
        "payload_calibracao": payload,
        "figuras": figuras,
    }


RESULTADO_LABORATORIO: dict | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--periodos",
        type=int,
        default=None,
        help="Sobrescreve o horizonte do benchmark calibrado.",
    )
    parser.add_argument(
        "--arquivo-parametros",
        type=Path,
        default=ARQUIVO_PARAMETROS_CALIBRADOS,
    )
    parser.add_argument(
        "--nao-mostrar",
        action="store_true",
        help="Executa sem chamar plt.show().",
    )
    argumentos = parser.parse_args()

    global RESULTADO_LABORATORIO
    RESULTADO_LABORATORIO = executar_laboratorio(
        seed=argumentos.seed,
        periodos=argumentos.periodos,
        arquivo_parametros=argumentos.arquivo_parametros,
        mostrar_graficos=not argumentos.nao_mostrar,
    )
    globals().update({
        "historico": RESULTADO_LABORATORIO["historico"],
        "resultados": RESULTADO_LABORATORIO["resultados"],
        "config": RESULTADO_LABORATORIO["CONFIG"],
        "config_abm": RESULTADO_LABORATORIO["CONFIG_ABM"],
        "figuras": RESULTADO_LABORATORIO["figuras"],
    })
    print("Benchmark concluido. Nenhum arquivo foi salvo.")
    print("Objetos disponiveis: historico, resultados, config, config_abm, figuras.")


if __name__ == "__main__":
    main()
