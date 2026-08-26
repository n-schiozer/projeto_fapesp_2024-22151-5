"""Monte Carlo do benchmark, sem cenários de choque.

Executa diversas trajetórias do modelo calibrado usando sementes diferentes e
salva a base completa, as médias por período e um gráfico macro em
``outputs/monte_carlo``.
"""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support
from pathlib import Path
import sys


PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

import matplotlib.pyplot as plt
import pandas as pd

from calibracao.calibrar_modelo import calibrar_modelo
from configuracao_projeto import ARQUIVO_CEI, DATA_DIR, OUTPUT_DIR, validar_caminhos_dados
from experimentos.monte_carlo_100 import (
    ARQUIVO_PARAMETROS_CALIBRADOS,
    NUMERO_PROCESSOS,
    NUMERO_SIMULACOES,
    carregar_configuracao_calibrada,
    executar_cenario,
)
from experimentos.paralelizacao import resolver_numero_processos
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais


OUTPUT_MONTE_CARLO = OUTPUT_DIR / "monte_carlo"
VARIAVEIS_GRAFICO = {
    "pib_real": ("PIB real", "nível"),
    "taxa_desemprego": ("Taxa de desemprego", "%"),
    "inflacao": ("Inflação", "%"),
}


def criar_grafico_medias(medias: pd.DataFrame, caminho: Path) -> None:
    """Salva um gráfico compacto das médias das simulações."""

    figura, eixos = plt.subplots(len(VARIAVEIS_GRAFICO), 1, figsize=(10, 9), sharex=True)
    for eixo, (variavel, (titulo, unidade)) in zip(eixos, VARIAVEIS_GRAFICO.items()):
        eixo.plot(medias["periodo"], medias[variavel], color="#2f5597", linewidth=2)
        eixo.set_title(titulo)
        eixo.set_ylabel(unidade)
        eixo.grid(alpha=0.25)
    eixos[-1].set_xlabel("Período")
    figura.tight_layout()
    figura.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(figura)


def executar_monte_carlo(
    *,
    numero_simulacoes: int = NUMERO_SIMULACOES,
    numero_processos: int = NUMERO_PROCESSOS,
    seed_inicial: int = 42,
    periodos: int | None = None,
    arquivo_parametros: Path = ARQUIVO_PARAMETROS_CALIBRADOS,
) -> dict[str, object]:
    """Executa repetições independentes apenas do benchmark calibrado."""

    if isinstance(numero_simulacoes, bool) or numero_simulacoes < 1:
        raise ValueError("numero_simulacoes deve ser um inteiro positivo.")
    if isinstance(seed_inicial, bool) or not isinstance(seed_inicial, int):
        raise ValueError("seed_inicial deve ser um inteiro.")
    if periodos is not None and (isinstance(periodos, bool) or periodos < 1):
        raise ValueError("periodos deve ser um inteiro positivo.")

    config, config_abm, payload = carregar_configuracao_calibrada(arquivo_parametros)
    if periodos is not None:
        config["periodos"] = periodos
        config["periodo_choque"] = min(config["periodo_choque"], periodos)
    config_abm["choques_climaticos"]["ativo"] = False
    numero_processos = resolver_numero_processos(numero_processos, numero_simulacoes)
    seeds = tuple(seed_inicial + indice for indice in range(numero_simulacoes))
    cenario = {
        "nome": "benchmark",
        "tipo": "benchmark",
        "multiplicador_produtividade": 1.0,
        "perda_produtividade": 0.0,
        "choque_permanente": False,
        "CONFIG_ABM": config_abm,
    }

    data_dir, arquivo_cei = validar_caminhos_dados(DATA_DIR, ARQUIVO_CEI)
    condicoes_iniciais = preparar_condicoes_iniciais(config, data_dir, arquivo_cei)
    calibracoes = calibrar_modelo(
        condicoes_iniciais=condicoes_iniciais,
        CONFIG=config,
        CONFIG_ABM=config_abm,
    )
    historico = executar_cenario(
        cenario=cenario,
        numero_simulacoes=numero_simulacoes,
        numero_processos=numero_processos,
        seeds=seeds,
        condicoes_iniciais=condicoes_iniciais,
        calibracoes=calibracoes,
        config=config,
    )
    medias = historico.groupby("periodo", as_index=False).mean(numeric_only=True)

    OUTPUT_MONTE_CARLO.mkdir(parents=True, exist_ok=True)
    caminhos = {
        "historico": OUTPUT_MONTE_CARLO / "historico_macro.csv",
        "medias": OUTPUT_MONTE_CARLO / "medias_macro.csv",
        "grafico": OUTPUT_MONTE_CARLO / "pib_emprego_inflacao.png",
    }
    historico.to_csv(caminhos["historico"], index=False)
    medias.to_csv(caminhos["medias"], index=False)
    criar_grafico_medias(medias, caminhos["grafico"])
    return {"historico": historico, "medias": medias, "caminhos": caminhos, "calibracao": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numero-simulacoes", type=int, default=NUMERO_SIMULACOES)
    parser.add_argument("--processos", type=int, default=NUMERO_PROCESSOS)
    parser.add_argument("--seed-inicial", type=int, default=42)
    parser.add_argument("--periodos", type=int, default=None)
    parser.add_argument("--arquivo-parametros", type=Path, default=ARQUIVO_PARAMETROS_CALIBRADOS)
    args = parser.parse_args()
    resultado = executar_monte_carlo(
        numero_simulacoes=args.numero_simulacoes,
        numero_processos=args.processos,
        seed_inicial=args.seed_inicial,
        periodos=args.periodos,
        arquivo_parametros=args.arquivo_parametros,
    )
    print(f"Monte Carlo concluído. Resultados: {OUTPUT_MONTE_CARLO}")
    print(f"Gráfico: {resultado['caminhos']['grafico']}")


if __name__ == "__main__":
    freeze_support()
    main()
