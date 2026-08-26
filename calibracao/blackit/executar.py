"""Executa a calibração Black-it sem entrar no núcleo econômico do modelo."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from black_it.calibrator import Calibrator
from black_it.samplers.best_batch import BestBatchSampler
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler

from calibracao.blackit.modelo import (
    ORDEM_MOMENTOS,
    ModeloCalibracao,
    PerdaQuadraticaNormalizada,
    validar_monte_carlo,
)
from calibracao.blackit.parametros import (
    aplicar_theta,
    especificacoes_parametros,
    normalizar_theta,
    tabela_parametros,
)
from configuracao_projeto import OUTPUT_DIR


RAIZ = Path(__file__).resolve().parents[2]
LABORATORIO = RAIZ / "laboratorio_abm_regulacao_preco_medio_demografia.py"
TARGETS = OUTPUT_DIR / "empirica" / "empirical_moments.csv"
OUTPUTS = Path(__file__).resolve().parent / "outputs"
MODOS = {
    "smoke": {
        "mc": 2,
        "batch_size": 2,
        "n_batches": 2,
    },

    "principal": {
        "mc": 5,
        "batch_size": 8,
        "n_batches": 20,
    },

    "final": {
        "mc": 10,
        "batch_size": 10,
        "n_batches": 30,
    },
}
MODO_EXECUCAO = "smoke"
SEEDS_CALIBRACAO = tuple(range(1301, 1311))
NUMERO_MC_VALIDACAO = 100
SEEDS_VALIDACAO = tuple(range(9101, 9101 + NUMERO_MC_VALIDACAO))


def carregar_base_modelo() -> dict:
    """Prepara TRU/CEI uma vez e reutiliza suas estruturas na busca."""
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida), contextlib.redirect_stderr(saida):
        base = runpy.run_path(
            str(LABORATORIO),
            run_name="__calibracao_base__",
        )
    if not base["CONFIG_ABM"].get("usar_demografia_empresas", False):
        raise RuntimeError(
            "A calibração Black-it exige usar_demografia_empresas=True."
        )
    return base


def carregar_targets() -> pd.DataFrame:
    targets = pd.read_csv(TARGETS)
    selecionados = targets.set_index("moment_id").loc[list(ORDEM_MOMENTOS)].reset_index()
    if not np.isfinite(selecionados[["value", "scale"]].to_numpy()).all():
        raise ValueError("Targets empíricos não finitos.")
    if not (selecionados["unit"].isin(["fraction", "correlation"])).all():
        raise ValueError("Targets não estão em frações/correlações.")
    return selecionados


def _salvar_outputs(
    *, calibrador, params_ordenados, losses_ordenadas, specs,
    targets, modo, configuracao,
) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    nomes = [item.nome for item in specs]
    resultados = pd.DataFrame(params_ordenados, columns=nomes)
    resultados.insert(0, "rank", np.arange(1, len(resultados) + 1))
    resultados["loss"] = losses_ordenadas
    resultados.to_csv(OUTPUTS / "resultados_blackit.csv", index=False)

    melhor = {nome: float(valor) for nome, valor in zip(nomes, params_ordenados[0], strict=True)}
    (OUTPUTS / "parametros_calibrados.json").write_text(
        json.dumps(
            {
                "modo": modo,
                "parametros": melhor,
                "loss": float(losses_ordenadas[0]),
                "script_input_blackit": str(LABORATORIO.resolve()),
                "contrato_theta": nomes,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    indice_melhor = int(np.argmin(calibrador.losses_samp))
    erros = calibrador.series_samp[indice_melhor, 0, 0, :]
    alvo = targets.set_index("moment_id").loc[list(ORDEM_MOMENTOS)]
    simulados = alvo["value"].to_numpy() + erros * alvo["scale"].to_numpy()
    comparacao = pd.DataFrame({
        "variavel": alvo["variable"].to_numpy(),
        "momento": alvo["moment"].to_numpy(),
        "empirico": alvo["value"].to_numpy(),
        "simulado": simulados,
    })
    comparacao["erro"] = comparacao["simulado"] - comparacao["empirico"]
    comparacao["erro_normalizado"] = erros
    comparacao.to_csv(OUTPUTS / "comparacao_momentos.csv", index=False)
    (OUTPUTS / "configuracao_calibracao.json").write_text(
        json.dumps(configuracao, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    perdas = np.asarray(calibrador.losses_samp, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(perdas) + 1), perdas, "o", label="loss")
    ax.plot(np.arange(1, len(perdas) + 1), np.minimum.accumulate(perdas), label="melhor acumulada")
    ax.set(xlabel="candidato", ylabel="J(theta)", title="Convergência Black-it")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "convergencia.png", dpi=150)
    plt.close(fig)


def executar(modo: str = "smoke") -> dict[str, object]:
    parametros_modo = MODOS[modo]
    base = carregar_base_modelo()
    targets = carregar_targets()
    specs = especificacoes_parametros(base["CONFIG"], base["CONFIG_ABM"])
    theta_atual = np.array([item.atual for item in specs], dtype=float)
    seeds = SEEDS_CALIBRACAO[: parametros_modo["mc"]]
    if len(seeds) != parametros_modo["mc"]:
        raise RuntimeError("Seeds insuficientes para o Monte Carlo de calibração.")
    modelo = ModeloCalibracao(
        condicoes_iniciais=base["condicoes_iniciais"], calibracoes=base["calibracoes"],
        CONFIG=base["CONFIG"], CONFIG_ABM=base["CONFIG_ABM"], targets=targets, seeds=seeds,
    )

    estoque_antes = base["CONFIG_ABM"]["parametro_estoque_desejado"]
    config_theta, config_abm_theta = aplicar_theta(theta_atual, base["CONFIG"], base["CONFIG_ABM"])
    assert config_theta["periodos"] == 15
    assert config_abm_theta["parametro_estoque_desejado"] == estoque_antes
    assert base["CONFIG_ABM"]["parametro_estoque_desejado"] == estoque_antes

    erros_1, individuais_1 = modelo.avaliar(theta_atual)
    erros_2, individuais_2 = modelo.avaliar(theta_atual)
    np.testing.assert_array_equal(erros_1, erros_2)
    assert individuais_1 == individuais_2
    validar_monte_carlo(individuais_1, parametros_modo["mc"])

    batch_size = parametros_modo["batch_size"]
    n_batches = parametros_modo["n_batches"]
    candidatos = batch_size * n_batches
    trajetorias_busca = candidatos * parametros_modo["mc"]
    print(f"Parâmetros em theta: {len(specs)}")
    print(f"Candidatos previstos: {candidatos}")
    print(f"MC por candidato: {parametros_modo['mc']}")
    print("Períodos por trajetória: 15")
    print(f"Trajetórias previstas na busca: {trajetorias_busca}")
    print(f"Períodos simulados previstos na busca: {15 * trajetorias_busca}")

    samplers = [
        HaltonSampler(batch_size=batch_size, random_state=501),
        RandomForestSampler(batch_size=batch_size, random_state=502, n_estimators=50),
        BestBatchSampler(batch_size=batch_size, random_state=503),
    ]
    bounds = [[item.lower for item in specs], [item.upper for item in specs]]
    calibrador = Calibrator(
        loss_function=PerdaQuadraticaNormalizada(coordinate_weights=np.ones(len(ORDEM_MOMENTOS))),
        real_data=np.zeros((1, len(ORDEM_MOMENTOS))), model=modelo,
        parameters_bounds=bounds, parameters_precision=[item.precision for item in specs],
        ensemble_size=1, samplers=samplers, sim_length=1, random_state=500,
        n_jobs=1, verbose=True,
    )
    params, losses = calibrador.calibrate(n_batches=n_batches)
    params = np.asarray(
        [
            normalizar_theta(
                candidato,
                base["CONFIG"],
                base["CONFIG_ABM"],
            )
            for candidato in params
        ],
        dtype=float,
    )
    configuracao = {
        "modo_executado": modo, "periodos": 15,
        "script_input_blackit": str(LABORATORIO.resolve()),
        "funcao_carregamento_input": "calibracao.blackit.executar.carregar_base_modelo",
        "demografia_empresas_ativa": True,
        "mc_smoke": MODOS["smoke"]["mc"], "mc_principal": MODOS["principal"]["mc"],
        "batch_size": batch_size, "n_batches": n_batches, "ensemble_size_blackit": 1,
        "n_jobs_blackit": 1,
        "execucao_mc_calibracao": "sequencial dentro de cada candidato",
        "execucao_experimentos": "ProcessPoolExecutor com contexto spawn",
        "seeds_calibracao": list(seeds),
        "seed_base_trajetorias": base["CONFIG_ABM"]["semente_qualidade"],
        "numero_mc_validacao_pronto_nao_executado": NUMERO_MC_VALIDACAO,
        "theta": tabela_parametros(base["CONFIG"], base["CONFIG_ABM"]),
        "momentos": list(ORDEM_MOMENTOS), "choques_nivel_neutros": True,
        "choques_climaticos_ativos": False,
        "sensibilidade_local": "implementada para avaliação baixo/atual/alto; não executada no smoke",
    }
    _salvar_outputs(
        calibrador=calibrador, params_ordenados=params, losses_ordenadas=losses,
        specs=specs, targets=targets, modo=modo, configuracao=configuracao,
    )
    return {"params": params, "losses": losses,
            "trajetorias_validacoes": 2 * parametros_modo["mc"],
            "trajetorias_busca": trajetorias_busca}


def auditar_sensibilidade_local(modelo, specs) -> pd.DataFrame:
    """Avalia baixo/atual/alto para cada parâmetro, sem grid combinado."""
    centro = np.array([item.atual for item in specs], dtype=float)
    linhas = []
    for indice, spec in enumerate(specs):
        avaliacoes = {}
        for ponto, valor in (
            ("baixo", spec.lower),
            ("atual", spec.atual),
            ("alto", spec.upper),
        ):
            theta = centro.copy()
            theta[indice] = valor
            erros, _ = modelo.avaliar(theta)
            avaliacoes[ponto] = erros
        mudanca = avaliacoes["alto"] - avaliacoes["baixo"]
        mais_sensivel = int(np.argmax(np.abs(mudanca)))
        linhas.append({
            "parametro": spec.nome,
            "momento_afetado": ORDEM_MOMENTOS[mais_sensivel],
            "sinal": "positivo" if mudanca[mais_sensivel] > 0 else "negativo",
            "intensidade_aproximada_normalizada": float(abs(mudanca[mais_sensivel])),
        })
    return pd.DataFrame(linhas)


def validar_theta_final_100(
    *, theta, condicoes_iniciais, calibracoes, CONFIG, CONFIG_ABM, targets,
) -> pd.DataFrame:
    """Executa a validação final MC=100 somente quando chamada explicitamente."""
    modelo = ModeloCalibracao(
        condicoes_iniciais=condicoes_iniciais,
        calibracoes=calibracoes,
        CONFIG=CONFIG,
        CONFIG_ABM=CONFIG_ABM,
        targets=targets,
        seeds=SEEDS_VALIDACAO,
    )
    _, individuais = modelo.avaliar(theta)
    alvo = targets.set_index("moment_id")
    linhas = []
    for nome in ORDEM_MOMENTOS:
        valores = np.array([item[nome] for item in individuais], dtype=float)
        linhas.append({
            "moment_id": nome,
            "empirico": float(alvo.loc[nome, "value"]),
            "media_mc": float(np.mean(valores)),
            "mediana_mc": float(np.median(valores)),
            "p5": float(np.quantile(valores, 0.05)),
            "p95": float(np.quantile(valores, 0.95)),
        })
    return pd.DataFrame(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--modo",
        choices=tuple(MODOS),
        default=MODO_EXECUCAO,
    )

    args = parser.parse_args()

    resultado = executar(args.modo)

    print(
        f"Black-it {args.modo}: concluído; "
        f"melhor loss={resultado['losses'][0]:.8g}"
    )


if __name__ == "__main__":
    main()
