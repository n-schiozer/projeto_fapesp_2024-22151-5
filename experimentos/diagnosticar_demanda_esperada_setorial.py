"""Gera diagnóstico setorial pareado de expectativas sem alterar o modelo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibracao.calibrar_modelo import calibrar_modelo
from configuracao_projeto import (
    ARQUIVO_CEI,
    DATA_DIR,
    OUTPUT_DIR,
    validar_caminhos_dados,
)
from experimentos.monte_carlo_100 import (
    ARQUIVO_PARAMETROS_CALIBRADOS,
    carregar_configuracao_calibrada,
    construir_cenarios,
)
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais
from simulacao.simular_trajetoria import simular_trajetoria


NUMERO_SIMULACOES_DIAGNOSTICO = 3
CENARIOS_PADRAO = (
    "benchmark",
    "choque_05pct_temporario",
    "choque_10pct_temporario",
    "choque_20pct_temporario",
    "choque_20pct_permanente",
)
OUTPUT_DIAGNOSTICO = (
    OUTPUT_DIR / "experimentos_calibrados" / "diagnostico_demanda_setorial"
)
COLUNAS_SOMA = (
    "demanda_esperada",
    "producao_desejada_real",
    "producao_planejada_real",
    "producao_real",
    "vendas_real",
    "demanda_recebida_real",
    "demanda_nao_atendida_real",
    "estoque",
    "estoque_capital_real",
    "investimento_bruto",
    "demanda_trabalho",
    "capacidade_produtiva_real",
    "valor_adicionado_realizado",
)
VARIAVEIS_IRF_PERCENTUAL = (
    "demanda_esperada",
    "producao_desejada_real",
    "producao_planejada_real",
    "producao_real",
    "vendas_real",
    "demanda_recebida_real",
    "estoque",
    "estoque_capital_real",
    "demanda_trabalho",
    "capacidade_produtiva_real",
    "valor_adicionado_realizado",
    "preco_transacao_ponderado",
)
VARIAVEIS_DIFERENCA = (
    "demanda_nao_atendida_real",
    "investimento_bruto",
    "redistribuicao_regulador",
    "erro_expectativa",
)
ROTULOS = {
    "demanda_esperada": "Demanda esperada",
    "vendas_real": "Vendas realizadas",
    "producao_desejada_real": "Produção desejada",
    "producao_real": "Produção realizada",
}


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


def _redistribuicao_por_setor(snapshot: dict) -> pd.Series:
    diagnostico = snapshot["diagnosticos"].get("clima", [])
    if not diagnostico:
        return pd.Series(dtype=float)
    tabela = pd.DataFrame(diagnostico)
    return tabela.groupby("setor")["redistribuicao_regulador"].sum()


def extrair_trajetoria_setorial(
    resultados: dict,
    *,
    simulacao: int,
    seed: int,
    cenario: dict,
) -> pd.DataFrame:
    """Agrega os estados das firmas por setor em cada período."""

    metadados = _metadados_cenario(cenario)
    linhas = []
    for periodo, snapshot in resultados.items():
        firmas = pd.DataFrame.from_dict(snapshot["firmas"], orient="index")
        firmas.index.name = "firma"
        redistribuicao = _redistribuicao_por_setor(snapshot)
        for setor, dados in firmas.groupby("setor", sort=False):
            linha = {
                **metadados,
                "simulacao": simulacao,
                "seed": seed,
                "periodo": periodo,
                "setor": setor,
                "numero_firmas": len(dados),
            }
            for coluna in COLUNAS_SOMA:
                valores = dados[coluna].to_numpy(dtype=float)
                if coluna == "capacidade_produtiva_real":
                    valores = valores[np.isfinite(valores)]
                    linha[coluna] = (
                        float(valores.sum()) if len(valores) else np.nan
                    )
                else:
                    linha[coluna] = float(valores.sum())
            vendas = dados["vendas_real"].to_numpy(dtype=float)
            precos = dados["preco_transacao"].to_numpy(dtype=float)
            linha["preco_transacao_ponderado"] = (
                float(np.average(precos, weights=vendas))
                if vendas.sum() > 0.0 else float(np.nanmean(precos))
            )
            linha["fator_clima_medio"] = float(
                dados["fator_produtividade_climatica"].mean()
            )
            linha["redistribuicao_regulador"] = float(
                redistribuicao.get(setor, 0.0)
            )
            linha["erro_expectativa"] = (
                linha["vendas_real"] - linha["demanda_esperada"]
            )
            linha["gap_producao_desejada"] = (
                linha["producao_real"] - linha["producao_desejada_real"]
            )
            linhas.append(linha)
    tabela = pd.DataFrame(linhas)
    numericas = [
        coluna
        for coluna in COLUNAS_SOMA
        if coluna != "capacidade_produtiva_real"
    ] + [
        "preco_transacao_ponderado",
        "fator_clima_medio",
        "redistribuicao_regulador",
        "erro_expectativa",
        "gap_producao_desejada",
    ]
    if not np.isfinite(tabela[numericas].to_numpy()).all():
        raise ValueError(
            f"Diagnóstico não finito em {cenario['nome']}, seed={seed}."
        )
    return tabela


def calcular_irfs_setoriais(
    historico: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula diferenças pareadas por seed, período e setor."""

    ids = ["simulacao", "seed", "periodo", "setor"]
    variaveis = list(VARIAVEIS_IRF_PERCENTUAL + VARIAVEIS_DIFERENCA)
    benchmark = historico.loc[
        historico["cenario"].eq("benchmark"), ids + variaveis
    ].set_index(ids).sort_index()
    if benchmark.empty or not benchmark.index.is_unique:
        raise ValueError("Benchmark setorial ausente ou duplicado.")

    tabelas = []
    experimentos = historico.loc[~historico["cenario"].eq("benchmark")]
    for cenario, dados in experimentos.groupby("cenario", sort=False):
        metadados = dados.iloc[0]
        valores = dados.set_index(ids).sort_index()
        if not valores.index.equals(benchmark.index):
            raise ValueError(f"Cenário {cenario} não está pareado ao benchmark.")
        for variavel in VARIAVEIS_IRF_PERCENTUAL:
            base = benchmark[variavel]
            experimento = valores[variavel]
            validos = (
                np.isfinite(base)
                & (base > 0.0)
                & np.isfinite(experimento)
            )
            irf = 100.0 * (experimento[validos] / base[validos] - 1.0)
            tabela = irf.rename("irf").reset_index()
            tabela["unidade"] = "% em relação ao benchmark"
            tabela["variavel"] = variavel
            tabelas.append(
                _inserir_metadados_irf(tabela, cenario, metadados)
            )
        for variavel in VARIAVEIS_DIFERENCA:
            irf = valores[variavel] - benchmark[variavel]
            tabela = irf.rename("irf").reset_index()
            tabela["unidade"] = "diferença absoluta em relação ao benchmark"
            tabela["variavel"] = variavel
            tabelas.append(
                _inserir_metadados_irf(tabela, cenario, metadados)
            )
    trajetorias = pd.concat(tabelas, ignore_index=True)
    chaves = [
        "cenario",
        "tipo_choque",
        "multiplicador_produtividade",
        "perda_produtividade",
        "choque_permanente",
        "periodo",
        "setor",
        "variavel",
        "unidade",
    ]
    resumo = (
        trajetorias.groupby(chaves, as_index=False)["irf"]
        .agg(
            numero_simulacoes="size",
            media="mean",
            mediana="median",
            desvio_padrao="std",
            p5=lambda valores: valores.quantile(0.05),
            p95=lambda valores: valores.quantile(0.95),
        )
    )
    return trajetorias, resumo


def _inserir_metadados_irf(
    tabela: pd.DataFrame,
    cenario: str,
    metadados: pd.Series,
) -> pd.DataFrame:
    tabela.insert(0, "cenario", cenario)
    tabela.insert(1, "tipo_choque", metadados["tipo_choque"])
    tabela.insert(
        2,
        "multiplicador_produtividade",
        metadados["multiplicador_produtividade"],
    )
    tabela.insert(3, "perda_produtividade", metadados["perda_produtividade"])
    tabela.insert(4, "choque_permanente", metadados["choque_permanente"])
    return tabela


def criar_graficos_setoriais(
    resumo: pd.DataFrame,
    *,
    periodo_choque: int,
    pasta: Path,
) -> dict[str, Path]:
    """Cria uma figura por setor para expectativas, vendas e produção."""

    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    for setor, dados_setor in resumo.groupby("setor", sort=False):
        figura, eixos = plt.subplots(2, 2, figsize=(17, 11), sharex=True)
        for eixo, variavel in zip(eixos.flat, ROTULOS, strict=True):
            recorte = dados_setor.loc[dados_setor["variavel"].eq(variavel)]
            for cenario, dados in recorte.groupby("cenario", sort=False):
                dados = dados.sort_values("periodo")
                eixo.plot(
                    dados["periodo"],
                    dados["media"],
                    linewidth=1.8,
                    label=cenario,
                )
            eixo.axhline(0.0, color="black", linewidth=0.8)
            eixo.axvline(periodo_choque, color="black", linestyle=":")
            eixo.set_title(ROTULOS[variavel])
            eixo.set_ylabel("% em relação ao benchmark")
            eixo.grid(alpha=0.2)
            eixo.legend(fontsize=8)
        eixos[-1, 0].set_xlabel("Período")
        eixos[-1, 1].set_xlabel("Período")
        figura.suptitle(setor, fontsize=15)
        figura.tight_layout()
        codigo = str(setor).split(" - ", 1)[0].strip().lower()
        caminho = pasta / f"demanda_esperada_setor_{codigo}.png"
        figura.savefig(caminho, dpi=200, bbox_inches="tight")
        plt.close(figura)
        caminhos[setor] = caminho
    return caminhos


def executar_diagnostico(
    *,
    numero_simulacoes: int = NUMERO_SIMULACOES_DIAGNOSTICO,
    nomes_cenarios: tuple[str, ...] = CENARIOS_PADRAO,
    arquivo_parametros: Path = ARQUIVO_PARAMETROS_CALIBRADOS,
) -> dict:
    """Executa somente as trajetórias necessárias ao diagnóstico setorial."""

    if numero_simulacoes < 1:
        raise ValueError("numero_simulacoes deve ser positivo.")
    config, config_abm, payload = carregar_configuracao_calibrada(
        arquivo_parametros
    )
    todos_cenarios = {
        cenario["nome"]: cenario for cenario in construir_cenarios(config_abm)
    }
    ausentes = sorted(set(nomes_cenarios) - set(todos_cenarios))
    if ausentes:
        raise ValueError(f"Cenários desconhecidos: {ausentes}")
    if "benchmark" not in nomes_cenarios:
        raise ValueError("O diagnóstico pareado exige o cenário benchmark.")
    cenarios = [todos_cenarios[nome] for nome in nomes_cenarios]
    seed_base = int(config_abm["semente_qualidade"])
    seeds = tuple(seed_base + indice for indice in range(numero_simulacoes))

    data_dir, arquivo_cei = validar_caminhos_dados(DATA_DIR, ARQUIVO_CEI)
    condicoes = preparar_condicoes_iniciais(config, data_dir, arquivo_cei)
    calibracoes = calibrar_modelo(
        condicoes_iniciais=condicoes,
        CONFIG=config,
        CONFIG_ABM=config_abm,
    )
    historicos = []
    for indice_cenario, cenario in enumerate(cenarios, start=1):
        print(
            f"\n=== Cenário {indice_cenario}/{len(cenarios)}: "
            f"{cenario['nome']} ==="
        )
        for simulacao, seed in enumerate(seeds, start=1):
            print(
                f"\n=== Simulação {simulacao}/{numero_simulacoes} "
                f"| seed={seed} ==="
            )
            resultados = simular_trajetoria(
                condicoes_iniciais=condicoes,
                calibracoes=calibracoes,
                CONFIG=config,
                CONFIG_ABM=cenario["CONFIG_ABM"],
                seed=seed,
            )
            historicos.append(
                extrair_trajetoria_setorial(
                    resultados,
                    simulacao=simulacao,
                    seed=seed,
                    cenario=cenario,
                )
            )
    historico = pd.concat(historicos, ignore_index=True)
    irf_trajetorias, irf_resumo = calcular_irfs_setoriais(historico)

    OUTPUT_DIAGNOSTICO.mkdir(parents=True, exist_ok=True)
    caminhos = {
        "historico": OUTPUT_DIAGNOSTICO / "historico_demanda_setorial.csv",
        "irf_trajetorias": OUTPUT_DIAGNOSTICO / "irf_setorial_trajetorias.csv",
        "irf_resumo": OUTPUT_DIAGNOSTICO / "irf_setorial_resumo.csv",
        "configuracao": OUTPUT_DIAGNOSTICO / "configuracao_diagnostico.json",
    }
    historico.to_csv(caminhos["historico"], index=False)
    irf_trajetorias.to_csv(caminhos["irf_trajetorias"], index=False)
    irf_resumo.to_csv(caminhos["irf_resumo"], index=False)
    graficos = criar_graficos_setoriais(
        irf_resumo,
        periodo_choque=int(config_abm["choques_climaticos"]["setores"][
            next(iter(config_abm["choques_climaticos"]["setores"]))
        ]["periodo_choque"]),
        pasta=OUTPUT_DIAGNOSTICO / "graficos",
    )
    configuracao_saida = {
        "arquivo_parametros": str(Path(arquivo_parametros).resolve()),
        "loss_calibracao": payload.get("loss"),
        "numero_simulacoes": numero_simulacoes,
        "seeds": list(seeds),
        "cenarios": list(nomes_cenarios),
        "observacao": (
            "Seeds idênticas são usadas em todos os cenários. As IRFs são "
            "calculadas trajetória a trajetória antes da média."
        ),
    }
    caminhos["configuracao"].write_text(
        json.dumps(configuracao_saida, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "historico": historico,
        "irf_trajetorias": irf_trajetorias,
        "irf_resumo": irf_resumo,
        "caminhos": caminhos,
        "graficos": graficos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numero-simulacoes",
        type=int,
        default=NUMERO_SIMULACOES_DIAGNOSTICO,
        help="Número de seeds pareadas por cenário (padrão: 3).",
    )
    parser.add_argument(
        "--cenarios",
        nargs="+",
        default=CENARIOS_PADRAO,
        help="Inclua sempre benchmark e os experimentos desejados.",
    )
    parser.add_argument(
        "--arquivo-parametros",
        type=Path,
        default=ARQUIVO_PARAMETROS_CALIBRADOS,
    )
    argumentos = parser.parse_args()
    resultado = executar_diagnostico(
        numero_simulacoes=argumentos.numero_simulacoes,
        nomes_cenarios=tuple(argumentos.cenarios),
        arquivo_parametros=argumentos.arquivo_parametros,
    )
    print(f"\nDiagnóstico concluído: {OUTPUT_DIAGNOSTICO}")
    print(f"IRFs setoriais: {resultado['caminhos']['irf_resumo']}")


if __name__ == "__main__":
    main()
