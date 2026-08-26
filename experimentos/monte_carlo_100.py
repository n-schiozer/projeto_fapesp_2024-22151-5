"""Benchmark calibrado, cenários climáticos pareados e funções impulso-resposta."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from multiprocessing import freeze_support
from pathlib import Path
import sys


PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibracao.blackit.parametros import (
    aplicar_theta,
    especificacoes_parametros,
    normalizar_theta,
)
from calibracao.calibrar_modelo import calibrar_modelo
from configuracao_projeto import (
    ARQUIVO_CEI,
    DATA_DIR,
    DEMOGRAFIA_RAW_DIR,
    OUTPUT_DIR,
    validar_caminhos_dados,
)
from experimentos.paralelizacao import (
    executar_cenario_paralelo,
    resolver_numero_processos,
)
from inicializacao.preparar_modelo_cei import preparar_condicoes_iniciais


NUMERO_SIMULACOES = 100
NUMERO_PROCESSOS = 0
PERIODO_CHOQUE = 5
INTENSIDADES_CHOQUE = (0.95, 0.90, 0.80)
ARQUIVO_PARAMETROS_CALIBRADOS = (
    PASTA_PROJETO
    / "calibracao"
    / "blackit"
    / "outputs"
    / "parametros_calibrados.json"
)
OUTPUT_EXPERIMENTOS = OUTPUT_DIR / "experimentos_calibrados"

VARIAVEIS_RESULTADO = {
    "pib_real": {
        "rotulo": "PIB real",
        "irf": "percentual",
        "unidade_irf": "% em relação ao benchmark",
    },
    "taxa_desemprego": {
        "rotulo": "Taxa de desemprego",
        "irf": "pontos_percentuais",
        "unidade_irf": "p.p. em relação ao benchmark",
    },
    "inflacao": {
        "rotulo": "Inflação",
        "irf": "pontos_percentuais",
        "unidade_irf": "p.p. em relação ao benchmark",
    },
    "deficit_governo": {
        "rotulo": "Déficit do governo / PIB",
        "irf": "pontos_percentuais",
        "unidade_irf": "p.p. do PIB em relação ao benchmark",
    },
    "deficit_externo": {
        "rotulo": "Déficit externo / PIB",
        "irf": "pontos_percentuais",
        "unidade_irf": "p.p. do PIB em relação ao benchmark",
    },
}

CONFIG = {
    "ano": 2020,
    "nivel": 20,
    "aba_cei": "Python",
    "periodos": 25,
    "multiplicador_governo": 1,
    "multiplicador_investimento": 1.0,
    "multiplicador_exportacoes": 1.0,
    "periodo_choque": 2,
    "choque_permanente": True,
    "taxa_desemprego_base": 0.138,
    "taxa_desemprego_inicial": 0.138,
    "taxa_crescimento_populacional": 0.0,
    "taxa_crescimento_demanda_autonoma": 0.0,
    "parcela_ativa_populacao": 0.50,
    "parcela_aposentados_inativos": 0.50,
    "setor_financeiro": 10,
    "setores_excluidos_investimento_nf": [
        "K - Atividades financeiras, de seguros e serviços relacionados",
        "O - Administração pública, defesa e seguridade social",
        "T - Serviços domésticos",
    ],
    "vida_util_capital": 20.0,
    "ano_inicial_beta": 2010,
    "ano_final_beta": 2019,
    "inicializacao_investimento_nf": "estacionaria",
    "razao_estoque_producao": 1.0 / 12.0,
    "velocidade_ajuste_estoques": 1,
    "a0": 0.03,
    "a1": 0.2,
    "a3": 0.2,
    "repasse_inflacao_cambio": 1.0,
    "taxa_juros_real": 0.06,
    "inertia_pm": 0.5,
    "fracao_reavaliacao_financeira": 1.0,
    "tolerancia_consumo": 1e-6,
    "max_iteracoes_consumo": 100,
    "executar_testes": False,
}

SETOR_FINANCEIRO = (
    "K - Atividades financeiras, de seguros e serviços relacionados"
)
SETORES_LEILAO = [
    "A - Agricultura, pecuária, produção florestal, pesca e aquicultura",
    "D - Eletricidade e gás",
]
SETORES_REGULADOS = ["D - Eletricidade e gás"]

CONFIG_ABM = {
    "usar_demografia_empresas": True,
    "numero_firmas_industria": 20,
    "numero_firmas_leilao": 20,
    "numero_firmas_por_setor": {SETOR_FINANCEIRO: 1},
    "setores_leilao": SETORES_LEILAO,
    "setores_regulados": SETORES_REGULADOS,
    "eta_preco_padrao": -1.2,
    "eta_qualidade_padrao": 2.0,
    "eta_atendimento_padrao": 1.0,
    "parametro_estoque_desejado": 0.0978561253333731,
    "ajustes_setoriais": {},
    "market_shares_domesticos": {},
    "precos_relativos_iniciais": {},
    "rho_qualidade": 0.90,
    "sigma_qualidade": 0.02,
    "rho_produtividade_idiossincratica": 0.90,
    "sigma_produtividade_idiossincratica": 0.02,
    "semente_qualidade": 42,
    "semente_exposicao_climatica": 202604,
    "usar_heterogeneidade_tecnologica": True,
    "peso_relativo_ci_eletricidade_exposta": 0.1,
    "probabilidades_exposicao_climatica": {
        SETORES_LEILAO[0]: 0.90,
        SETORES_LEILAO[1]: 0.50,
    },
    "demografia_empresas": {
        "arquivo": DEMOGRAFIA_RAW_DIR / "Demografia_Empresas.xlsx",
        "aba": "Planilha1",
        "distribuicao": "pareto",
        "semente": 42,
        "tamanho_coorte": 1000,
    },
    "parametros_markup": {
        "parametro_markup": 0.1,
        "markup_min": 0.0,
        "markup_max": 10.0,
        "epsilon_market_share": 1e-12,
    },
    "multiplicador_capacidade_importada": 1.5,
    "velocidade_ajuste_expectativa_demanda": 0.50,
    "velocidade_ajuste_estoques_firmas": 0.25,
    "lambda_expectativa_precos": 1.0,
    "adj_r_obs_inicial": 1,
    "utilizacao_capacidade_normal": 0.80,
    "gamma_investimento_retorno": 0.5,
    "gamma_investimento_capacidade": 0.5,
    "choques_climaticos": {
        "ativo": False,
        "setores": {
            SETORES_LEILAO[0]: {
                "periodo_choque": 5,
                "multiplicador_produtividade": 0.95,
                "choque_permanente": False,
            },
            SETORES_LEILAO[1]: {
                "periodo_choque": 5,
                "multiplicador_produtividade": 0.95,
                "choque_permanente": False,
            },
        },
    },
}


def carregar_configuracao_calibrada(
    arquivo: Path = ARQUIVO_PARAMETROS_CALIBRADOS,
    *,
    config_base: dict | None = None,
    config_abm_base: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Aplica theta* sobre as configurações-base e devolve cópias novas.

    ``config_base`` e ``config_abm_base`` permitem que o arquivo-base do
    benchmark exponha todas as hipóteses ao usuário, sem perder os parâmetros
    estimados da calibração.
    """

    arquivo = Path(arquivo)
    if not arquivo.is_file():
        raise FileNotFoundError(f"Parâmetros calibrados ausentes: {arquivo}")
    payload = json.loads(arquivo.read_text(encoding="utf-8"))
    parametros = payload.get("parametros")
    if not isinstance(parametros, dict):
        raise ValueError("parametros_calibrados.json não contém 'parametros'.")

    config_referencia = deepcopy(CONFIG if config_base is None else config_base)
    config_abm_referencia = deepcopy(
        CONFIG_ABM if config_abm_base is None else config_abm_base
    )
    especificacoes = especificacoes_parametros(
        config_referencia,
        config_abm_referencia,
    )
    nomes = [item.nome for item in especificacoes]
    ausentes = [nome for nome in nomes if nome not in parametros]
    extras = sorted(set(parametros) - set(nomes))
    if ausentes or extras:
        raise ValueError(
            f"Contrato de theta incompatível; ausentes={ausentes}, extras={extras}."
        )
    theta = np.array([float(parametros[nome]) for nome in nomes], dtype=float)
    theta = np.asarray(
        normalizar_theta(theta, config_referencia, config_abm_referencia),
        dtype=float,
    )

    config, config_abm = aplicar_theta(
        theta,
        config_referencia,
        config_abm_referencia,
    )
    # O horizonte é uma decisão da rodada, não da busca de calibração.
    config["periodos"] = config_referencia["periodos"]
    return config, config_abm, payload


def construir_cenarios(
    config_abm_calibrado: dict,
    intensidades: tuple[float, ...] = INTENSIDADES_CHOQUE,
) -> list[dict]:
    """Cria um benchmark e choques climáticos temporários/permanentes."""

    intensidades = tuple(float(valor) for valor in intensidades)
    if not intensidades:
        raise ValueError("Informe ao menos uma intensidade de choque.")
    if len(set(intensidades)) != len(intensidades):
        raise ValueError("As intensidades de choque não podem se repetir.")
    benchmark = deepcopy(config_abm_calibrado)
    benchmark["choques_climaticos"]["ativo"] = False
    cenarios = [{
        "nome": "benchmark",
        "tipo": "benchmark",
        "multiplicador_produtividade": 1.0,
        "perda_produtividade": 0.0,
        "choque_permanente": False,
        "CONFIG_ABM": benchmark,
    }]
    for multiplicador in intensidades:
        multiplicador = float(multiplicador)
        if not 0.0 < multiplicador < 1.0:
            raise ValueError("Cada intensidade deve estar estritamente entre 0 e 1.")
        perda = 1.0 - multiplicador
        codigo_perda = int(round(perda * 100))
        for permanente in (False, True):
            duracao = "permanente" if permanente else "temporario"
            config_cenario = deepcopy(config_abm_calibrado)
            config_cenario["choques_climaticos"]["ativo"] = True
            for parametros_setor in config_cenario["choques_climaticos"][
                "setores"
            ].values():
                parametros_setor["periodo_choque"] = PERIODO_CHOQUE
                parametros_setor["multiplicador_produtividade"] = multiplicador
                parametros_setor["choque_permanente"] = permanente
            cenarios.append({
                "nome": f"choque_{codigo_perda:02d}pct_{duracao}",
                "tipo": duracao,
                "multiplicador_produtividade": multiplicador,
                "perda_produtividade": perda,
                "choque_permanente": permanente,
                "CONFIG_ABM": config_cenario,
            })
    return cenarios


def validar_historico_cenario(
    historico: pd.DataFrame,
    *,
    numero_simulacoes: int,
    periodos: int,
    seeds: tuple[int, ...],
) -> None:
    """Verifica cobertura temporal e pareamento das realizações."""

    if historico["simulacao"].nunique() != numero_simulacoes:
        raise RuntimeError("O histórico não contém todas as simulações.")
    if tuple(sorted(historico["seed"].unique())) != tuple(sorted(seeds)):
        raise RuntimeError("O histórico não contém exatamente as seeds comuns.")
    esperados = list(range(periodos + 1))
    for _, dados in historico.groupby("simulacao", sort=False):
        if dados.sort_values("periodo")["periodo"].tolist() != esperados:
            raise RuntimeError("Uma trajetória não contém exatamente t=0,...,T.")


def calcular_estatisticas_cenarios(historicos: pd.DataFrame) -> pd.DataFrame:
    """Resume níveis por cenário, período e variável."""

    long = historicos.melt(
        id_vars=[
            "cenario", "tipo_choque", "multiplicador_produtividade",
            "perda_produtividade", "choque_permanente", "simulacao",
            "seed", "periodo",
        ],
        value_vars=list(VARIAVEIS_RESULTADO),
        var_name="variavel",
        value_name="valor",
    )
    return (
        long.groupby(
            [
                "cenario", "tipo_choque", "multiplicador_produtividade",
                "perda_produtividade", "choque_permanente", "periodo",
                "variavel",
            ],
            as_index=False,
        )["valor"]
        .agg(
            media="mean",
            mediana="median",
            desvio_padrao="std",
            p5=lambda valores: valores.quantile(0.05),
            p95=lambda valores: valores.quantile(0.95),
        )
    )


def calcular_irfs(historicos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula IRFs pareadas por seed antes de resumir o Monte Carlo."""

    ids = ["simulacao", "seed", "periodo"]
    benchmark = (
        historicos.loc[historicos["cenario"].eq("benchmark"), ids + list(VARIAVEIS_RESULTADO)]
        .set_index(ids)
        .sort_index()
    )
    if benchmark.empty:
        raise ValueError("Benchmark ausente para o cálculo das IRFs.")
    if not benchmark.index.is_unique:
        raise ValueError("O benchmark contém observações pareadas duplicadas.")

    linhas = []
    experimentos = historicos.loc[~historicos["cenario"].eq("benchmark")]
    for cenario, dados in experimentos.groupby("cenario", sort=False):
        metadados = dados.iloc[0]
        valores = dados.set_index(ids).sort_index()
        if not valores.index.is_unique:
            raise ValueError(f"O cenário {cenario} contém observações duplicadas.")
        if not valores.index.equals(benchmark.index):
            raise RuntimeError(f"O cenário {cenario} não está pareado ao benchmark.")
        for variavel, especificacao in VARIAVEIS_RESULTADO.items():
            valor_experimento = valores[variavel]
            valor_benchmark = benchmark[variavel]
            if especificacao["irf"] == "percentual":
                if (valor_benchmark == 0.0).any():
                    raise ValueError(f"Benchmark zero impede IRF de {variavel}.")
                resposta = 100.0 * (valor_experimento / valor_benchmark - 1.0)
            else:
                resposta = 100.0 * (valor_experimento - valor_benchmark)
            tabela = resposta.rename("irf").reset_index()
            tabela.insert(0, "cenario", cenario)
            tabela.insert(1, "tipo_choque", metadados["tipo_choque"])
            tabela.insert(
                2,
                "multiplicador_produtividade",
                metadados["multiplicador_produtividade"],
            )
            tabela.insert(3, "perda_produtividade", metadados["perda_produtividade"])
            tabela.insert(4, "choque_permanente", metadados["choque_permanente"])
            tabela.insert(8, "variavel", variavel)
            tabela["unidade"] = especificacao["unidade_irf"]
            linhas.append(tabela)
    irf_trajetorias = pd.concat(linhas, ignore_index=True)
    irf_resumo = (
        irf_trajetorias.groupby(
            [
                "cenario", "tipo_choque", "multiplicador_produtividade",
                "perda_produtividade", "choque_permanente", "periodo",
                "variavel", "unidade",
            ],
            as_index=False,
        )["irf"]
        .agg(
            media="mean",
            mediana="median",
            desvio_padrao="std",
            p5=lambda valores: valores.quantile(0.05),
            p95=lambda valores: valores.quantile(0.95),
        )
    )
    return irf_trajetorias, irf_resumo


def criar_grafico_irfs(irf_resumo: pd.DataFrame, caminho: Path) -> None:
    """Compara intensidades em painéis temporários e permanentes."""

    figura, eixos = plt.subplots(
        len(VARIAVEIS_RESULTADO),
        2,
        figsize=(16, 22),
        sharex=True,
        squeeze=False,
    )
    multiplicadores_globais = sorted(
        irf_resumo["multiplicador_produtividade"].unique(), reverse=True
    )
    cores = plt.cm.viridis(
        np.linspace(0.15, 0.85, len(multiplicadores_globais))
    )
    cor_por_multiplicador = dict(zip(multiplicadores_globais, cores, strict=True))
    for linha, (variavel, especificacao) in enumerate(VARIAVEIS_RESULTADO.items()):
        for coluna, tipo in enumerate(("temporario", "permanente")):
            eixo = eixos[linha, coluna]
            subset = irf_resumo.loc[
                irf_resumo["variavel"].eq(variavel)
                & irf_resumo["tipo_choque"].eq(tipo)
            ]
            multiplicadores = sorted(
                subset["multiplicador_produtividade"].unique(), reverse=True
            )
            for multiplicador in multiplicadores:
                cor = cor_por_multiplicador[multiplicador]
                dados = subset.loc[
                    subset["multiplicador_produtividade"].eq(multiplicador)
                ].sort_values("periodo")
                perda = 100.0 * (1.0 - multiplicador)
                eixo.plot(
                    dados["periodo"], dados["media"], color=cor,
                    linewidth=2.0, label=f"perda de {perda:.0f}%",
                )
                eixo.fill_between(
                    dados["periodo"], dados["p5"], dados["p95"],
                    color=cor, alpha=0.10,
                )
            eixo.axhline(0.0, color="black", linewidth=0.8)
            eixo.axvline(PERIODO_CHOQUE, color="black", linestyle=":", linewidth=1.0)
            eixo.set_title(f"{especificacao['rotulo']} — choque {tipo}")
            eixo.set_ylabel(especificacao["unidade_irf"])
            eixo.grid(alpha=0.2)
            eixo.legend()
    eixos[-1, 0].set_xlabel("Período")
    eixos[-1, 1].set_xlabel("Período")
    figura.tight_layout()
    figura.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(figura)


def criar_grafico_medias(estatisticas: pd.DataFrame, caminho: Path) -> None:
    """Plota níveis médios do benchmark e de todos os experimentos."""

    figura, eixos = plt.subplots(
        len(VARIAVEIS_RESULTADO), 1, figsize=(14, 20), sharex=True
    )
    for eixo, (variavel, especificacao) in zip(
        eixos, VARIAVEIS_RESULTADO.items(), strict=True
    ):
        subset = estatisticas.loc[estatisticas["variavel"].eq(variavel)]
        for cenario, dados in subset.groupby("cenario", sort=False):
            escala = 1.0 if variavel == "pib_real" else 100.0
            largura = 2.8 if cenario == "benchmark" else 1.2
            eixo.plot(
                dados["periodo"], dados["media"] * escala,
                linewidth=largura, label=cenario,
            )
        eixo.axvline(PERIODO_CHOQUE, color="black", linestyle=":", linewidth=1.0)
        eixo.set_title(especificacao["rotulo"])
        eixo.set_ylabel("nível" if variavel == "pib_real" else "% / p.p. do PIB")
        eixo.grid(alpha=0.2)
        eixo.legend(ncol=2, fontsize=8)
    eixos[-1].set_xlabel("Período")
    figura.tight_layout()
    figura.savefig(caminho, dpi=180, bbox_inches="tight")
    plt.close(figura)


def executar_cenario(
    *,
    cenario: dict,
    numero_simulacoes: int,
    numero_processos: int,
    seeds: tuple[int, ...],
    condicoes_iniciais: dict,
    calibracoes: dict,
    config: dict,
) -> pd.DataFrame:
    """Executa um cenário pela API única de trajetórias."""

    historico = executar_cenario_paralelo(
        cenario=cenario,
        numero_simulacoes=numero_simulacoes,
        numero_processos=numero_processos,
        seeds=seeds,
        condicoes_iniciais=condicoes_iniciais,
        calibracoes=calibracoes,
        config=config,
    )
    validar_historico_cenario(
        historico,
        numero_simulacoes=numero_simulacoes,
        periodos=config["periodos"],
        seeds=seeds,
    )
    return historico


def executar_experimento(
    *,
    numero_simulacoes: int = NUMERO_SIMULACOES,
    numero_processos: int = NUMERO_PROCESSOS,
    intensidades: tuple[float, ...] = INTENSIDADES_CHOQUE,
    arquivo_parametros: Path = ARQUIVO_PARAMETROS_CALIBRADOS,
) -> dict:
    """Executa benchmark calibrado e todos os cenários com seeds comuns."""

    if (
        isinstance(numero_simulacoes, bool)
        or not isinstance(numero_simulacoes, int)
        or numero_simulacoes < 1
    ):
        raise ValueError("numero_simulacoes deve ser inteiro positivo.")
    config, config_abm, payload_calibracao = carregar_configuracao_calibrada(
        arquivo_parametros
    )
    numero_processos = resolver_numero_processos(
        numero_processos,
        numero_simulacoes,
    )
    cenarios = construir_cenarios(config_abm, intensidades)
    seed_base = int(config_abm["semente_qualidade"])
    seeds = tuple(seed_base + indice for indice in range(numero_simulacoes))

    numero_cenarios = len(cenarios)
    total_trajetorias = numero_cenarios * numero_simulacoes
    print(f"Cenários: {numero_cenarios} (1 benchmark + {numero_cenarios - 1} experimentos)")
    print(f"Simulações por cenário: {numero_simulacoes}")
    print(f"Processos paralelos: {numero_processos}")
    print(f"Trajetórias totais: {total_trajetorias}")
    print(f"Períodos por trajetória: {config['periodos']}")
    print(f"Períodos econômicos totais: {total_trajetorias * config['periodos']}")

    data_dir, arquivo_cei = validar_caminhos_dados(DATA_DIR, ARQUIVO_CEI)
    condicoes_iniciais = preparar_condicoes_iniciais(
        config, data_dir, arquivo_cei
    )
    calibracoes = calibrar_modelo(
        condicoes_iniciais=condicoes_iniciais,
        CONFIG=config,
        CONFIG_ABM=config_abm,
    )

    OUTPUT_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    historicos = []
    caminhos_cenarios = {}
    for indice, cenario in enumerate(cenarios, start=1):
        print(f"\n=== Cenário {indice}/{numero_cenarios}: {cenario['nome']} ===")
        historico = executar_cenario(
            cenario=cenario,
            numero_simulacoes=numero_simulacoes,
            numero_processos=numero_processos,
            seeds=seeds,
            condicoes_iniciais=condicoes_iniciais,
            calibracoes=calibracoes,
            config=config,
        )
        pasta_cenario = OUTPUT_EXPERIMENTOS / cenario["nome"]
        pasta_cenario.mkdir(parents=True, exist_ok=True)
        caminho = pasta_cenario / "historico_macro.csv"
        historico.to_csv(caminho, index=False)
        caminhos_cenarios[cenario["nome"]] = caminho
        historicos.append(historico)

    historicos_df = pd.concat(historicos, ignore_index=True)
    estatisticas = calcular_estatisticas_cenarios(historicos_df)
    irf_trajetorias, irf_resumo = calcular_irfs(historicos_df)
    caminhos = {
        "historicos": OUTPUT_EXPERIMENTOS / "historico_macro_cenarios.csv",
        "estatisticas": OUTPUT_EXPERIMENTOS / "estatisticas_cenarios.csv",
        "irf_trajetorias": OUTPUT_EXPERIMENTOS / "irf_trajetorias.csv",
        "irf_resumo": OUTPUT_EXPERIMENTOS / "irf_resumo.csv",
        "grafico_irf": OUTPUT_EXPERIMENTOS / "irfs_comparativas.png",
        "grafico_medias": OUTPUT_EXPERIMENTOS / "medias_comparativas.png",
        "configuracao": OUTPUT_EXPERIMENTOS / "configuracao_experimentos.json",
    }
    historicos_df.to_csv(caminhos["historicos"], index=False)
    estatisticas.to_csv(caminhos["estatisticas"], index=False)
    irf_trajetorias.to_csv(caminhos["irf_trajetorias"], index=False)
    irf_resumo.to_csv(caminhos["irf_resumo"], index=False)
    criar_grafico_irfs(irf_resumo, caminhos["grafico_irf"])
    criar_grafico_medias(estatisticas, caminhos["grafico_medias"])

    configuracao_saida = {
        "arquivo_parametros_calibrados": str(Path(arquivo_parametros).resolve()),
        "modo_calibracao": payload_calibracao.get("modo"),
        "loss_calibracao": payload_calibracao.get("loss"),
        "parametros_calibrados": payload_calibracao["parametros"],
        "numero_simulacoes_por_cenario": numero_simulacoes,
        "numero_processos": numero_processos,
        "periodos": config["periodos"],
        "periodo_choque": PERIODO_CHOQUE,
        "intensidades_multiplicador": list(intensidades),
        "setores_atingidos": list(
            config_abm["choques_climaticos"]["setores"]
        ),
        "seeds": list(seeds),
        "cenarios": [
            {chave: valor for chave, valor in cenario.items() if chave != "CONFIG_ABM"}
            for cenario in cenarios
        ],
        "definicoes_irf": VARIAVEIS_RESULTADO,
    }
    caminhos["configuracao"].write_text(
        json.dumps(configuracao_saida, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "historicos": historicos_df,
        "estatisticas": estatisticas,
        "irf_trajetorias": irf_trajetorias,
        "irf_resumo": irf_resumo,
        "caminhos": caminhos,
        "caminhos_cenarios": caminhos_cenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numero-simulacoes", type=int, default=NUMERO_SIMULACOES,
        help="Número de trajetórias por cenário (padrão: 100).",
    )
    parser.add_argument(
        "--processos", type=int, default=NUMERO_PROCESSOS,
        help="Processos paralelos; 0 escolhe automaticamente até 4 (padrão: 0).",
    )
    parser.add_argument(
        "--intensidades", type=float, nargs="+", default=INTENSIDADES_CHOQUE,
        help="Multiplicadores de produtividade, por exemplo 0.95 0.90 0.80.",
    )
    parser.add_argument(
        "--arquivo-parametros", type=Path,
        default=ARQUIVO_PARAMETROS_CALIBRADOS,
    )
    args = parser.parse_args()
    resultado = executar_experimento(
        numero_simulacoes=args.numero_simulacoes,
        numero_processos=args.processos,
        intensidades=tuple(args.intensidades),
        arquivo_parametros=args.arquivo_parametros,
    )
    print(f"\nEstudo concluído. Resultados: {OUTPUT_EXPERIMENTOS}")
    print(f"IRFs resumidas: {resultado['caminhos']['irf_resumo']}")


if __name__ == "__main__":
    freeze_support()
    main()
