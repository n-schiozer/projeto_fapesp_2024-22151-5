"""Analisa as IRFs já simuladas sem executar novamente o modelo."""

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

from configuracao_projeto import OUTPUT_DIR


PASTA_RESULTADOS = OUTPUT_DIR / "experimentos_calibrados"
NOME_PASTA_ANALISE = "analise_irfs"
VARIAVEIS = (
    "pib_real",
    "taxa_desemprego",
    "inflacao",
    "deficit_governo",
    "deficit_externo",
)
ROTULOS = {
    "pib_real": "PIB real",
    "taxa_desemprego": "Desemprego",
    "inflacao": "Inflação",
    "deficit_governo": "Déficit do governo / PIB",
    "deficit_externo": "Déficit externo / PIB",
}
ORDEM_CENARIOS = (
    "choque_05pct_temporario",
    "choque_05pct_permanente",
    "choque_10pct_temporario",
    "choque_10pct_permanente",
    "choque_20pct_temporario",
    "choque_20pct_permanente",
)
CHAVES_GRUPO = [
    "cenario",
    "tipo_choque",
    "multiplicador_produtividade",
    "perda_produtividade",
    "choque_permanente",
    "periodo",
    "variavel",
    "unidade",
]


def carregar_resultados(pasta_resultados: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lê trajetórias, resumo original e metadados do estudo."""

    pasta_resultados = Path(pasta_resultados)
    caminhos = {
        "trajetorias": pasta_resultados / "irf_trajetorias.csv",
        "resumo": pasta_resultados / "irf_resumo.csv",
        "configuracao": pasta_resultados / "configuracao_experimentos.json",
    }
    ausentes = [str(caminho) for caminho in caminhos.values() if not caminho.is_file()]
    if ausentes:
        raise FileNotFoundError(f"Arquivos de resultados ausentes: {ausentes}")

    trajetorias = pd.read_csv(caminhos["trajetorias"])
    resumo_original = pd.read_csv(caminhos["resumo"])
    configuracao = json.loads(
        caminhos["configuracao"].read_text(encoding="utf-8")
    )
    validar_resultados(trajetorias, resumo_original, configuracao)
    return trajetorias, resumo_original, configuracao


def validar_resultados(
    trajetorias: pd.DataFrame,
    resumo_original: pd.DataFrame,
    configuracao: dict,
) -> None:
    """Impede analisar arquivos incompletos ou sem pareamento Monte Carlo."""

    colunas_trajetorias = set(CHAVES_GRUPO + ["simulacao", "seed", "irf"])
    colunas_resumo = set(CHAVES_GRUPO + ["media", "p5", "p95"])
    faltantes_trajetorias = sorted(colunas_trajetorias - set(trajetorias.columns))
    faltantes_resumo = sorted(colunas_resumo - set(resumo_original.columns))
    if faltantes_trajetorias or faltantes_resumo:
        raise ValueError(
            "Contrato dos CSVs inválido; "
            f"trajetórias={faltantes_trajetorias}, resumo={faltantes_resumo}."
        )
    if trajetorias.empty or resumo_original.empty:
        raise ValueError("Os arquivos de IRF estão vazios.")
    if set(trajetorias["variavel"]) != set(VARIAVEIS):
        raise ValueError("As cinco variáveis esperadas não estão presentes.")
    identificadores = ["cenario", "simulacao", "seed", "periodo", "variavel"]
    if trajetorias.duplicated(identificadores).any():
        raise ValueError("Há observações de trajetória duplicadas.")
    numericas = [
        "multiplicador_produtividade",
        "perda_produtividade",
        "periodo",
        "simulacao",
        "seed",
        "irf",
    ]
    if not np.isfinite(trajetorias[numericas].to_numpy(dtype=float)).all():
        raise ValueError("As trajetórias contêm valores não finitos.")

    numero_esperado = int(configuracao["numero_simulacoes_por_cenario"])
    tamanhos = trajetorias.groupby(
        ["cenario", "periodo", "variavel"], sort=False
    )["simulacao"].nunique()
    if not tamanhos.eq(numero_esperado).all():
        raise ValueError("Nem todas as IRFs possuem o Monte Carlo completo.")


def calcular_estatisticas_detalhadas(trajetorias: pd.DataFrame) -> pd.DataFrame:
    """Inclui intervalo de confiança da média e quantis das trajetórias."""

    estatisticas = (
        trajetorias.groupby(CHAVES_GRUPO, as_index=False)["irf"]
        .agg(
            numero_simulacoes="size",
            media="mean",
            mediana="median",
            desvio_padrao="std",
            p5=lambda valores: valores.quantile(0.05),
            p25=lambda valores: valores.quantile(0.25),
            p75=lambda valores: valores.quantile(0.75),
            p95=lambda valores: valores.quantile(0.95),
        )
    )
    estatisticas["erro_padrao"] = (
        estatisticas["desvio_padrao"]
        / np.sqrt(estatisticas["numero_simulacoes"])
    )
    margem = 1.96 * estatisticas["erro_padrao"]
    estatisticas["media_ic95_inferior"] = estatisticas["media"] - margem
    estatisticas["media_ic95_superior"] = estatisticas["media"] + margem
    estatisticas["media_diferente_zero_95"] = (
        (estatisticas["media_ic95_inferior"] > 0.0)
        | (estatisticas["media_ic95_superior"] < 0.0)
    )
    return estatisticas


def validar_resumo_recalculado(
    estatisticas: pd.DataFrame,
    resumo_original: pd.DataFrame,
) -> None:
    """Confere se o resumo entregue corresponde às trajetórias salvas."""

    comparacao = estatisticas.merge(
        resumo_original[CHAVES_GRUPO + ["media", "p5", "p95"]],
        on=CHAVES_GRUPO,
        suffixes=("_novo", "_original"),
        validate="one_to_one",
    )
    for coluna in ("media", "p5", "p95"):
        if not np.allclose(
            comparacao[f"{coluna}_novo"],
            comparacao[f"{coluna}_original"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"O resumo original diverge em {coluna}.")


def definir_horizontes(periodo_choque: int, periodo_final: int) -> dict[str, int]:
    candidatos = {
        "impacto": periodo_choque,
        "+1": periodo_choque + 1,
        "+5": periodo_choque + 5,
        "+10": periodo_choque + 10,
        "final": periodo_final,
    }
    return {
        nome: periodo
        for nome, periodo in candidatos.items()
        if periodo <= periodo_final
    }


def construir_tabela_horizontes(
    estatisticas: pd.DataFrame,
    horizontes: dict[str, int],
) -> pd.DataFrame:
    periodo_para_horizonte = {periodo: nome for nome, periodo in horizontes.items()}
    tabela = estatisticas.loc[
        estatisticas["periodo"].isin(periodo_para_horizonte)
    ].copy()
    tabela.insert(
        tabela.columns.get_loc("periodo") + 1,
        "horizonte",
        tabela["periodo"].map(periodo_para_horizonte),
    )
    return tabela


def resumir_cenarios(
    estatisticas: pd.DataFrame,
    *,
    periodo_choque: int,
    horizontes: dict[str, int],
) -> pd.DataFrame:
    """Resume impacto, persistência, extremos e efeito acumulado por cenário."""

    chaves = [chave for chave in CHAVES_GRUPO if chave != "periodo"]
    linhas = []
    for valores_chave, dados in estatisticas.groupby(chaves, sort=False):
        dados = dados.sort_values("periodo")
        pos_choque = dados.loc[dados["periodo"].ge(periodo_choque)]
        pico = pos_choque.loc[pos_choque["media"].abs().idxmax()]
        minimo = pos_choque.loc[pos_choque["media"].idxmin()]
        maximo = pos_choque.loc[pos_choque["media"].idxmax()]
        significativos = pos_choque.loc[pos_choque["media_diferente_zero_95"]]
        linha = dict(zip(chaves, valores_chave, strict=True))
        linha.update({
            "perda_percentual": 100.0 * linha["perda_produtividade"],
            "periodo_pico_absoluto": int(pico["periodo"]),
            "irf_pico_absoluto": float(pico["media"]),
            "periodo_minimo": int(minimo["periodo"]),
            "irf_minima": float(minimo["media"]),
            "periodo_maximo": int(maximo["periodo"]),
            "irf_maxima": float(maximo["media"]),
            "soma_irf_pos_choque": float(pos_choque["media"].sum()),
            "media_irf_pos_choque": float(pos_choque["media"].mean()),
            "periodos_significativos_95": int(len(significativos)),
            "primeiro_periodo_significativo_95": (
                int(significativos["periodo"].min())
                if not significativos.empty else np.nan
            ),
            "ultimo_periodo_significativo_95": (
                int(significativos["periodo"].max())
                if not significativos.empty else np.nan
            ),
        })
        for nome_horizonte, periodo in horizontes.items():
            observacao = dados.loc[dados["periodo"].eq(periodo)]
            if not observacao.empty:
                linha[f"irf_{nome_horizonte}"] = float(observacao.iloc[0]["media"])
        linhas.append(linha)
    return pd.DataFrame(linhas)


def comparar_duracoes(resumo_cenarios: pd.DataFrame) -> pd.DataFrame:
    """Calcula permanente menos temporário para cada intensidade e variável."""

    indices = ["perda_produtividade", "perda_percentual", "variavel", "unidade"]
    metricas = [
        coluna
        for coluna in resumo_cenarios.columns
        if coluna.startswith("irf_") or coluna in {
            "soma_irf_pos_choque",
            "media_irf_pos_choque",
            "periodos_significativos_95",
        }
    ]
    temporario = resumo_cenarios.loc[
        resumo_cenarios["tipo_choque"].eq("temporario"), indices + metricas
    ]
    permanente = resumo_cenarios.loc[
        resumo_cenarios["tipo_choque"].eq("permanente"), indices + metricas
    ]
    comparacao = temporario.merge(
        permanente,
        on=indices,
        suffixes=("_temporario", "_permanente"),
        validate="one_to_one",
    )
    for metrica in metricas:
        comparacao[f"diferenca_{metrica}"] = (
            comparacao[f"{metrica}_permanente"]
            - comparacao[f"{metrica}_temporario"]
        )
    return comparacao


def estimar_dose_resposta(tabela_horizontes: pd.DataFrame) -> pd.DataFrame:
    """Estima a resposta a um ponto percentual adicional de perda produtiva."""

    linhas = []
    chaves = ["tipo_choque", "variavel", "unidade", "horizonte", "periodo"]
    for valores_chave, dados in tabela_horizontes.groupby(chaves, sort=False):
        dados = dados.sort_values("perda_produtividade")
        x = 100.0 * dados["perda_produtividade"].to_numpy(dtype=float)
        y = dados["media"].to_numpy(dtype=float)
        inclinacao, intercepto = np.polyfit(x, y, 1)
        previsto = inclinacao * x + intercepto
        soma_quadrados = float(np.sum((y - y.mean()) ** 2))
        r2 = (
            1.0 - float(np.sum((y - previsto) ** 2)) / soma_quadrados
            if soma_quadrados > 0.0 else 1.0
        )
        inclinacao_origem = float(np.dot(x, y) / np.dot(x, x))
        linha = dict(zip(chaves, valores_chave, strict=True))
        linha.update({
            "numero_intensidades": len(dados),
            "inclinacao_por_1pp_perda": float(inclinacao),
            "intercepto": float(intercepto),
            "r2_linear": r2,
            "inclinacao_pela_origem_por_1pp_perda": inclinacao_origem,
        })
        linhas.append(linha)
    return pd.DataFrame(linhas)


def diagnosticar_pre_choque(
    estatisticas: pd.DataFrame,
    periodo_choque: int,
) -> pd.DataFrame:
    pre = estatisticas.loc[estatisticas["periodo"].lt(periodo_choque)].copy()
    return (
        pre.groupby(["cenario", "variavel"], as_index=False)
        .agg(
            max_abs_media=("media", lambda valores: valores.abs().max()),
            max_abs_limite_distribuicao=(
                "p95",
                lambda valores: valores.abs().max(),
            ),
        )
    )


def _cor_por_perda(perdas: list[float]) -> dict[float, object]:
    cores = plt.cm.viridis(np.linspace(0.15, 0.85, len(perdas)))
    return dict(zip(perdas, cores, strict=True))


def grafico_irfs(
    estatisticas: pd.DataFrame,
    *,
    periodo_choque: int,
    pasta: Path,
) -> dict[str, Path]:
    """Cria uma figura de IRF separada para cada variável."""

    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    perdas = sorted(estatisticas["perda_produtividade"].unique())
    cores = _cor_por_perda(perdas)
    for variavel in VARIAVEIS:
        figura, eixos = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
        for coluna, tipo in enumerate(("temporario", "permanente")):
            eixo = eixos[coluna]
            recorte = estatisticas.loc[
                estatisticas["variavel"].eq(variavel)
                & estatisticas["tipo_choque"].eq(tipo)
            ]
            for perda in perdas:
                dados = recorte.loc[
                    recorte["perda_produtividade"].eq(perda)
                ].sort_values("periodo")
                cor = cores[perda]
                x = dados["periodo"].to_numpy(dtype=float)
                eixo.fill_between(
                    x,
                    dados["p5"].to_numpy(dtype=float),
                    dados["p95"].to_numpy(dtype=float),
                    color=cor,
                    alpha=0.08,
                )
                eixo.fill_between(
                    x,
                    dados["media_ic95_inferior"].to_numpy(dtype=float),
                    dados["media_ic95_superior"].to_numpy(dtype=float),
                    color=cor,
                    alpha=0.22,
                )
                eixo.plot(
                    x,
                    dados["media"].to_numpy(dtype=float),
                    color=cor,
                    linewidth=2.0,
                    label=f"perda de {100 * perda:.0f}%",
                )
            eixo.axhline(0.0, color="black", linewidth=0.8)
            eixo.axvline(periodo_choque, color="black", linestyle=":")
            eixo.set_title(f"{ROTULOS[variavel]} — {tipo}")
            eixo.set_ylabel(recorte["unidade"].iloc[0])
            eixo.grid(alpha=0.2)
            eixo.legend()
        eixos[0].set_xlabel("Período")
        eixos[1].set_xlabel("Período")
        figura.suptitle(
            f"{ROTULOS[variavel]} — média, IC95 e faixa p5--p95",
            fontsize=16,
        )
        figura.tight_layout()
        caminho = pasta / f"irf_{variavel}.png"
        figura.savefig(caminho, dpi=220, bbox_inches="tight")
        plt.close(figura)
        caminhos[f"irf_{variavel}"] = caminho
    return caminhos


def grafico_dose_resposta(
    tabela_horizontes: pd.DataFrame,
    pasta: Path,
) -> dict[str, Path]:
    """Cria uma figura de dose--resposta separada para cada variável."""

    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    horizontes_exibidos = [
        horizonte
        for horizonte in ("impacto", "+5", "final")
        if horizonte in set(tabela_horizontes["horizonte"])
    ]
    estilos = {"impacto": "o-", "+5": "s--", "final": "^-."}
    for variavel in VARIAVEIS:
        figura, eixos = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
        for coluna, tipo in enumerate(("temporario", "permanente")):
            eixo = eixos[coluna]
            recorte = tabela_horizontes.loc[
                tabela_horizontes["variavel"].eq(variavel)
                & tabela_horizontes["tipo_choque"].eq(tipo)
            ]
            for horizonte in horizontes_exibidos:
                dados = recorte.loc[
                    recorte["horizonte"].eq(horizonte)
                ].sort_values("perda_produtividade")
                eixo.plot(
                    100.0 * dados["perda_produtividade"],
                    dados["media"],
                    estilos[horizonte],
                    linewidth=1.8,
                    label=horizonte,
                )
            eixo.axhline(0.0, color="black", linewidth=0.8)
            eixo.set_title(f"{ROTULOS[variavel]} — {tipo}")
            eixo.set_ylabel(recorte["unidade"].iloc[0])
            eixo.grid(alpha=0.2)
            eixo.legend()
        eixos[0].set_xlabel("Perda de produtividade (%)")
        eixos[1].set_xlabel("Perda de produtividade (%)")
        figura.suptitle(f"Dose--resposta — {ROTULOS[variavel]}", fontsize=16)
        figura.tight_layout()
        caminho = pasta / f"dose_resposta_{variavel}.png"
        figura.savefig(caminho, dpi=220, bbox_inches="tight")
        plt.close(figura)
        caminhos[f"dose_resposta_{variavel}"] = caminho
    return caminhos


def grafico_efeito_acumulado(
    resumo_cenarios: pd.DataFrame,
    pasta: Path,
) -> dict[str, Path]:
    """Cria um gráfico acumulado separado para cada variável."""

    pasta.mkdir(parents=True, exist_ok=True)
    caminhos = {}
    ordem = {cenario: indice for indice, cenario in enumerate(ORDEM_CENARIOS)}
    for variavel in VARIAVEIS:
        figura, eixo = plt.subplots(figsize=(13, 7))
        dados = resumo_cenarios.loc[
            resumo_cenarios["variavel"].eq(variavel)
        ].copy()
        dados["ordem"] = dados["cenario"].map(ordem)
        dados = dados.sort_values("ordem", ascending=False)
        cores = [
            "#355f8d" if tipo == "temporario" else "#de8f05"
            for tipo in dados["tipo_choque"]
        ]
        eixo.barh(dados["cenario"], dados["soma_irf_pos_choque"], color=cores)
        eixo.axvline(0.0, color="black", linewidth=0.8)
        eixo.set_title(ROTULOS[variavel])
        eixo.set_xlabel(f"Soma pós-choque ({dados['unidade'].iloc[0]}-período)")
        eixo.grid(axis="x", alpha=0.2)
        figura.tight_layout()
        caminho = pasta / f"efeito_acumulado_{variavel}.png"
        figura.savefig(caminho, dpi=220, bbox_inches="tight")
        plt.close(figura)
        caminhos[f"efeito_acumulado_{variavel}"] = caminho
    return caminhos


def escrever_relatorio(
    *,
    caminho: Path,
    configuracao: dict,
    resumo_cenarios: pd.DataFrame,
    diagnostico_pre_choque: pd.DataFrame,
    caminhos: dict[str, Path],
) -> None:
    max_pre = float(diagnostico_pre_choque["max_abs_media"].max())
    linhas = [
        "# Análise dos experimentos climáticos calibrados",
        "",
        f"- Simulações por cenário: {configuracao['numero_simulacoes_por_cenario']}",
        f"- Período do choque: {configuracao['periodo_choque']}",
        f"- Loss da calibração: {configuracao.get('loss_calibracao')}",
        f"- Maior IRF média absoluta antes do choque: {max_pre:.6g}",
        "",
        "## Síntese por cenário e variável",
        "",
        "| Cenário | Variável | Pico médio | Período do pico | Soma pós-choque | IRF final |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, linha in resumo_cenarios.sort_values(
        ["cenario", "variavel"]
    ).iterrows():
        linhas.append(
            f"| {linha['cenario']} | {ROTULOS[linha['variavel']]} | "
            f"{linha['irf_pico_absoluto']:.4f} | "
            f"{int(linha['periodo_pico_absoluto'])} | "
            f"{linha['soma_irf_pos_choque']:.4f} | "
            f"{linha['irf_final']:.4f} |"
        )
    linhas.extend([
        "",
        "## Arquivos produzidos",
        "",
    ])
    for nome, arquivo in caminhos.items():
        linhas.append(f"- `{nome}`: `{arquivo.name}`")
    linhas.extend([
        "",
        "As IRFs do PIB estão em percentual relativo ao benchmark. As demais "
        "estão em pontos percentuais relativos ao benchmark. O IC95 refere-se "
        "à incerteza Monte Carlo da média; p5--p95 descreve a dispersão das "
        "trajetórias.",
    ])
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def executar_analise(pasta_resultados: Path = PASTA_RESULTADOS) -> dict:
    """Executa toda a análise sobre resultados existentes e salva os artefatos."""

    pasta_resultados = Path(pasta_resultados)
    trajetorias, resumo_original, configuracao = carregar_resultados(
        pasta_resultados
    )
    periodo_choque = int(configuracao["periodo_choque"])
    periodo_final = int(trajetorias["periodo"].max())
    horizontes = definir_horizontes(periodo_choque, periodo_final)

    estatisticas = calcular_estatisticas_detalhadas(trajetorias)
    validar_resumo_recalculado(estatisticas, resumo_original)
    tabela_horizontes = construir_tabela_horizontes(estatisticas, horizontes)
    resumo_cenarios = resumir_cenarios(
        estatisticas,
        periodo_choque=periodo_choque,
        horizontes=horizontes,
    )
    comparacao_duracoes = comparar_duracoes(resumo_cenarios)
    dose_resposta = estimar_dose_resposta(tabela_horizontes)
    pre_choque = diagnosticar_pre_choque(estatisticas, periodo_choque)

    pasta_analise = pasta_resultados / NOME_PASTA_ANALISE
    pasta_analise.mkdir(parents=True, exist_ok=True)
    caminhos = {
        "estatisticas_detalhadas": pasta_analise / "estatisticas_irf_detalhadas.csv",
        "horizontes": pasta_analise / "irfs_horizontes.csv",
        "resumo_cenarios": pasta_analise / "resumo_cenarios.csv",
        "comparacao_duracoes": pasta_analise / "comparacao_temporario_permanente.csv",
        "dose_resposta": pasta_analise / "dose_resposta.csv",
        "diagnostico_pre_choque": pasta_analise / "diagnostico_pre_choque.csv",
        "relatorio": pasta_analise / "relatorio_analise.md",
        "manifesto": pasta_analise / "manifesto_analise.json",
    }
    tabelas = {
        "estatisticas_detalhadas": estatisticas,
        "horizontes": tabela_horizontes,
        "resumo_cenarios": resumo_cenarios,
        "comparacao_duracoes": comparacao_duracoes,
        "dose_resposta": dose_resposta,
        "diagnostico_pre_choque": pre_choque,
    }
    for nome, tabela in tabelas.items():
        tabela.to_csv(caminhos[nome], index=False)
    pasta_graficos = pasta_analise / "graficos_individuais"
    caminhos.update(
        grafico_irfs(
            estatisticas,
            periodo_choque=periodo_choque,
            pasta=pasta_graficos,
        )
    )
    caminhos.update(
        grafico_dose_resposta(tabela_horizontes, pasta_graficos)
    )
    caminhos.update(
        grafico_efeito_acumulado(resumo_cenarios, pasta_graficos)
    )
    escrever_relatorio(
        caminho=caminhos["relatorio"],
        configuracao=configuracao,
        resumo_cenarios=resumo_cenarios,
        diagnostico_pre_choque=pre_choque,
        caminhos=caminhos,
    )
    manifesto = {
        "pasta_resultados": str(pasta_resultados.resolve()),
        "periodo_choque": periodo_choque,
        "periodo_final": periodo_final,
        "horizontes": horizontes,
        "numero_simulacoes_por_cenario": configuracao[
            "numero_simulacoes_por_cenario"
        ],
        "arquivos": {nome: str(caminho.resolve()) for nome, caminho in caminhos.items()},
    }
    caminhos["manifesto"].write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"tabelas": tabelas, "caminhos": caminhos, "manifesto": manifesto}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pasta-resultados",
        type=Path,
        default=PASTA_RESULTADOS,
        help="Pasta que contém irf_trajetorias.csv e irf_resumo.csv.",
    )
    argumentos = parser.parse_args()
    resultado = executar_analise(argumentos.pasta_resultados)
    print(f"Análise concluída: {resultado['caminhos']['relatorio'].parent}")
    print(f"Relatório: {resultado['caminhos']['relatorio']}")


if __name__ == "__main__":
    main()
