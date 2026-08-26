"""Executa, passo a passo, a calibração de firmas da Demografia.

Uso, a partir da pasta ``ABM BRASIL``::

    python demografia_empresas/laboratorio_funcoes_calibrar_firmas_demografia.py

Os resultados de cada etapa são exibidos no terminal e salvos em
``demografia_empresas/saidas_laboratorio_calibracao_demografia``. O script
não altera os arquivos usados pela simulação ABM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demografia.calibrar_firmas_demografia import (
    ARQUIVO_TRU_PADRAO,
    calibrar_atributos_mercado,
    coorte_setor_t,
    gerar_firmas_sinteticas,
    ler_demografia,
    ler_ocupacoes_tru,
    normalizar_coortes_para_abm,
    sortear_pessoal,
)


PASTA_DEMOGRAFIA = Path(__file__).resolve().parent
ARQUIVO_DEMOGRAFIA_PADRAO = PASTA_DEMOGRAFIA / "Demografia_Empresas.xlsx"
PASTA_SAIDAS_PADRAO = PASTA_DEMOGRAFIA / "saidas_laboratorio_calibracao_demografia"


def exibir_etapa(titulo: str, tabela: pd.DataFrame | pd.Series, linhas: int = 8) -> None:
    """Mostra no terminal uma amostra curta e identificada de cada saída."""

    print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}")
    if isinstance(tabela, pd.Series):
        print(tabela.head(linhas).to_string())
    else:
        print(tabela.head(linhas).to_string(index=False))
    print(f"Linhas: {len(tabela):,}")


def salvar(tabela: pd.DataFrame | pd.Series, caminho: Path) -> None:
    """Salva as tabelas intermediárias sem substituir os dados de produção."""

    if isinstance(tabela, pd.Series):
        tabela.rename(tabela.name or "valor").to_csv(caminho, header=True)
    else:
        tabela.to_csv(caminho, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Laboratório sequencial das funções de calibração demográfica."
    )
    parser.add_argument("--arquivo", type=Path, default=ARQUIVO_DEMOGRAFIA_PADRAO)
    parser.add_argument("--aba", default="Planilha1")
    parser.add_argument("--arquivo-tru", type=Path, default=ARQUIVO_TRU_PADRAO)
    parser.add_argument("--aba-tru", default="VA")
    parser.add_argument(
        "--distribuicao", choices=("lognormal", "pareto"), default="pareto"
    )
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--tamanho-coorte", type=int, default=10_000)
    parser.add_argument("--pasta-saidas", type=Path, default=PASTA_SAIDAS_PADRAO)
    argumentos = parser.parse_args()

    if argumentos.tamanho_coorte < 1:
        raise ValueError("--tamanho-coorte deve ser inteiro positivo.")
    if not argumentos.arquivo.is_file():
        raise FileNotFoundError(f"Demografia não encontrada: {argumentos.arquivo}")
    if not argumentos.arquivo_tru.is_file():
        raise FileNotFoundError(f"TRU não encontrada: {argumentos.arquivo_tru}")

    argumentos.pasta_saidas.mkdir(parents=True, exist_ok=True)

    print("Configuração do laboratório")
    print(f"  Demografia: {argumentos.arquivo}")
    print(f"  TRU:        {argumentos.arquivo_tru}")
    print(f"  Distribuição: {argumentos.distribuicao}")
    print(f"  Semente: {argumentos.semente}")
    print(f"  Tamanho da coorte: {argumentos.tamanho_coorte:,}")

    # 1. ler_ocupacoes_tru
    ocupacoes_tru = ler_ocupacoes_tru(argumentos.arquivo_tru, argumentos.aba_tru)
    exibir_etapa("1. ler_ocupacoes_tru()", ocupacoes_tru)
    salvar(ocupacoes_tru, argumentos.pasta_saidas / "01_ocupacoes_tru.csv")

    # 2. ler_demografia
    dados_demografia = ler_demografia(argumentos.arquivo, argumentos.aba)
    exibir_etapa("2. ler_demografia()", dados_demografia)
    salvar(dados_demografia, argumentos.pasta_saidas / "02_demografia_lida.csv")

    # 3. sortear_pessoal, exibida em uma faixa para tornar o sorteio visível.
    exemplo_faixa = dados_demografia.iloc[0]
    limites = dados_demografia.loc[
        dados_demografia["setor"] == exemplo_faixa["setor"], "limite_inferior"
    ].sort_values().to_numpy(dtype=float)
    posicao = int(np.where(limites == exemplo_faixa["limite_inferior"])[0][0])
    limite_superior = (
        limites[posicao + 1] - 1.0
        if posicao + 1 < len(limites)
        else max(2.0 * limites[posicao], 2.0 * exemplo_faixa["pessoal_ocupado"] / exemplo_faixa["numero_empresas"])
    )
    sorteio = sortear_pessoal(
        numero_empresas=int(exemplo_faixa["numero_empresas"]),
        pessoal_ocupado=float(exemplo_faixa["pessoal_ocupado"]),
        limite_inferior_faixa=float(exemplo_faixa["limite_inferior"]),
        limite_superior_faixa=float(limite_superior),
        distribuicao=argumentos.distribuicao,
        gerador=np.random.default_rng(argumentos.semente),
    )
    saida_sorteio = pd.DataFrame(
        {
            "setor": exemplo_faixa["setor"],
            "faixa_pessoal": exemplo_faixa["Faias de pessoal ocupado total"],
            "pessoal_ocupado_firma_sorteado": sorteio,
        }
    )
    exibir_etapa("3. sortear_pessoal() — primeira faixa da Demografia", saida_sorteio)
    salvar(saida_sorteio, argumentos.pasta_saidas / "03_sorteio_exemplo.csv")

    # 4. calibrar_atributos_mercado, em um exemplo mínimo legível.
    exemplo_atributos = pd.DataFrame(
        {
            "setor": ["EXEMPLO"] * 3,
            "id_firma": ["EX_1", "EX_2", "EX_3"],
            "pessoal_ocupado_firma": [10, 30, 60],
        }
    )
    exemplo_atributos = calibrar_atributos_mercado(exemplo_atributos)
    exibir_etapa("4. calibrar_atributos_mercado() — exemplo de três firmas", exemplo_atributos)
    salvar(exemplo_atributos, argumentos.pasta_saidas / "04_atributos_mercado_exemplo.csv")

    # 5. gerar_firmas_sinteticas. K é mantido agregado no ABM e T é adicionado
    # pela função específica abaixo, tal como no laboratório de simulação.
    ocupacoes_abm = ocupacoes_tru.drop(index=["K", "T"], errors="ignore")
    dados_abm = dados_demografia.loc[
        dados_demografia["setor"].isin(ocupacoes_abm.index)
    ].copy()
    resumo, coortes_antes_abm = gerar_firmas_sinteticas(
        dados=dados_abm,
        distribuicao=argumentos.distribuicao,
        semente=argumentos.semente,
        ocupacoes_tru=ocupacoes_abm,
        tamanho_coorte=argumentos.tamanho_coorte,
    )
    exibir_etapa("5a. gerar_firmas_sinteticas() — resumo por setor", resumo)
    exibir_etapa("5b. gerar_firmas_sinteticas() — coortes antes da calibração ABM", coortes_antes_abm)
    salvar(resumo, argumentos.pasta_saidas / "05_resumo_firmas_sinteticas.csv")
    salvar(coortes_antes_abm, argumentos.pasta_saidas / "06_coortes_antes_calibracao_abm.csv")

    # 6. coorte_setor_t
    coorte_t = coorte_setor_t(int(ocupacoes_tru.at["T"]))
    exibir_etapa("6. coorte_setor_t()", coorte_t)
    salvar(coorte_t, argumentos.pasta_saidas / "07_coorte_setor_t.csv")

    # 7. normalizar_coortes_para_abm. Na regra atual da tabela de referência,
    # todos os preços iniciais são um e a qualidade reproduz cada share.
    coortes_iniciais = pd.concat([coortes_antes_abm, coorte_t], ignore_index=True)
    coortes_finais = normalizar_coortes_para_abm(coortes_iniciais)
    exibir_etapa("7. normalizar_coortes_para_abm() — coortes finais", coortes_finais)
    salvar(coortes_finais, argumentos.pasta_saidas / "08_coortes_finais_abm.csv")

    atratividade = (
        coortes_finais["qualidade"] ** 2.0
        * coortes_finais["preco_relativo"] ** -1.2
    )
    diagnostico = coortes_finais.assign(atratividade=atratividade).groupby("setor").agg(
        numero_coortes=("id_firma", "size"),
        firmas_representadas=("numero_firmas_representadas", "sum"),
        ocupacoes=("pessoal_ocupado_firma", "sum"),
        soma_shares=("market_share_domestico", "sum"),
        preco_medio_ponderado=("preco_relativo", lambda p: float(
            (p * coortes_finais.loc[p.index, "market_share_domestico"]).sum()
        )),
        soma_atratividade=("atratividade", "sum"),
    ).reset_index()
    diagnostico["erro_share_multilogit"] = (
        diagnostico["soma_atratividade"] - 1.0
    ).abs()
    exibir_etapa("8. Diagnóstico das coortes finais", diagnostico)
    salvar(diagnostico, argumentos.pasta_saidas / "09_diagnostico_coortes_finais.csv")

    print(f"\nArquivos intermediários salvos em: {argumentos.pasta_saidas}")


if __name__ == "__main__":
    main()
