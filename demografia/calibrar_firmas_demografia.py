"""Gera coortes de firmas e atributos compatíveis com o mercado industrial ABM.

Uso:
    python calibrar_firmas_demografia.py
    python calibrar_firmas_demografia.py --distribuicao pareto --semente 42

Por padrão, cada linha de saída representa uma coorte de firmas sorteadas da
distribuição de pessoal da Demografia. O emprego e os fluxos dessa linha são
agregados; ``numero_firmas_representadas`` informa sua multiplicidade.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from configuracao_projeto import DATA_DIR


COLUNA_CNAE = "Código CNAE 2.0"
COLUNA_FAIXA = "Faias de pessoal ocupado total"
COLUNA_EMPRESAS = "Número de empresas \ne outras \norganizações"
COLUNA_PESSOAS = "Pessoal ocupado total"
COLUNA_REMUNERACOES = "Salários e outras remunerações"
ARQUIVO_TRU_PADRAO = DATA_DIR / "20_tab2_2020.xlsx"


def limite_inferior(faixa: object) -> int:
    numeros = re.findall(r"\d+", str(faixa))
    if not numeros:
        raise ValueError(f"Faixa de pessoal inválida: {faixa!r}.")
    return int(numeros[0])


def ler_demografia(caminho: Path, aba: str) -> pd.DataFrame:
    """Lê somente as linhas setoriais A--U e converte os totais em números."""

    dados = pd.read_excel(caminho, sheet_name=aba)
    faltantes = {
        COLUNA_CNAE, COLUNA_FAIXA, COLUNA_EMPRESAS, COLUNA_PESSOAS
    }.difference(dados.columns)
    if faltantes:
        raise KeyError(f"Colunas ausentes na planilha: {sorted(faltantes)}")

    dados = dados.copy()
    dados[COLUNA_CNAE] = dados[COLUNA_CNAE].astype(str).str.strip()
    dados = dados.loc[dados[COLUNA_CNAE].str.fullmatch(r"[A-U]")].copy()
    dados["setor"] = dados[COLUNA_CNAE]
    dados["limite_inferior"] = dados[COLUNA_FAIXA].map(limite_inferior)
    dados["numero_empresas"] = pd.to_numeric(
        dados[COLUNA_EMPRESAS], errors="coerce"
    ).fillna(0).astype(int)
    dados["pessoal_ocupado"] = pd.to_numeric(
        dados[COLUNA_PESSOAS], errors="coerce"
    ).fillna(0.0)
    dados["remuneracoes"] = pd.to_numeric(
        dados[COLUNA_REMUNERACOES], errors="coerce"
    ).fillna(0.0)
    return dados.loc[dados["numero_empresas"] > 0].copy()


def ler_ocupacoes_tru(caminho: Path, aba: str = "VA") -> pd.Series:
    """Lê as ocupações da linha ``Fator trabalho`` da TRU."""

    dados = pd.read_excel(caminho, sheet_name=aba, header=None)
    linha_ocupacoes = dados.index[
        dados.iloc[:, 0].astype(str).str.contains("Fator trabalho", na=False)
    ]
    if len(linha_ocupacoes) != 1:
        raise ValueError("A TRU deve conter uma única linha 'Fator trabalho'.")

    linhas_cabecalho = [
        linha
        for linha in range(int(linha_ocupacoes[0]))
        if dados.iloc[linha, 1:].astype(str).str.match(r"^[A-U](?:\n|$)").sum() >= 10
    ]
    if len(linhas_cabecalho) != 1:
        raise ValueError("Não foi possível identificar os cabeçalhos setoriais da TRU.")

    ocupacoes: dict[str, int] = {}
    for coluna, cabecalho in dados.iloc[linhas_cabecalho[0], 1:].items():
        codigo = re.match(r"^([A-U])(?:\n|$)", str(cabecalho))
        if codigo:
            valor = pd.to_numeric(dados.iat[linha_ocupacoes[0], coluna], errors="coerce")
            ocupacoes[codigo.group(1)] = int(round(float(valor)))
    if not ocupacoes:
        raise ValueError("Não foram encontradas ocupações setoriais válidas na TRU.")
    return pd.Series(ocupacoes, name="ocupacoes_tru").sort_index()


def _limites_superiores(dados_setor: pd.DataFrame) -> np.ndarray:
    """Interpola o limite superior usando a próxima faixa observada."""

    limites = dados_setor["limite_inferior"].to_numpy(dtype=float)
    superiores = np.empty_like(limites)
    superiores[:-1] = limites[1:] - 1.0
    media_aberta = (
        dados_setor["pessoal_ocupado"].iloc[-1]
        / dados_setor["numero_empresas"].iloc[-1]
    )
    superiores[-1] = max(2.0 * limites[-1], 2.0 * media_aberta)
    return superiores


def _ajustar_total_inteiro_na_faixa(
    valores: np.ndarray,
    total: int,
    minimo: float,
    maximo: float,
) -> np.ndarray:
    """Arredonda sem retirar nenhuma firma de sua faixa de pessoal."""

    limite_inferior = int(np.ceil(minimo))
    limite_superior = int(np.floor(maximo))
    if limite_superior < limite_inferior:
        raise ValueError("A faixa de pessoal não contém valores inteiros válidos.")
    if not len(valores) * limite_inferior <= total <= len(valores) * limite_superior:
        raise ValueError(
            "O pessoal observado é incompatível com o número de firmas e os "
            "limites da faixa."
        )

    resultado = np.clip(
        np.rint(valores).astype(int), limite_inferior, limite_superior
    )
    diferenca = int(total - resultado.sum())
    if diferenca > 0:
        ordem = np.argsort(-(valores - np.floor(valores)), kind="stable")
        for indice in ordem:
            acrescimo = min(diferenca, limite_superior - resultado[indice])
            resultado[indice] += acrescimo
            diferenca -= acrescimo
            if diferenca == 0:
                break
    elif diferenca < 0:
        ordem = np.argsort(valores - np.floor(valores), kind="stable")
        for indice in ordem:
            reducao = min(-diferenca, resultado[indice] - limite_inferior)
            resultado[indice] -= reducao
            diferenca += reducao
            if diferenca == 0:
                break
    if diferenca != 0:
        raise AssertionError("Não foi possível fechar o total dentro da faixa.")
    return resultado


def sortear_pessoal(
    numero_empresas: int,
    pessoal_ocupado: float | None,
    limite_inferior_faixa: float,
    limite_superior_faixa: float,
    distribuicao: str,
    gerador: np.random.Generator,
    alpha_pareto: float = 1.5,
) -> np.ndarray:
    """Interpola tamanhos individuais dentro de uma faixa de emprego.

    A versão Pareto usa o ``alpha`` setorial estimado na cauda. Ambas as
    distribuições são truncadas pela faixa CNAE. Quando ``pessoal_ocupado`` é
    informado, o total também é fechado dentro da própria faixa; quando é
    ``None``, o fechamento ocorre posteriormente na coorte sequencial.
    """

    if numero_empresas < 1:
        return np.array([], dtype=int)
    minimo = max(1.0, float(limite_inferior_faixa))
    maximo = max(minimo, float(limite_superior_faixa))
    if distribuicao == "pareto":
        if not np.isfinite(alpha_pareto) or alpha_pareto <= 0.0:
            raise ValueError("alpha_pareto deve ser positivo e finito.")
        u = gerador.uniform(0.0, 1.0, numero_empresas)
        bruto = minimo / (1.0 - u * (1.0 - (minimo / maximo) ** alpha_pareto)) ** (
            1.0 / alpha_pareto
        )
    elif distribuicao == "lognormal":
        media_log = np.log(np.sqrt(minimo * maximo))
        bruto = gerador.lognormal(mean=media_log, sigma=0.75, size=numero_empresas)
    else:
        raise ValueError("distribuicao deve ser 'pareto' ou 'lognormal'.")
    bruto = np.clip(bruto, minimo, maximo)
    if pessoal_ocupado is None:
        return np.clip(
            np.rint(bruto).astype(int), int(np.ceil(minimo)), int(np.floor(maximo))
        )
    return _ajustar_total_inteiro_na_faixa(
        bruto * float(pessoal_ocupado) / bruto.sum(),
        int(round(pessoal_ocupado)),
        minimo,
        maximo,
    )


def estimar_alpha_pareto(dados_setor: pd.DataFrame) -> float:
    """Estima a cauda de Pareto com as faixas de 30 ou mais ocupados.

    Esta é a regra dos scripts demográficos originais: a regressão usa o log
    da cauda acumulada dos shares de remunerações por faixa. A estimativa é
    setorial e depois é extrapolada para as faixas inferiores, que continuam
    existindo na população sintética mas não entram na regressão por conterem
    limite inferior zero.
    """

    cauda = dados_setor.loc[
        (dados_setor["limite_inferior"] >= 30)
        & (dados_setor["remuneracoes"] > 0.0)
    ].sort_values("limite_inferior")
    if len(cauda) < 2:
        raise ValueError(
            "São necessárias ao menos duas faixas da cauda (limite >= 30) "
            "com remuneração positiva para estimar alpha no setor "
            f"{dados_setor['setor'].iat[0]}."
        )
    remuneracoes = cauda["remuneracoes"].astype(float)
    share_faixa = remuneracoes / remuneracoes.sum()
    cauda_acumulada = share_faixa.iloc[::-1].cumsum().iloc[::-1]
    inclinacao, _ = np.polyfit(
        np.log(cauda["limite_inferior"].to_numpy(dtype=float)),
        np.log(cauda_acumulada.to_numpy(dtype=float)),
        1,
    )
    alpha = float(-inclinacao)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            f"Alpha de Pareto inválido ({alpha}) no setor "
            f"{dados_setor['setor'].iat[0]}."
        )
    return alpha


def calcular_market_shares_individuais(
    firmas: pd.DataFrame,
) -> pd.DataFrame:
    """Define o share individual diretamente pela participação no emprego."""

    resultado = firmas.copy()
    pesos = resultado["pessoal_ocupado_firma"].astype(float)
    soma_pesos_setor = pesos.groupby(resultado["setor"]).transform("sum")
    resultado["market_share_domestico"] = pesos / soma_pesos_setor
    if not np.isclose(resultado["market_share_domestico"].sum(), 1.0):
        raise AssertionError("Os market shares individuais não somam um.")
    return resultado


def calibrar_atributos_mercado(
    firmas: pd.DataFrame,
    eta_preco: float = -1.2,
    eta_qualidade: float = 2.0,
    elasticidade_tamanho: float = 1.0,
) -> pd.DataFrame:
    """Calcula share, preço relativo e qualidade compatíveis com o multilogit.

    Para cada setor, ``share_i ∝ pessoal_i**elasticidade_tamanho``. Com
    ``preco_i = 1/(1-share_i)``, a qualidade é escolhida para que
    ``qualidade_i**eta_qualidade * preco_i**eta_preco = share_i``. Logo, na
    ausência de importados, o multilogit da função de mercado recupera os
    shares sintéticos exatamente (até precisão numérica).
    """

    if eta_preco >= 0.0 or eta_qualidade <= 0.0:
        raise ValueError("eta_preco deve ser negativa e eta_qualidade positiva.")
    if elasticidade_tamanho <= 0.0:
        raise ValueError("elasticidade_tamanho deve ser positiva.")

    resultado = firmas.copy()
    pesos = resultado["pessoal_ocupado_firma"].astype(float) ** elasticidade_tamanho
    resultado["market_share_domestico"] = pesos / pesos.groupby(resultado["setor"]).transform("sum")
    resultado["preco_relativo"] = 1.0 / (1.0 - resultado["market_share_domestico"])
    return calibrar_qualidade_implicita(
        resultado, eta_preco=eta_preco, eta_qualidade=eta_qualidade
    )


def calibrar_qualidade_implicita(
    firmas: pd.DataFrame,
    eta_preco: float = -1.2,
    eta_qualidade: float = 2.0,
    peso_variedade: pd.Series | None = None,
    coluna_saida: str = "qualidade",
) -> pd.DataFrame:
    """Obtém a qualidade que reproduz share e preço de cada linha.

    Após a normalização geométrica por setor, o multilogit continua
    reproduzindo os shares, pois a normalização multiplica a atratividade de
    todas as linhas do setor pelo mesmo fator.
    """

    resultado = firmas.copy()
    if (resultado["market_share_domestico"] <= 0.0).any():
        raise ValueError("Todo share doméstico deve ser estritamente positivo.")
    if (resultado["preco_relativo"] <= 0.0).any():
        raise ValueError("Todo preço relativo deve ser estritamente positivo.")
    if peso_variedade is None:
        pesos = pd.Series(1.0, index=resultado.index)
    else:
        pesos = pd.Series(peso_variedade, index=resultado.index, dtype=float)
        if (pesos <= 0.0).any():
            raise ValueError("Todo peso de variedade deve ser estritamente positivo.")

    resultado[coluna_saida] = (
        resultado["market_share_domestico"] / pesos
        / resultado["preco_relativo"] ** eta_preco
    ) ** (1.0 / eta_qualidade)

    # A normalização por setor não altera shares domésticos: multiplica todas
    # as atratividades pelo mesmo fator. Quando há peso de variedade, a média
    # geométrica é ponderada pelo número de firmas representadas.
    log_qualidade = np.log(resultado[coluna_saida])
    soma_pesos = pesos.groupby(resultado["setor"]).transform("sum")
    media_log = (log_qualidade * pesos).groupby(resultado["setor"]).transform("sum") / soma_pesos
    resultado[coluna_saida] /= np.exp(media_log)
    return resultado


def normalizar_coortes_para_abm(
    coortes: pd.DataFrame,
    eta_preco: float = -1.2,
    eta_qualidade: float = 2.0,
) -> pd.DataFrame:
    """Calibra coortes no ano-base conforme a tabela de referência.

    O preço básico de todas as coortes no ano-base é um. A heterogeneidade
    necessária para reproduzir os shares domésticos vem exclusivamente da
    qualidade, isto é, ``qualidade_i ** eta_qualidade = share_i``. Assim,
    quantidade, receita e todos os componentes de custo da firma podem ser
    repartidos pelo mesmo share sem romper a identidade contábil individual.
    """

    resultado = coortes.copy()
    colunas = {"setor", "market_share_domestico", "preco_relativo"}
    faltantes = colunas.difference(resultado.columns)
    if faltantes:
        raise KeyError(f"Coortes sem colunas obrigatórias: {sorted(faltantes)}")
    if eta_preco >= 0.0 or eta_qualidade <= 0.0:
        raise ValueError("eta_preco deve ser negativa e eta_qualidade positiva.")

    for setor, indice in resultado.groupby("setor", sort=False).groups.items():
        shares = resultado.loc[indice, "market_share_domestico"].astype(float)
        if (shares <= 0.0).any() or not np.isclose(shares.sum(), 1.0):
            raise ValueError(f"Shares inválidos no setor {setor}.")
        resultado.loc[indice, "preco_relativo"] = 1.0
        resultado.loc[indice, "qualidade"] = shares ** (1.0 / eta_qualidade)
    return resultado


def agrupar_em_coortes(
    firmas: pd.DataFrame,
    tamanho_coorte: int,
    semente: int | None = None,
    eta_preco: float = -1.2,
    eta_qualidade: float = 2.0,
    ocupacoes_demografia: pd.Series | None = None,
    ocupacoes_tru: pd.Series | None = None,
) -> pd.DataFrame:
    """Agrega blocos sequenciais de firmas, preservando a hierarquia de porte.

    As firmas já foram geradas por faixa. Elas são ordenadas da menor para a
    maior faixa e, dentro de cada faixa, do menor para o maior porte antes de
    receberem blocos consecutivos de até ``tamanho_coorte``. Logo uma coorte
    pode cruzar a fronteira de duas faixas, mas nunca mistura firmas aleatórias.
    ``semente`` é mantida apenas por compatibilidade de chamada e não é usada.
    """

    if tamanho_coorte < 1:
        raise ValueError("tamanho_coorte deve ser inteiro positivo.")

    coortes: list[pd.DataFrame] = []
    for setor, grupo in firmas.groupby("setor", sort=True):
        grupo = grupo.sort_values(
            ["faixa_ordem", "pessoal_ocupado_firma", "id_firma"],
            kind="stable",
        ).reset_index(drop=True)
        grupo["_coorte"] = np.arange(len(grupo)) // tamanho_coorte
        grupo["_preco_ponderado"] = (
            grupo["market_share_domestico"] * grupo["preco_relativo"]
        )
        grupo["_qualidade_ponderada"] = (
            grupo["market_share_domestico"] * grupo["qualidade"]
        )
        agregado = grupo.groupby("_coorte", sort=True).agg(
            numero_firmas_representadas=("id_firma", "size"),
            pessoal_ocupado_demografia=("pessoal_ocupado_firma", "sum"),
            market_share_domestico=("market_share_domestico", "sum"),
            _preco_ponderado=("_preco_ponderado", "sum"),
            _qualidade_ponderada=("_qualidade_ponderada", "sum"),
            pessoal_ocupado_minimo_original=("pessoal_ocupado_firma", "min"),
            pessoal_ocupado_maximo_original=("pessoal_ocupado_firma", "max"),
            faixa_inicial=("faixa_pessoal", "first"),
            faixa_final=("faixa_pessoal", "last"),
        )
        agregado["preco_relativo"] = (
            agregado.pop("_preco_ponderado") / agregado["market_share_domestico"]
        )
        agregado["qualidade_media_ponderada"] = (
            agregado.pop("_qualidade_ponderada")
            / agregado["market_share_domestico"]
        )
        agregado["pessoal_ocupado_medio_demografia_por_firma"] = (
            agregado["pessoal_ocupado_demografia"]
            / agregado["numero_firmas_representadas"]
        )
        agregado["pessoal_ocupado_firma"] = agregado["pessoal_ocupado_demografia"]
        agregado["pessoal_ocupado_medio_tru_por_firma"] = (
            agregado["pessoal_ocupado_medio_demografia_por_firma"]
        )
        # Nome antigo mantido como metadado compatível e sempre demográfico.
        agregado["pessoal_ocupado_medio_por_firma"] = (
            agregado["pessoal_ocupado_medio_demografia_por_firma"]
        )
        agregado["multiplicador_tru"] = 1.0
        agregado = agregado.reset_index(drop=True)
        agregado.insert(0, "setor", setor)
        agregado.insert(
            1,
            "id_firma",
            [f"{setor}_C{i:06d}" for i in range(1, len(agregado) + 1)],
        )
        agregado.insert(
            2,
            "faixa_pessoal",
            [
                inicio if inicio == fim else f"{inicio} a {fim}"
                for inicio, fim in zip(agregado["faixa_inicial"], agregado["faixa_final"])
            ],
        )
        agregado["peso_variedade"] = agregado["numero_firmas_representadas"]
        agregado["tipo_agente"] = "coorte_firmas"
        coortes.append(agregado)

    resultado = pd.concat(coortes, ignore_index=True)
    if ocupacoes_demografia is not None:
        resultado = compatibilizar_emprego_demografia_coortes(
            resultado, ocupacoes_demografia
        )
    if ocupacoes_tru is not None:
        resultado = compatibilizar_emprego_tru_coortes(resultado, ocupacoes_tru)
    resultado = calibrar_qualidade_implicita(
        resultado, eta_preco=eta_preco, eta_qualidade=eta_qualidade
    )
    resultado["qualidade_efetiva_coorte"] = resultado["qualidade"]
    return calibrar_qualidade_implicita(
        resultado,
        eta_preco=eta_preco,
        eta_qualidade=eta_qualidade,
        peso_variedade=resultado["peso_variedade"],
        coluna_saida="qualidade_por_firma",
    )


def compatibilizar_emprego_demografia_coortes(
    coortes: pd.DataFrame,
    ocupacoes_demografia: pd.Series,
) -> pd.DataFrame:
    """Fecha o emprego da Demografia por setor após as coortes sequenciais."""

    resultado = coortes.copy()
    for setor, indice in resultado.groupby("setor", sort=False).groups.items():
        emprego_bruto = int(resultado.loc[indice, "pessoal_ocupado_firma"].sum())
        emprego_observado = int(round(float(ocupacoes_demografia.at[setor])))
        if emprego_bruto <= 0 or emprego_observado <= 0:
            raise ValueError(f"Emprego inválido para compatibilizar o setor {setor}.")
        multiplicador = emprego_observado / emprego_bruto
        ajustado = np.rint(
            resultado.loc[indice, "pessoal_ocupado_firma"].to_numpy(dtype=float)
            * multiplicador
        ).astype(int)
        ajustado[-1] += emprego_observado - int(ajustado.sum())
        if (ajustado <= 0).any():
            raise ValueError(
                f"O fechamento demográfico deixou coorte sem emprego no setor {setor}."
            )
        resultado.loc[indice, "pessoal_ocupado_demografia"] = ajustado
        resultado.loc[indice, "pessoal_ocupado_firma"] = ajustado
        resultado.loc[indice, "pessoal_ocupado_medio_demografia_por_firma"] = (
            ajustado / resultado.loc[indice, "numero_firmas_representadas"].to_numpy()
        )
        resultado.loc[indice, "pessoal_ocupado_medio_por_firma"] = (
            resultado.loc[indice, "pessoal_ocupado_medio_demografia_por_firma"]
        )
    return resultado


def compatibilizar_emprego_tru_coortes(
    coortes: pd.DataFrame,
    ocupacoes_tru: pd.Series,
) -> pd.DataFrame:
    """Escala emprego somente após formar as coortes, sem mexer nos shares."""

    resultado = coortes.copy()
    for setor, indice in resultado.groupby("setor", sort=False).groups.items():
        if setor not in ocupacoes_tru.index:
            raise KeyError(f"A TRU não contém ocupações para o setor {setor}.")
        emprego_demografia = int(
            resultado.loc[indice, "pessoal_ocupado_demografia"].sum()
        )
        emprego_tru = int(ocupacoes_tru.at[setor])
        if emprego_demografia <= 0 or emprego_tru <= 0:
            raise ValueError(f"Emprego inválido para compatibilizar o setor {setor}.")
        multiplicador = emprego_tru / emprego_demografia
        ajustado = np.rint(
            resultado.loc[indice, "pessoal_ocupado_demografia"].to_numpy(dtype=float)
            * multiplicador
        ).astype(int)
        ajustado[-1] += emprego_tru - int(ajustado.sum())
        if (ajustado <= 0).any():
            raise ValueError(
                f"A compatibilização TRU deixou coorte sem emprego no setor {setor}."
            )
        resultado.loc[indice, "pessoal_ocupado_firma"] = ajustado
        resultado.loc[indice, "pessoal_ocupado_medio_tru_por_firma"] = (
            ajustado / resultado.loc[indice, "numero_firmas_representadas"].to_numpy()
        )
        resultado.loc[indice, "multiplicador_tru"] = multiplicador
    return resultado


def gerar_firmas_sinteticas(
    dados: pd.DataFrame,
    distribuicao: str,
    semente: int | None,
    ocupacoes_tru: pd.Series | None = None,
    tamanho_coorte: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gera a população observada e só então a compatibiliza com a TRU.

    ``alpha`` é estimado em cada setor apenas com as faixas cujo limite
    inferior é ao menos 30. O mesmo ``alpha`` é extrapolado para gerar todas
    as faixas observadas, inclusive 0--4 (tratada como [1, 5)). A restrição de
    cauda serve exclusivamente à estimação: nenhuma faixa observada é retirada
    da população de firmas sintéticas.
    """

    gerador = np.random.default_rng(semente)
    linhas: list[pd.DataFrame] = []
    resumo: list[dict[str, object]] = []

    for setor, grupo in dados.groupby("setor", sort=True):
        if ocupacoes_tru is not None and setor not in ocupacoes_tru.index:
            continue
        grupo = grupo.sort_values("limite_inferior").reset_index(drop=True)
        limites_superiores = _limites_superiores(grupo)
        alpha_pareto = (
            estimar_alpha_pareto(grupo)
            if distribuicao == "pareto"
            else np.nan
        )
        firmas_faixas: list[pd.DataFrame] = []
        for indice, linha in grupo.iterrows():
            limite_inferior_faixa = max(1.0, float(linha["limite_inferior"]))
            limite_superior_faixa = float(limites_superiores[indice])
            pessoal = sortear_pessoal(
                int(linha["numero_empresas"]),
                None,
                limite_inferior_faixa,
                limite_superior_faixa,
                distribuicao,
                gerador,
                alpha_pareto=alpha_pareto,
            )
            firmas_faixas.append(
                pd.DataFrame(
                    {
                        "setor": setor,
                        "faixa_pessoal": linha[COLUNA_FAIXA],
                        "faixa_ordem": indice,
                        "pessoal_ocupado_firma": pessoal,
                    }
                )
            )
        firmas_setor = pd.concat(firmas_faixas, ignore_index=True)
        firmas_setor.insert(
            1,
            "id_firma",
            [f"{setor}_{i:06d}" for i in range(1, len(firmas_setor) + 1)],
        )
        firmas_setor = calcular_market_shares_individuais(firmas_setor)
        firmas_setor["preco_relativo"] = 1.0 / (
            1.0 - firmas_setor["market_share_domestico"]
        )
        firmas_setor = calibrar_qualidade_implicita(firmas_setor)
        pessoal_demografia = int(round(float(grupo["pessoal_ocupado"].sum())))
        pessoal_tru = (
            pessoal_demografia
            if ocupacoes_tru is None
            else int(ocupacoes_tru.at[setor])
        )
        linhas.append(firmas_setor)
        resumo.append(
            {
                "setor": setor,
                "numero_empresas": len(firmas_setor),
                "pessoal_ocupado_demografia": pessoal_demografia,
                "pessoal_ocupado_tru": pessoal_tru,
                "alpha_pareto": alpha_pareto,
                "beta_market_share": 1.0,
            }
        )

    firmas = pd.concat(linhas, ignore_index=True)
    return pd.DataFrame(resumo), agrupar_em_coortes(
        firmas,
        tamanho_coorte=tamanho_coorte,
        semente=semente,
        ocupacoes_demografia=dados.groupby("setor")["pessoal_ocupado"].sum(),
        ocupacoes_tru=ocupacoes_tru,
    )


def coorte_setor_t(ocupacoes_tru: int) -> pd.DataFrame:
    """Representa T como um único bloco agregado de serviços domésticos.

    O share unitário torna a fórmula geral ``1 / (1 - share)`` indefinida.
    Por isso T recebe preço e qualidade normalizados a um: não participa da
    competição intrassetorial usual e preserva somente seus agregados reais.
    """

    return pd.DataFrame(
        {
            "setor": ["T"],
            "id_firma": ["T_C000001"],
            "faixa_pessoal": ["coorte agregada de serviços domésticos"],
            "faixa_inicial": ["coorte agregada de serviços domésticos"],
            "faixa_final": ["coorte agregada de serviços domésticos"],
            "numero_firmas_representadas": [1],
            "pessoal_ocupado_demografia": [ocupacoes_tru],
            "pessoal_ocupado_firma": [ocupacoes_tru],
            "pessoal_ocupado_medio_demografia_por_firma": [np.nan],
            "pessoal_ocupado_medio_tru_por_firma": [np.nan],
            "pessoal_ocupado_medio_por_firma": [np.nan],
            "pessoal_ocupado_minimo_original": [ocupacoes_tru],
            "pessoal_ocupado_maximo_original": [ocupacoes_tru],
            "multiplicador_tru": [1.0],
            "peso_variedade": [1],
            "tipo_agente": ["bloco_setorial"],
            "market_share_domestico": [1.0],
            "preco_relativo": [1.0],
            "qualidade_media_ponderada": [1.0],
            "qualidade": [1.0],
            "qualidade_efetiva_coorte": [1.0],
            "qualidade_por_firma": [1.0],
        }
    )


def diagnosticar_coortes_demografia(
    resumo: pd.DataFrame,
    coortes: pd.DataFrame,
) -> pd.DataFrame:
    """Produz os controles de contagem, emprego, share e porte por setor."""

    registros: list[dict[str, object]] = []
    for setor, grupo in coortes.groupby("setor", sort=True):
        grupo = grupo.sort_values("id_firma", kind="stable")
        referencia = resumo.loc[resumo["setor"] == setor]
        numero_observado = (
            int(referencia["numero_empresas"].iat[0]) if len(referencia) else 1
        )
        emprego_demografia = float(grupo["pessoal_ocupado_demografia"].sum())
        emprego_tru = float(grupo["pessoal_ocupado_firma"].sum())
        shares = grupo["market_share_domestico"].astype(float)
        media = grupo["pessoal_ocupado_medio_demografia_por_firma"].astype(float)
        if grupo["id_firma"].duplicated().any():
            raise AssertionError(f"IDs de coorte duplicados no setor {setor}.")
        if not np.isclose(shares.sum(), 1.0):
            raise AssertionError(f"Shares das coortes não somam um no setor {setor}.")
        if (grupo["pessoal_ocupado_firma"] <= 0).any():
            raise AssertionError(f"Há emprego não positivo no setor {setor}.")
        registros.append(
            {
                "setor": setor,
                "numero_firmas_demografia": numero_observado,
                "numero_firmas_sinteticas": int(grupo["numero_firmas_representadas"].sum()),
                "numero_coortes": len(grupo),
                "emprego_demografia": emprego_demografia,
                "emprego_tru": emprego_tru,
                "multiplicador_tru": emprego_tru / emprego_demografia,
                "share_domestico_total": float(shares.sum()),
                "porte_medio_inicio": float(media.iloc[0]),
                "porte_medio_mediana": float(media.median()),
                "porte_medio_final": float(media.iloc[-1]),
                "pessoal_minimo_original": float(
                    grupo["pessoal_ocupado_minimo_original"].min()
                ),
                "pessoal_maximo_original": float(
                    grupo["pessoal_ocupado_maximo_original"].max()
                ),
            }
        )
    return pd.DataFrame(registros)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arquivo",
        type=Path,
        default=Path(__file__).with_name("Demografia_Empresas.xlsx"),
    )
    parser.add_argument("--aba", default="Planilha1")
    parser.add_argument("--arquivo-tru", type=Path, default=ARQUIVO_TRU_PADRAO)
    parser.add_argument("--aba-tru", default="VA")
    parser.add_argument("--distribuicao", choices=("lognormal", "pareto"), default="lognormal")
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument(
        "--tamanho-coorte",
        type=int,
        default=100,
        help="Número máximo de firmas reais sorteadas e agregadas em cada coorte.",
    )
    argumentos = parser.parse_args()
    if argumentos.tamanho_coorte < 1:
        raise ValueError("--tamanho-coorte deve ser inteiro positivo.")

    ocupacoes_tru = ler_ocupacoes_tru(argumentos.arquivo_tru, argumentos.aba_tru)
    dados = ler_demografia(argumentos.arquivo, argumentos.aba)
    resumo, firmas = gerar_firmas_sinteticas(
        dados,
        argumentos.distribuicao,
        argumentos.semente,
        ocupacoes_tru,
        tamanho_coorte=argumentos.tamanho_coorte,
    )
    pasta = argumentos.arquivo.parent
    firmas = pd.concat(
        [firmas, coorte_setor_t(int(ocupacoes_tru.at["T"]))], ignore_index=True
    )
    caminho_firmas = pasta / "coortes_firmas_tru_2020.csv"
    firmas.to_csv(caminho_firmas, index=False)
    resumo = pd.concat(
        [
            resumo,
            pd.DataFrame(
                [{
                    "setor": "T",
                    "numero_empresas": np.nan,
                    "numero_coortes": 1,
                    "pessoal_ocupado_demografia": np.nan,
                    "pessoal_ocupado_tru": int(ocupacoes_tru.at["T"]),
                }]
            ),
        ],
        ignore_index=True,
    ).sort_values("setor")
    resumo["numero_coortes"] = resumo["setor"].map(
        firmas.groupby("setor")["id_firma"].size()
    )
    resumo.to_csv(pasta / "resumo_coortes_firmas_tru_2020.csv", index=False)
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
