"""Calibração histórica do investimento das firmas não financeiras."""

from pathlib import Path

import numpy as np
import pandas as pd

from contabilidade.cei_abm.tru_sector_sector import (
    load_tru_data,
    transform_tru_to_sector_sector,
)
from contabilidade.estrutura_cei import C, L


def calibrar_investimento_nf(
    data_dir: Path,
    arquivo_cei: Path,
    *,
    ano_base: int = 2020,
    nivel: int = 20,
    aba_cei: str = "Python",
    vida_util_capital: float = 20.0,
    ano_inicial_beta: int = 2010,
    ano_final_beta: int | None = None,
    setor_financeiro: str = (
        "K - Atividades financeiras, de seguros e serviços relacionados"
    ),
    setores_excluidos: list[str] | tuple[str, ...] | None = None,
    setor_construcao: str = "F - Construção",
) -> dict:
    """Calibra beta, v, estoques iniciais e a composição da FBCF das NF."""

    depreciacao = 1.0 / vida_util_capital
    if ano_final_beta is None:
        # O comportamento é estimado somente até o ano anterior ao ano-base.
        # Assim, a pandemia de 2020 não é usada para calibrar a persistência.
        ano_final_beta = ano_base - 1
    if ano_inicial_beta + 2 > ano_final_beta or ano_final_beta > ano_base:
        raise ValueError("Janela inválida para estimar beta.")

    anos = list(range(ano_inicial_beta, ano_base + 1))

    # ------------------------------------------------------------------
    # Produção corrente: tabelas 1 e 2 da TRU
    # ------------------------------------------------------------------
    dados_tru = {}
    setores = None
    producao_corrente = None

    for ano in anos:
        dados = load_tru_data(data_dir, year=ano, level=nivel)
        dados_tru[ano] = dados

        if setores is None:
            setores = list(dados.sector_names)
            producao_corrente = pd.DataFrame(
                0.0,
                index=pd.Index(anos, name="ano"),
                columns=pd.Index(setores, name="setor"),
            )
        elif list(dados.sector_names) != setores:
            raise ValueError(f"Os setores da TRU de {ano} não coincidem.")

        producao_corrente.loc[ano] = dados.production.sum(axis=0)

    # ------------------------------------------------------------------
    # Crescimento real: produção de t avaliada aos preços de t-1
    # ------------------------------------------------------------------
    producao_precos_ano_anterior = pd.DataFrame(
        0.0,
        index=pd.Index(anos[1:], name="ano"),
        columns=pd.Index(setores, name="setor"),
    )

    for ano in anos[1:]:
        arquivo_tab3 = data_dir / f"{nivel}_tab3_{ano}.xlsx"
        tabela = pd.read_excel(
            arquivo_tab3,
            sheet_name="producao",
            header=None,
            engine="openpyxl",
        )
        numero_produtos, numero_setores = dados_tru[ano].production.shape
        bloco = (
            tabela.iloc[
                5 : 5 + numero_produtos,
                2 : 2 + numero_setores,
            ]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        # ``bloco.sum()`` tem índice numérico, enquanto as colunas da tabela
        # têm nomes de setores. A conversão posicional evita que o pandas tente
        # alinhar esses rótulos diferentes e preencha a linha com NaN.
        producao_precos_ano_anterior.loc[ano] = (
            bloco.sum(axis=0).to_numpy(dtype=float)
        )

    fator_volume = pd.DataFrame(
        0.0,
        index=producao_precos_ano_anterior.index,
        columns=producao_precos_ano_anterior.columns,
    )
    for ano in anos[1:]:
        fator_volume.loc[ano] = (
            producao_precos_ano_anterior.loc[ano]
            / producao_corrente.loc[ano - 1].replace(0.0, np.nan)
        ).fillna(1.0)
        if (fator_volume.loc[ano] <= 0.0).any():
            raise ValueError(f"Fator de volume não positivo em {ano}.")

    # Encadeamento para trás, com a produção corrente do ano-base como âncora.
    producao_real = pd.DataFrame(
        0.0,
        index=producao_corrente.index,
        columns=producao_corrente.columns,
    )
    producao_real.loc[ano_base] = producao_corrente.loc[ano_base]
    for ano in reversed(anos[1:]):
        producao_real.loc[ano - 1] = (
            producao_real.loc[ano] / fator_volume.loc[ano]
        )

    # Setores de atividade que não representam firmas não financeiras
    # investidoras. Esta exclusão vale apenas para o lado do INVESTIDOR: eles
    # ainda podem aparecer como fornecedores na cesta de bens de capital.
    if setores_excluidos is None:
        setores_excluidos = [setor_financeiro]
    else:
        setores_excluidos = list(setores_excluidos)

    setores_nao_encontrados = [
        setor for setor in setores_excluidos if setor not in setores
    ]
    if setores_nao_encontrados:
        raise KeyError(
            "Setores excluídos não encontrados na TRU: "
            f"{setores_nao_encontrados}"
        )
    setores_nf = [
        setor for setor in setores if setor not in setores_excluidos
    ]

    # ------------------------------------------------------------------
    # Beta comum: ΔY_t = beta * ΔY_(t-1), sem intercepto
    # ------------------------------------------------------------------
    variacao_producao = producao_real.loc[:, setores_nf].diff()
    anos_regressao = list(
        range(ano_inicial_beta + 2, ano_final_beta + 1)
    )
    delta_y = variacao_producao.loc[anos_regressao].to_numpy().ravel()
    delta_y_anterior = (
        variacao_producao.shift(1).loc[anos_regressao].to_numpy().ravel()
    )

    denominador_beta = float(delta_y_anterior @ delta_y_anterior)
    if denominador_beta == 0.0:
        raise ValueError("Não foi possível estimar beta: regressor sem variação.")
    beta = float(delta_y_anterior @ delta_y / denominador_beta)

    residuo_beta = delta_y - beta * delta_y_anterior
    r2_beta_sem_constante = 1.0 - float(
        (residuo_beta @ residuo_beta) / (delta_y @ delta_y)
    )

    # O painel acima atribui o mesmo beta aos 19 setores. A estimativa agregada
    # fica disponível apenas como comparação; ela não é usada na simulação.
    variacao_agregada = producao_real.loc[:, setores_nf].sum(axis=1).diff()
    delta_y_agregado = variacao_agregada.loc[anos_regressao].to_numpy()
    delta_y_agregado_anterior = (
        variacao_agregada.shift(1).loc[anos_regressao].to_numpy()
    )
    beta_agregado = float(
        (delta_y_agregado_anterior @ delta_y_agregado)
        / (delta_y_agregado_anterior @ delta_y_agregado_anterior)
    )

    # ------------------------------------------------------------------
    # FBCF institucional observada na CEI
    # ------------------------------------------------------------------
    planilha = pd.read_excel(arquivo_cei, sheet_name=aba_cei)
    linhas_validas = planilha.iloc[:, 1:].fillna(0).ne(0).any(axis="columns")
    valores = (
        planilha.loc[linhas_validas, planilha.columns[:11]]
        .copy()
        .reset_index(drop=True)
    )
    valores.iloc[:, 1:11] = (
        valores.iloc[:, 1:11]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    fbcf_nf = float(valores.iat[L["fbcf"], C["nf_s"]])
    fbcf_familias = float(valores.iat[L["fbcf"], C["familias_s"]])

    # ------------------------------------------------------------------
    # Calibração de v com investimento líquido + reposição
    # ------------------------------------------------------------------
    # A variação da produção esperada para o ano-base é formada com os dados
    # conhecidos no ano anterior:
    #   ΔY_e,t = beta * (Y_(t-1) - Y_(t-2))
    variacao_producao_esperada_base = (
        beta
        * (
            producao_real.loc[ano_base - 1]
            - producao_real.loc[ano_base - 2]
        )
    ).rename("variacao_producao_esperada_base")
    producao_esperada_base = (
        producao_real.loc[ano_base - 1]
        + variacao_producao_esperada_base
    ).rename("producao_esperada_base")

    # No ano anterior, K_(t-1) = v * Y_(t-1). Logo:
    #   I_líquido,t = v * ΔY_e,t
    #   I_reposição,t = depreciação * v * Y_(t-1)
    #   I_bruto,t = v * [ΔY_e,t + depreciação * Y_(t-1)]
    # O piso zero é aplicado por setor também na calibração.
    base_calibracao_v_sem_piso = (
        variacao_producao_esperada_base.loc[setores_nf]
        + depreciacao * producao_real.loc[ano_base - 1, setores_nf]
    ).rename("base_calibracao_v_sem_piso")
    base_calibracao_v = base_calibracao_v_sem_piso.clip(lower=0.0).rename(
        "base_calibracao_v"
    )

    denominador_v = float(base_calibracao_v.sum())
    if fbcf_nf <= 0.0 or denominador_v <= 0.0:
        raise ValueError("Não foi possível calibrar um v positivo.")
    v = fbcf_nf / denominador_v

    estoque_anterior = (
        v * producao_real.loc[ano_base - 1, setores_nf]
    ).rename("estoque_capital_nf_anterior")
    investimento_liquido_base = (
        v * variacao_producao_esperada_base.loc[setores_nf]
    ).rename("investimento_liquido_nf_base")
    investimento_reposicao_base = (
        depreciacao * estoque_anterior
    ).rename("investimento_reposicao_nf_base")
    investimento_sem_piso_base = (
        investimento_liquido_base + investimento_reposicao_base
    ).rename("investimento_nf_sem_piso_base")
    investimento_por_investidor = (
        investimento_sem_piso_base.clip(lower=0.0)
    ).rename("investimento_nf_por_setor_investidor")
    estoque_base = (
        (1.0 - depreciacao) * estoque_anterior
        + investimento_por_investidor
    ).rename("estoque_capital_nf_base")

    if not np.isclose(investimento_por_investidor.sum(), fbcf_nf, atol=1e-6):
        raise RuntimeError("A calibração não reproduziu a FBCF NF observada.")

    # ------------------------------------------------------------------
    # Composição dos bens de capital comprados pelas NF
    # ------------------------------------------------------------------
    tru_base = transform_tru_to_sector_sector(
        dados_tru[ano_base],
        validate=False,
    )
    fbcf_tru = tru_base.gross_investment_sector.iloc[:, 0]
    estoques_tru = tru_base.stocks_investment_sector.iloc[:, 0].rename(
        "estoques_base"
    )

    fbcf_familias_fornecedor = pd.Series(
        0.0,
        index=pd.Index(setores, name="setor_fornecedor"),
        name="fbcf_familias_base",
    )
    if setor_construcao not in setores:
        raise KeyError(f"Setor de Construção não encontrado: {setor_construcao}")
    fbcf_familias_fornecedor.loc[setor_construcao] = fbcf_familias

    residual_sem_familias = fbcf_tru - fbcf_familias_fornecedor
    if (residual_sem_familias < -1e-8).any():
        raise ValueError("A FBCF das famílias supera a FBCF da TRU.")
    residual_sem_familias = residual_sem_familias.clip(lower=0.0)
    pesos_bens_capital = (
        residual_sem_familias / residual_sem_familias.sum()
    ).rename("peso_bens_capital_nf")
    fbcf_nf_fornecedor = (pesos_bens_capital * fbcf_nf).rename(
        "fbcf_nf_base"
    )
    fbcf_outros = (
        fbcf_tru - fbcf_familias_fornecedor - fbcf_nf_fornecedor
    ).rename("fbcf_outros_base")

    fechamento_fbcf = float(
        abs(
            fbcf_tru.sum()
            - fbcf_familias_fornecedor.sum()
            - fbcf_nf_fornecedor.sum()
            - fbcf_outros.sum()
        )
    )
    if fechamento_fbcf > 1e-6 or (fbcf_outros < -1e-8).any():
        raise RuntimeError("A decomposição da FBCF não fechou.")

    return {
        "beta": beta,
        "beta_agregado": beta_agregado,
        "v": v,
        "depreciacao": depreciacao,
        "setores_nf": setores_nf,
        "setores_excluidos": setores_excluidos,
        "producao_real": producao_real,
        "producao_esperada_base": producao_esperada_base,
        "variacao_producao_esperada_base": variacao_producao_esperada_base,
        "base_calibracao_v": base_calibracao_v,
        "estoque_capital_nf_anterior": estoque_anterior,
        "estoque_capital_nf_base": estoque_base,
        "investimento_liquido_nf_base_por_investidor": investimento_liquido_base,
        "investimento_reposicao_nf_base_por_investidor": investimento_reposicao_base,
        "investimento_nf_base_por_investidor": investimento_por_investidor,
        "pesos_bens_capital_nf": pesos_bens_capital,
        "fbcf_nf_base_fornecedor": fbcf_nf_fornecedor,
        "fbcf_familias_base_fornecedor": fbcf_familias_fornecedor,
        "fbcf_outros_base": fbcf_outros,
        "estoques_base": estoques_tru,
        "diagnosticos": pd.Series(
            {
                "beta_painel_calibrado": beta,
                "beta_agregado_comparacao": beta_agregado,
                "r2_beta_sem_constante": r2_beta_sem_constante,
                "observacoes_beta": len(delta_y),
                "primeiro_ano_beta": anos_regressao[0],
                "ultimo_ano_beta": anos_regressao[-1],
                "v_calibrado": v,
                "depreciacao": depreciacao,
                "investimento_liquido_nf_base": float(
                    investimento_liquido_base.sum()
                ),
                "investimento_reposicao_nf_base": float(
                    investimento_reposicao_base.sum()
                ),
                "setores_no_piso_ano_base": int(
                    (investimento_sem_piso_base < 0.0).sum()
                ),
                "ajuste_piso_ano_base": float(
                    (
                        investimento_por_investidor
                        - investimento_sem_piso_base
                    ).sum()
                ),
                "fbcf_nf_cei": fbcf_nf,
                "fbcf_nf_reproduzida": float(investimento_por_investidor.sum()),
                "fbcf_total_tru": float(fbcf_tru.sum()),
                "fbcf_outros_autonoma": float(fbcf_outros.sum()),
                "fechamento_decomposicao_fbcf": fechamento_fbcf,
            },
            name="valor",
        ),
    }
