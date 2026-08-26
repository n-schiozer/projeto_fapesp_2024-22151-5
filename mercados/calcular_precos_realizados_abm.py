import pandas as pd
import numpy as np

def calcular_precos_realizados_abm(
    setores: list,
    firmas: dict,
    importados: dict,
    G: pd.DataFrame,
) -> dict:
    """Constrói os preços realizados após os mercados do período.

    Pb é o preço médio das transações domésticas.
    Pm é o preço médio das importações realizadas.
    O preço do produto combina as duas origens pelas quantidades
    efetivamente transacionadas.
    Pc adiciona impostos e margens da estrutura da TRU.
    """

    pb_realizado = pd.Series(
        index=setores,
        dtype=float,
        name="preco_basico_realizado",
    )

    pm_realizado = pd.Series(
        index=setores,
        dtype=float,
        name="preco_importado_realizado",
    )

    preco_produto_realizado = pd.Series(
        index=setores,
        dtype=float,
        name="preco_produto_realizado",
    )


    for setor in setores:

        firmas_setor = [
            firma
            for firma in firmas.values()
            if firma.setor == setor
        ]

        importado = importados[setor]

        # ======================================================
        # PRODUÇÃO DOMÉSTICA
        # ======================================================

        vendas_domesticas_real = float(
            sum(
                firma.vendas_real
                for firma in firmas_setor
            )
        )

        vendas_domesticas_nominal = float(
            sum(
                firma.vendas_nominal
                for firma in firmas_setor
            )
        )

        if vendas_domesticas_real > 0.0:

            pb_realizado.at[setor] = (
                vendas_domesticas_nominal
                / vendas_domesticas_real
            )

        else:

            # Não existe Pb observado sem transação doméstica.
            # O valor abaixo é apenas uma referência para manter
            # a série definida; ele não entra no preço composto
            # se a quantidade doméstica realizada for zero.

            producao_domestica = float(
                sum(
                    firma.producao_real
                    for firma in firmas_setor
                )
            )

            if producao_domestica > 0.0:

                pb_realizado.at[setor] = (
                    sum(
                        firma.preco_transacao
                        * firma.producao_real
                        for firma in firmas_setor
                    )
                    / producao_domestica
                )

            else:
                pb_realizado.at[setor] = np.nan


        # ======================================================
        # IMPORTAÇÕES
        # ======================================================

        importacoes_real = float(
            importado.vendas_real
        )

        importacoes_nominal = float(
            importado.vendas_nominal
        )

        if importacoes_real > 0.0:

            pm_realizado.at[setor] = (
                importacoes_nominal
                / importacoes_real
            )

        else:

            # Também é apenas preço de referência quando
            # não houve importação efetiva.
            pm_realizado.at[setor] = (
                importado.preco
            )


        # ======================================================
        # PREÇO REALIZADO DO PRODUTO
        # ======================================================

        quantidade_realizada = (
            vendas_domesticas_real
            + importacoes_real
        )

        valor_realizado = (
            vendas_domesticas_nominal
            + importacoes_nominal
        )

        if quantidade_realizada <= 0.0:
            raise RuntimeError(
                "Não houve transação no mercado do setor "
                f"{setor}."
            )

        preco_produto_realizado.at[setor] = (
            valor_realizado
            / quantidade_realizada
        )


    # ==========================================================
    # PREÇO DE COMPRADOR
    # ==========================================================

    pc_realizado = (
        G @ preco_produto_realizado
    ).rename(
        "preco_comprador_realizado"
    )

    if (
        preco_produto_realizado.isna().any()
        or pc_realizado.isna().any()
    ):
        raise RuntimeError(
            "Há preços realizados indefinidos."
        )

    if (
        (preco_produto_realizado <= 0.0).any()
        or (pc_realizado <= 0.0).any()
    ):
        raise RuntimeError(
            "Há preços realizados não positivos."
        )


    return {
        "pb": pb_realizado,
        "pm": pm_realizado,
        "preco_produto": preco_produto_realizado,
        "pc": pc_realizado,
    }