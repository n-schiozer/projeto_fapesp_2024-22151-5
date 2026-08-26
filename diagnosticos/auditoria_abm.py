"""Diagnósticos de unidades e valorações da versão ABM.

Este módulo não participa das equações do modelo. Ele apenas torna explícitas
as óticas real, PB, PM e de transação usadas nos resultados dos mercados.
"""

from __future__ import annotations

import pandas as pd


DICIONARIO_UNIDADES_ABM = pd.DataFrame(
    [
        ("producao_real", "real", "PB", "quantidade produzida", "ex ante", "firma"),
        ("vendas_real", "real", "PB", "quantidade vendida", "mercado", "mercado"),
        ("demanda_real", "real", "variável", "quantidade demandada", "mercado", "mercado"),
        ("estoque_real", "real", "produção doméstica", "estoque físico", "fim do período", "firma"),
        ("variacao_estoques_real", "real", "produção doméstica", "fluxo físico", "fim do período", "firma"),
        ("consumo_nominal", "nominal", "PM/Pc", "valor de consumo", "pré-mercado", "CEI"),
        ("demanda_total_pm_nominal", "nominal", "PM/Pc", "valor dos usos", "pré-mercado", "demanda"),
        ("demanda_total_pb_nominal", "nominal", "PB", "orçamento do mercado", "pré-mercado", "conversão PM→PB"),
        ("pb", "índice", "PB", "preço básico doméstico", "realizado", "firmas/mercado"),
        ("pm", "índice", "PM", "preço de importação", "realizado", "importado"),
        ("pc", "índice", "Pc", "preço de comprador", "realizado", "TRU"),
        ("preco_firma", "índice", "PB", "preço de oferta doméstico", "ex ante", "firma"),
        ("preco_transacao", "índice", "mercado", "preço efetivamente transacionado", "mercado", "mercado"),
        ("quantidade_importada_real", "real", "PM", "quantidade importada", "mercado", "importado"),
        ("valor_importacao_pm_nominal", "nominal", "PM", "M real × Pm", "mercado", "diagnóstico"),
        ("valor_transacao_mercado_nominal", "nominal", "mercado", "M real × preço de transação", "mercado", "diagnóstico"),
        ("impostos_produtos", "nominal", "Pc", "imposto sobre produtos", "CEI", "legado nesta etapa"),
        ("valor_adicionado", "nominal", "PB", "VA da produção doméstica", "pré-mercado", "firmas"),
    ],
    columns=["nome", "real_nominal", "otica_preco", "unidade_economica", "momento", "fonte"],
).set_index("nome")


MAPA_BLOCOS_ABM = pd.DataFrame(
    [
        ("produção ex ante", "simulacao_cei_2.simul_", "Firma.decidir_producao"),
        ("preços ex ante", "ciclo_abm.calcular_precos_ex_ante", "Pc esperado, Pb, Pm e Pc"),
        ("agregação produtiva", "agregar_firmas.agregar_firmas", "produção, CI, VA e ocupações"),
        ("consumo", "simulacao_cei_2.simul_CEI", "renda disponível e consumo nominal"),
        ("demanda PM/PB", "ciclo_abm.montar_demandas_periodo", "C+G+I+X+CI e PM→PB"),
        ("mercado industrial", "mercados_abm.executar_mercados_industriais", "multilogit, vendas e racionamento"),
        ("mercado homogêneo", "mercados_abm.executar_mercados_homogeneos_legado", "despacho e preço uniforme"),
        ("preços realizados", "ciclo_abm.calcular_precos_realizados", "Pb, Pm e Pc após mercado"),
        ("valorações importadas", "auditoria_abm.calcular_valoracoes_importacao", "PM versus preço de transação"),
        ("estoques", "simulacao_cei_2.simul_", "bloco físico e valoração legados"),
        ("impostos", "simulacao_cei_2.simul_", "fluxos_transitorios legados"),
        ("CEI e B.9", "simulacao_cei_2.simul_CEI", "fluxos institucionais e capacidade"),
        ("financeiro", "ciclo_abm.atualizar_financeiro_periodo", "B.9→ativos/passivos"),
        ("histórico", "ciclo_abm.montar_registro_historico", "linha pública do período"),
        ("estado t+1", "ciclo_abm.atualizar_estado_periodo", "estados herdados"),
    ],
    columns=["bloco", "local", "saida"],
).set_index("bloco")


def calcular_valoracoes_importacao(
    quantidade_importada_real: float,
    preco_importado_pm: float,
    preco_transacao_mercado: float,
) -> dict[str, float]:
    """Expõe as duas valorações sem escolher uma ponte econômica entre elas."""

    valor_importacao_pm_nominal = quantidade_importada_real * preco_importado_pm
    valor_transacao_mercado_nominal = (
        quantidade_importada_real * preco_transacao_mercado
    )
    return {
        "quantidade_importada_real": quantidade_importada_real,
        "preco_importado_pm": preco_importado_pm,
        "preco_transacao_mercado": preco_transacao_mercado,
        "valor_importacao_pm_nominal": valor_importacao_pm_nominal,
        "valor_transacao_mercado_nominal": valor_transacao_mercado_nominal,
        "diferenca_valoracao": (
            valor_transacao_mercado_nominal - valor_importacao_pm_nominal
        ),
    }


def diagnosticar_importacoes(
    importados: dict,
    precos_transacao: pd.Series,
) -> pd.DataFrame:
    """Retorna as óticas PM e de mercado, por setor, sem afetar a simulação."""

    linhas = {
        setor: calcular_valoracoes_importacao(
            float(importado.quantidade_importada_real),
            float(importado.preco_importado),
            float(precos_transacao.at[setor]),
        )
        for setor, importado in importados.items()
    }
    return pd.DataFrame.from_dict(linhas, orient="index").rename_axis("setor")
