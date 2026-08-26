"""Mercados da versão ABM, isolados da orquestração temporal.

As funções deste módulo preservam o comportamento legado. Em particular, o
leilão homogêneo continua valorando a venda importada pelo preço uniforme. A
ótica PM é exposta apenas em um diagnóstico separado e não fecha conta alguma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from diagnosticos.auditoria_abm import calcular_valoracoes_importacao
from mercados.mercado_leilao import executar_leilao_uniforme


def executar_mercados_periodo(
    setores: list[str],
    firmas: dict,
    importados: dict,
    demanda_total_pm_nominal: pd.Series,
    demanda_total_pb_nominal: pd.Series,
    demanda_intermediaria_real: pd.Series,
    pb: pd.Series,
) -> dict:
    """Executa os dois mercados legados e reúne seus registros públicos."""

    industrial = executar_mercados_industriais(
        setores, firmas, importados, demanda_total_pb_nominal
    )
    homogeneo = executar_mercados_homogeneos_legado(
        setores, firmas, importados, demanda_total_pb_nominal, pb
    )
    diagnostico = pd.concat(
        [industrial["diagnostico_importacoes"], homogeneo["diagnostico_importacoes"]]
    ).reindex(setores)
    return {
        "pb": homogeneo["pb"],
        "setores_leilao": homogeneo["setores"],
        "participantes_industriais": industrial["participantes"],
        "participantes_leilao": homogeneo["participantes"],
        "diagnostico_importacoes": diagnostico,
        "registro_industrial": {
            "demanda_total_pm_nominal": demanda_total_pm_nominal.copy(),
            "demanda_total_pb_nominal": demanda_total_pb_nominal.copy(),
            "demanda_intermediaria_real": demanda_intermediaria_real.copy(),
            "participantes": industrial["participantes"].copy(),
        },
        "registro_leilao": {
            "demanda_total_pm_nominal": demanda_total_pm_nominal.copy(),
            "demanda_total_pb_nominal": demanda_total_pb_nominal.copy(),
            "demanda_intermediaria_real": demanda_intermediaria_real.copy(),
            "setores": homogeneo["setores"],
            "participantes": homogeneo["participantes"].copy(),
        },
    }


def executar_mercados_industriais(
    setores: list[str],
    firmas: dict,
    importados: dict,
    demanda_total_pb_nominal: pd.Series,
) -> dict:
    """Executa o multilogit e a restrição de oferta doméstica já existentes."""

    participantes = []
    diagnosticos_importacao = {}

    for setor in setores:
        firmas_setor = [
            firma
            for firma in firmas.values()
            if firma.setor == setor and firma.regime == "industrial"
        ]
        if not firmas_setor:
            continue

        importado = importados[setor]

        eta_preco = firmas_setor[0].eta_preco
        eta_qualidade = firmas_setor[0].eta_qualidade

        if any(
            firma.eta_preco != eta_preco
            or firma.eta_qualidade != eta_qualidade
            for firma in firmas_setor
        ):
            raise RuntimeError(f"Elasticidades inconsistentes em {setor}.")

        atratividades_domesticas = np.asarray(
            [firma.calcular_atratividade() for firma in firmas_setor],
            dtype=float,
        )

        importado.fator_atendimento = 1.0
        importado.atratividade_importado = float(
            importado.qualidade_importado**eta_qualidade
            * importado.preco_importado**eta_preco
        )

        atratividades = np.append(
            atratividades_domesticas, importado.atratividade_importado
        )

        if (atratividades < 0.0).any() or not np.isfinite(atratividades).all():
            raise RuntimeError(f"Atratividade inválida em {setor}.")

        denominador = float(atratividades.sum())

        if denominador <= 0.0:
            raise RuntimeError(f"Atratividade total não positiva em {setor}.")

        shares_desejados = atratividades / denominador

        if not np.allclose(shares_desejados.sum(), 1.0, atol=1e-12):
            raise RuntimeError(f"Shares desejados não fecham em {setor}.")

        orcamento_setorial = float(demanda_total_pb_nominal.at[setor])

        fatores_atendimento_usados = {}
        
        for firma, share_desejado in zip(
            firmas_setor,
            shares_desejados[:-1],
            strict=True,
        ):
            fatores_atendimento_usados[firma.id] = firma.fator_atendimento
            firma.market_share_desejado = float(share_desejado)
            firma.demanda_nominal_desejada = (
                firma.market_share_desejado * orcamento_setorial
            )
            firma.demanda_recebida_real = (
                firma.demanda_nominal_desejada / firma.preco_firma
            )
            oferta_disponivel_real = firma.producao_real
            firma.vendas_real = min(
                firma.demanda_recebida_real, oferta_disponivel_real
            )
            firma.vendas_nominal = firma.vendas_real * firma.preco_transacao
            firma.demanda_nao_atendida_real = max(
                0.0, firma.demanda_recebida_real - firma.vendas_real
            )
            firma.producao_nao_vendida_real = max(
                0.0, firma.producao_real - firma.vendas_real
            )
            firma.taxa_atendimento = (
                firma.vendas_real / firma.demanda_recebida_real
                if firma.demanda_recebida_real > 0.0
                else 1.0
            )
            firma.fator_atendimento = firma.taxa_atendimento

        importado.market_share_desejado = float(shares_desejados[-1])
        importado.market_share_importado = importado.market_share_desejado
        importado.demanda_nominal_desejada = (
            importado.market_share_desejado * orcamento_setorial
        )
        importado.demanda_real_desejada = (
            importado.demanda_nominal_desejada / importado.preco_importado
        )
        # Hipótese legada: o exterior atende integralmente no mercado industrial.
        importado.vendas_real = importado.demanda_real_desejada
        importado.vendas_nominal = importado.demanda_nominal_desejada

        vendas_nominais = np.asarray(
            [firma.vendas_nominal for firma in firmas_setor]
            + [importado.vendas_nominal],
            dtype=float,
        )
        total_vendas_nominais = float(vendas_nominais.sum())
        shares_realizados = (
            vendas_nominais / total_vendas_nominais
            if total_vendas_nominais > 0.0
            else np.zeros_like(vendas_nominais)
        )
        for firma, share_realizado, atratividade in zip(
            firmas_setor,
            shares_realizados[:-1],
            atratividades_domesticas,
            strict=True,
        ):
            firma.market_share_realizado = float(share_realizado)
            firma.atualizar_market_shares_defasados(
                firma.market_share_realizado
            )
            participantes.append(
                {
                    "participante": firma.id,
                    "setor": setor,
                    "tipo": "domestico",
                    "market_share_desejado": firma.market_share_desejado,
                    "market_share_realizado": firma.market_share_realizado,
                    "demanda_nominal_desejada": firma.demanda_nominal_desejada,
                    "demanda_real_recebida": firma.demanda_recebida_real,
                    "vendas_real": firma.vendas_real,
                    "vendas_nominal": firma.vendas_nominal,
                    "demanda_nao_atendida_real": firma.demanda_nao_atendida_real,
                    "fator_atendimento": firma.fator_atendimento,
                    "fator_atendimento_usado": fatores_atendimento_usados[firma.id],
                    "atratividade": float(atratividade),
                }
            )
        importado.market_share_realizado = float(shares_realizados[-1])
        participantes.append(
            {
                "participante": f"importado::{setor}",
                "setor": setor,
                "tipo": "importado",
                "market_share_desejado": importado.market_share_desejado,
                "market_share_realizado": importado.market_share_realizado,
                "demanda_nominal_desejada": importado.demanda_nominal_desejada,
                "demanda_real_recebida": importado.demanda_real_desejada,
                "vendas_real": importado.vendas_real,
                "vendas_nominal": importado.vendas_nominal,
                "demanda_nao_atendida_real": 0.0,
                "fator_atendimento": importado.fator_atendimento,
                "fator_atendimento_usado": 1.0,
                "atratividade": importado.atratividade_importado,
            }
        )
        diagnosticos_importacao[setor] = calcular_valoracoes_importacao(
            float(importado.vendas_real),
            float(importado.preco_importado),
            float(importado.preco_importado),
        )

    return {
        "participantes": pd.DataFrame(participantes).set_index("participante"),
        "diagnostico_importacoes": pd.DataFrame.from_dict(
            diagnosticos_importacao, orient="index"
        ).rename_axis("setor"),
    }


def executar_mercados_homogeneos_legado(
    setores: list[str],
    firmas: dict,
    importados: dict,
    demanda_total_pb_nominal: pd.Series,
    pb: pd.Series,
) -> dict:
    """Executa literalmente o leilão uniforme legado e expõe suas valorações."""

    participantes = []
    leiloes_setoriais = {}
    diagnosticos_importacao = {}
    for setor in setores:
        firmas_setor = [
            firma
            for firma in firmas.values()
            if firma.setor == setor and firma.regime == "leilao"
        ]
        if not firmas_setor:
            continue
        importado = importados[setor]
        ofertas = pd.DataFrame(
            [
                {
                    "participante": firma.id,
                    "tipo": "domestico",
                    "oferta_disponivel_real": firma.producao_real,
                    "preco_oferta": firma.preco_oferta_leilao,
                }
                for firma in firmas_setor
            ]
            + [
                {
                    "participante": f"importado::{setor}",
                    "tipo": "importado",
                    "oferta_disponivel_real": importado.oferta_maxima_importado_real,
                    "preco_oferta": importado.preco_oferta_importado,
                }
            ]
        )
        leilao = executar_leilao_uniforme(
            ofertas, float(demanda_total_pb_nominal.at[setor])
        )
        ofertas_despachadas = leilao["ofertas"].set_index("participante")
        preco_uniforme = leilao["preco_transacao_setorial"]
        total_vendas_nominais = 0.0
        for firma in firmas_setor:
            vendas_real = float(
                ofertas_despachadas.at[firma.id, "quantidade_despachada_real"]
            )
            firma.vendas_real = vendas_real
            firma.vendas_nominal = vendas_real * preco_uniforme
            firma.demanda_recebida_real = vendas_real
            firma.demanda_nominal_desejada = firma.vendas_nominal
            firma.demanda_nao_atendida_real = 0.0
            firma.producao_nao_vendida_real = max(
                0.0, firma.producao_real - vendas_real
            )
            firma.taxa_atendimento = 1.0
            firma.fator_atendimento = 1.0
            if vendas_real > 0.0:
                firma.preco_transacao = preco_uniforme
            else:
                firma.preco_transacao = firma.preco_oferta_leilao
            total_vendas_nominais += firma.vendas_nominal

        id_importado = f"importado::{setor}"
        importado.quantidade_importada_real = float(
            ofertas_despachadas.at[id_importado, "quantidade_despachada_real"]
        )
        importado.vendas_real = importado.quantidade_importada_real
        # Comportamento legado preservado: a venda usa o preço uniforme.
        importado.valor_importado_nominal = (
            importado.quantidade_importada_real * preco_uniforme
        )
        importado.vendas_nominal = importado.valor_importado_nominal
        importado.demanda_real_desejada = leilao["quantidade_demandada_real"]
        importado.demanda_nominal_desejada = float(
            demanda_total_pb_nominal.at[setor]
        )
        importado.demanda_nao_atendida_real = leilao[
            "demanda_nao_atendida_real"
        ]
        total_vendas_nominais += importado.vendas_nominal

        for firma in firmas_setor:
            firma.market_share_desejado = (
                firma.vendas_nominal / total_vendas_nominais
                if total_vendas_nominais > 0.0
                else 0.0
            )
            firma.market_share_realizado = firma.market_share_desejado
            participantes.append(
                {
                    "participante": firma.id,
                    "setor": setor,
                    "tipo": "domestico",
                    "preco_oferta": firma.preco_oferta_leilao,
                    "preco_transacao": firma.preco_transacao,
                    "oferta_disponivel_real": firma.producao_real,
                    "vendas_real": firma.vendas_real,
                    "vendas_nominal": firma.vendas_nominal,
                    "producao_nao_vendida_real": firma.producao_nao_vendida_real,
                }
            )
        importado.market_share_realizado = (
            importado.vendas_nominal / total_vendas_nominais
            if total_vendas_nominais > 0.0
            else 0.0
        )
        importado.market_share_importado = importado.market_share_realizado
        participantes.append(
            {
                "participante": id_importado,
                "setor": setor,
                "tipo": "importado",
                "preco_oferta": importado.preco_oferta_importado,
                "preco_transacao": preco_uniforme,
                "oferta_disponivel_real": importado.oferta_maxima_importado_real,
                "vendas_real": importado.vendas_real,
                "vendas_nominal": importado.vendas_nominal,
                "producao_nao_vendida_real": 0.0,
            }
        )
        leiloes_setoriais[setor] = {**leilao, "ofertas": ofertas_despachadas}
        pb.at[setor] = preco_uniforme
        diagnosticos_importacao[setor] = calcular_valoracoes_importacao(
            float(importado.quantidade_importada_real),
            float(importado.preco_importado),
            float(preco_uniforme),
        )

    return {
        "pb": pb,
        "setores": leiloes_setoriais,
        "participantes": pd.DataFrame(participantes).set_index("participante"),
        "diagnostico_importacoes": pd.DataFrame.from_dict(
            diagnosticos_importacao, orient="index"
        ).rename_axis("setor"),
    }
