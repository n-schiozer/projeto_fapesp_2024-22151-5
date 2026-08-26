"""Coordenação de produção independente do regime de mercado."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def calcular_decisoes_regulador(
    firmas: dict,
    decisoes_producao: dict[str, float],
    setores_regulados: Iterable[str],
    *,
    fatores_clima: dict[str, float] | None = None,
    usar_despacho_por_atratividade: bool = False,
) -> dict[str, float]:
    """Coordena a produção dos setores regulados antes da realização do mercado.

    A função devolve somente decisões para firmas de setores regulados. Nos
    demais setores, o chamador deve aplicar a decisão descentralizada original.

    Sem despacho por atratividade, preserva-se a regra legada de redistribuir
    somente os déficits físicos das decisões individuais. No despacho por
    atratividade, a soma das decisões descentralizadas é o alvo setorial ex
    ante, distribuído por qualidade e preço relativo. Um choque climático
    limita a firma exposta relativamente à sua parcela normal; as firmas sem
    choque podem usar a capacidade física ociosa para absorver o déficit.
    """
    setores_regulados = set(setores_regulados)
    decisoes_regulador: dict[str, float] = {}

    for setor in setores_regulados:
        firmas_setor = [
            firma for firma in firmas.values() if firma.setor == setor
        ]
        if not firmas_setor:
            continue

        capacidades = np.asarray(
            [float(firma.capacidade_produtiva_real) for firma in firmas_setor],
            dtype=float,
        )
        if not np.isfinite(capacidades).all() or (capacidades < 0.0).any():
            raise ValueError(
                "Setores regulados exigem capacidade_produtiva_real finita "
                f"e não negativa; setor inválido: {setor}."
            )

        desejadas = np.asarray(
            [max(0.0, float(decisoes_producao[firma.id])) for firma in firmas_setor],
            dtype=float,
        )

        if usar_despacho_por_atratividade:
            precos_referencia = []
            for firma in firmas_setor:
                preco = float(firma.preco_oferta_leilao)
                if not np.isfinite(preco) or preco <= 0.0:
                    preco = float(firma.preco_transacao)
                precos_referencia.append(preco)
            precos = np.asarray(precos_referencia, dtype=float)
            eta_precos = np.asarray(
                [float(firma.eta_preco) for firma in firmas_setor],
                dtype=float,
            )
            eta_qualidades = np.asarray(
                [float(firma.eta_qualidade) for firma in firmas_setor],
                dtype=float,
            )
            qualidades_referencia = []
            for firma, eta_preco, eta_qualidade in zip(
                firmas_setor, eta_precos, eta_qualidades, strict=True
            ):
                qualidade = float(firma.qualidade)
                if not np.isfinite(qualidade) or qualidade <= 0.0:
                    share_base = max(
                        float(firma.share_domestico_inicial),
                        1e-12,
                    )
                    preco_base = float(firma.preco_relativo)
                    qualidade = (
                        share_base / preco_base**eta_preco
                    ) ** (1.0 / eta_qualidade)
                qualidades_referencia.append(qualidade)
            qualidades = np.asarray(qualidades_referencia, dtype=float)
            if (
                not np.isfinite(precos).all()
                or (precos <= 0.0).any()
                or not np.isfinite(qualidades).all()
                or (qualidades <= 0.0).any()
                or not np.isfinite(eta_precos).all()
                or not np.isfinite(eta_qualidades).all()
            ):
                raise ValueError(
                    "Despacho regulado exige preço e qualidade positivos e "
                    "elasticidades finitas."
                )

            atratividades = (
                qualidades**eta_qualidades * precos**eta_precos
            )
            atratividade_total = float(atratividades.sum())
            if not np.isfinite(atratividade_total) or atratividade_total <= 0.0:
                raise ValueError(
                    f"Atratividade total inválida no setor regulado {setor}."
                )

            alvo_setorial = float(desejadas.sum())
            shares_normais = atratividades / atratividade_total
            producao_normal = alvo_setorial * shares_normais

            fatores = np.asarray(
                [
                    float(
                        (fatores_clima or {}).get(
                            firma.id,
                            getattr(firma, "fator_produtividade_climatica", 1.0),
                        )
                    )
                    for firma in firmas_setor
                ],
                dtype=float,
            )
            if (
                not np.isfinite(fatores).all()
                or (fatores < 0.0).any()
                or (fatores > 1.0).any()
            ):
                raise ValueError(
                    "fatores_clima devem ser finitos e estar no intervalo [0, 1]."
                )

            limites_operacionais = capacidades.copy()
            firmas_sob_choque = fatores < 1.0 - 1e-12
            limites_operacionais[firmas_sob_choque] = np.minimum(
                limites_operacionais[firmas_sob_choque],
                fatores[firmas_sob_choque]
                * producao_normal[firmas_sob_choque],
            )
            producao_final = np.minimum(producao_normal, limites_operacionais)
            deficit = max(0.0, alvo_setorial - float(producao_final.sum()))

            while deficit > 1e-12:
                folgas = np.maximum(0.0, limites_operacionais - producao_final)
                elegiveis = folgas > 1e-12
                if not elegiveis.any():
                    break
                pesos = np.where(elegiveis, shares_normais, 0.0)
                incremento = deficit * pesos / float(pesos.sum())
                incremento = np.minimum(incremento, folgas)
                realizado = float(incremento.sum())
                if realizado <= 1e-12:
                    break
                producao_final += incremento
                deficit -= realizado

            for firma, quantidade in zip(
                firmas_setor, producao_final, strict=True
            ):
                decisoes_regulador[firma.id] = float(quantidade)
            continue

        producao_restrita = np.minimum(desejadas, capacidades)
        deficit = max(0.0, float(desejadas.sum() - producao_restrita.sum()))
        folgas = np.maximum(0.0, capacidades - producao_restrita)
        folga_total = float(folgas.sum())
        quantidade_redistribuir = min(deficit, folga_total)

        if quantidade_redistribuir > 0.0:
            producao_final = (
                producao_restrita
                + quantidade_redistribuir * folgas / folga_total
            )
        else:
            producao_final = producao_restrita

        for firma, quantidade in zip(firmas_setor, producao_final, strict=True):
            decisao = float(quantidade)
            if decisao > float(firma.capacidade_produtiva_real) + 1e-10:
                raise RuntimeError(
                    f"Regulador excedeu a capacidade da firma {firma.id}."
                )
            decisoes_regulador[firma.id] = decisao

    return decisoes_regulador
