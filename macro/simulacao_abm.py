"""Ciclo temporal das firmas sobre a estrutura agregada SFC-IO."""

import numpy as np
import pandas as pd


def simul_abm(
    periodos: int,
    condicoes_iniciais: dict,
    resultado_sfc_io: dict,
    resultado_abm_inicial: dict,
    config_abm: dict,
) -> dict:
    """Simula expectativas, preços, vendas, produção e estoques das firmas."""

    ci = condicoes_iniciais
    setores = ci["setores"]
    firmas_base = resultado_abm_inicial["firmas"].copy()
    importadores_base = resultado_abm_inicial["fornecedores_importados"].copy()
    parametros_setoriais = resultado_abm_inicial["parametros_setoriais"]
    tecnologias = resultado_abm_inicial["tecnologias_firmas"]

    velocidade_expectativas = float(config_abm["velocidade_expectativas"])
    velocidade_estoques = float(
        config_abm["velocidade_ajuste_estoques_firmas"]
    )
    
    tolerancia_precos = float(config_abm["tolerancia_precos_abm"])
    max_iteracoes_precos = int(config_abm["max_iteracoes_precos_abm"])
    amortecimento_precos = float(config_abm["amortecimento_precos_abm"])

    if not 0.0 <= velocidade_expectativas <= 1.0:
        raise ValueError("velocidade_expectativas deve estar entre 0 e 1.")
    if not 0.0 <= velocidade_estoques <= 1.0:
        raise ValueError(
            "velocidade_ajuste_estoques_firmas deve estar entre 0 e 1."
        )
    if not 0.0 < amortecimento_precos <= 1.0:
        raise ValueError("amortecimento_precos_abm deve estar em (0, 1].")

    # ------------------------------------------------------------------
    # Estados carregados de um período ao outro
    # ------------------------------------------------------------------
    expectativa_vendas = firmas_base["producao_vendida_real"].copy()
    estoque_ciclico = pd.Series(0.0, index=firmas_base.index)
    preco_comprador_anterior = pd.Series(1.0, index=setores)

    firmas_periodos = {}
    setores_periodos = {}
    demanda_intermediaria_periodos = {}
    precos_comprador_periodos = {
        0: pd.Series(1.0, index=setores, name="preco_comprador")
    }
    precos_basicos_periodos = {
        0: pd.Series(1.0, index=setores, name="preco_basico")
    }
    importacoes_periodos = {
        0: (
            ci["parcela_importada"]
            * (ci["conversao_de_pm_pb"] @ ci["demanda_final_base"])
        ).rename("vendas_importadas_real")
    }

    firmas_zero = firmas_base.copy()
    firmas_zero["periodo"] = 0
    firmas_zero["preco_firma"] = firmas_zero["preco_relativo"]
    firmas_zero["preco_transacao"] = firmas_zero["preco_relativo"]
    firmas_zero["expectativa_vendas_real"] = expectativa_vendas
    firmas_zero["producao_planejada_real"] = firmas_zero["producao_real"]
    firmas_zero["demanda_recebida_real"] = firmas_zero[
        "producao_vendida_real"
    ]
    firmas_zero["vendas_real"] = firmas_zero["producao_vendida_real"]
    firmas_zero["demanda_nao_atendida_real"] = 0.0
    firmas_zero["estoque_ciclico_real"] = 0.0
    firmas_zero["estoque_total_real"] = firmas_zero["estoque_inicial_real"]
    firmas_zero["massa_salarial_nominal"] = (
        firmas_zero["salario_unitario_base"]
        * firmas_zero["producao_real"]
    )
    firmas_zero["lucro_normal_nominal"] = (
        firmas_zero["lucro_normal_unitario_base"]
        * firmas_zero["producao_real"]
    )
    firmas_zero["lucro_contabil_nominal"] = firmas_zero[
        "lucro_normal_nominal"
    ]
    firmas_zero["emprego"] = (
        firmas_zero["ocupacoes_unitario_base"]
        * firmas_zero["producao_real"]
    )
    firmas_periodos[0] = firmas_zero

    historico = [{
        "periodo": 0,
        "producao_planejada_real": float(
            firmas_zero["producao_planejada_real"].sum()
        ),
        "producao_real": float(firmas_zero["producao_real"].sum()),
        "vendas_real": float(firmas_zero["vendas_real"].sum()),
        "demanda_nao_atendida_real": 0.0,
        "estoque_ciclico_real": 0.0,
        "emprego": float(firmas_zero["emprego"].sum()),
        "iteracoes_precos": 0,
        "residuo_precos": 0.0,
    }]

    # ==================================================================
    # CICLO TEMPORAL DO ABM
    # ==================================================================

    for t in range(1, periodos + 1):

        indice_salarios = float(
            resultado_sfc_io["historico"].at[t, "indice_salarios"]
        )
        preco_importacoes = resultado_sfc_io["precos_importacoes"].loc[t]

        # A demanda final contém a variação de estoques. Ela será convertida a
        # preços básicos junto com as margens; somente depois retiraremos das
        # vendas a parcela do próprio bem doméstico que foi estocada. Serviços
        # de comércio/transporte ligados ao estoque continuam sendo vendidos.
        demanda_final_nominal = resultado_sfc_io["tru_nominal"][t][
            "demanda_final"
        ]
        estoques_nominais = resultado_sfc_io[
            "variacao_estoques_nominal"
        ][t]
        diagonal_conversao = pd.Series(
            np.diag(ci["conversao_domestica"]),
            index=setores,
        )

        # A produção planejada usa a expectativa de vendas e repõe o fluxo
        # autônomo de estoques observado na base. Estoque cíclico positivo faz
        # a firma reduzir gradualmente a produção planejada.
        producao_planejada = (
            expectativa_vendas
            + firmas_base["variacao_estoque_autonoma_real"]
            - velocidade_estoques * estoque_ciclico
        ).clip(lower=0.0)

        demanda_intermediaria = tecnologias.mul(
            producao_planejada,
            axis="columns",
        )
        demanda_intermediaria_total = demanda_intermediaria.sum(axis=1)

        # --------------------------------------------------------------
        # Preços: substituição simples, sem Newton
        # Pc -> custo dos intermediários -> preços/ofertas das firmas
        # -> preço básico setorial -> novo Pc.
        # --------------------------------------------------------------
        pc = preco_comprador_anterior.copy()
        convergiu_preco = False

        for iteracao_preco in range(1, max_iteracoes_precos + 1):
            custo_intermediario_unitario = tecnologias.T @ pc
            salario_unitario = (
                firmas_base["salario_unitario_base"] * indice_salarios
            )
            lucro_normal_unitario = (
                firmas_base["lucro_normal_unitario_base"]
                * indice_salarios
            )
            outros_va_unitario = (
                firmas_base["outros_va_unitario_base"] * indice_salarios
            )
            custo_fatores_unitario = (
                salario_unitario
                + lucro_normal_unitario
                + outros_va_unitario
            )
            custo_total_unitario = (
                custo_intermediario_unitario + custo_fatores_unitario
            )

            preco_firma = custo_total_unitario.copy()
            industriais = firmas_base["regime"] == "industrial"
            preco_firma.loc[industriais] = (
                preco_firma.loc[industriais]
                * firmas_base.loc[industriais, "preco_relativo"]
            )

            demanda_final_real = demanda_final_nominal / pc
            variacao_estoques_real = estoques_nominais / pc
            demanda_total_comprador = (
                demanda_final_real + demanda_intermediaria_total
            )
            demanda_basica_total = (
                ci["conversao_de_pm_pb"] @ demanda_total_comprador
            ).clip(lower=0.0)
            demanda_domestica_incluindo_estoques = (
                ci["Sd"] @ demanda_basica_total
            ).clip(lower=0.0)
            oferta_importada = (
                ci["Sm"] @ demanda_basica_total
            ).clip(lower=0.0)
            variacao_estoque_domestica = (
                diagonal_conversao * variacao_estoques_real
            )
            demanda_domestica = (
                demanda_domestica_incluindo_estoques
                - variacao_estoque_domestica
            ).clip(lower=0.0)

            demanda_firmas = pd.Series(0.0, index=firmas_base.index)
            preco_basico = pd.Series(0.0, index=setores)
            preco_basico_composto = pd.Series(0.0, index=setores)
            vendas_importadas = pd.Series(0.0, index=setores)
            demanda_mercado = demanda_domestica + oferta_importada

            for setor in setores:
                ids = firmas_base.index[firmas_base["setor"] == setor]
                demanda_setor = float(demanda_domestica[setor])
                regime = parametros_setoriais.at[setor, "regime"]

                if regime == "industrial":
                    eta_preco = float(
                        parametros_setoriais.at[setor, "eta_preco"]
                    )
                    eta_qualidade = float(
                        parametros_setoriais.at[setor, "eta_qualidade"]
                    )
                    atratividade = (
                        firmas_base.loc[ids, "qualidade"] ** eta_qualidade
                        * preco_firma.loc[ids] ** eta_preco
                    )
                    atratividade_importada = (
                        importadores_base.at[setor, "qualidade"]
                        ** eta_qualidade
                        * preco_importacoes[setor] ** eta_preco
                    )
                    atratividade_total = (
                        atratividade.sum() + atratividade_importada
                    )
                    parcelas = atratividade / atratividade_total
                    parcela_importada = (
                        atratividade_importada / atratividade_total
                    )
                    demanda_setor = float(demanda_mercado[setor])
                    demanda_firmas.loc[ids] = demanda_setor * parcelas
                    vendas_importadas[setor] = (
                        demanda_setor * parcela_importada
                    )
                    preco_basico[setor] = float(
                        (parcelas * preco_firma.loc[ids]).sum()
                        / max(1e-30, float(parcelas.sum()))
                    )
                    preco_basico_composto[setor] = float(
                        (parcelas * preco_firma.loc[ids]).sum()
                        + parcela_importada * preco_importacoes[setor]
                    )
                    continue

                # Leilão de preço uniforme. A firma marginal determina o preço;
                # firmas empatadas no preço marginal dividem a demanda na
                # proporção das quantidades ofertadas.
                forma_estoque = bool(
                    parametros_setoriais.at[setor, "forma_estoque"]
                )
                oferta = producao_planejada.loc[ids].copy()
                if forma_estoque:
                    # O fluxo autônomo negativo representa a liberação normal
                    # de estoques já calibrada na base. Estoques cíclicos
                    # positivos podem ser adicionalmente colocados no leilão.
                    oferta = (
                        oferta
                        - firmas_base.loc[
                            ids,
                            "variacao_estoque_autonoma_real",
                        ]
                        + estoque_ciclico.loc[ids].clip(lower=0.0)
                    ).clip(lower=0.0)

                demanda_setor = float(
                    demanda_domestica[setor] + oferta_importada[setor]
                )
                demanda_mercado[setor] = demanda_setor
                id_externo = f"EXTERNO::{setor}"

                ordem = pd.DataFrame({
                    "preco": [*preco_firma.loc[ids], preco_importacoes[setor]],
                    "oferta": [*oferta, oferta_importada[setor]],
                }, index=[*ids, id_externo]).sort_values("preco")
                restante = demanda_setor
                preco_marginal = float(ordem["preco"].max())
                quantidades_aceitas = pd.Series(0.0, index=ordem.index)

                ofertas_pendentes = ordem.copy()
                while not ofertas_pendentes.empty:
                    preco_oferta = float(ofertas_pendentes["preco"].iloc[0])
                    grupo = ofertas_pendentes.index[
                        np.isclose(
                            ofertas_pendentes["preco"],
                            preco_oferta,
                            rtol=0.0,
                            atol=1e-10,
                        )
                    ]
                    oferta_grupo = float(
                        ofertas_pendentes.loc[grupo, "oferta"].sum()
                    )
                    if restante >= oferta_grupo - 1e-12:
                        quantidades_aceitas.loc[grupo] = ofertas_pendentes.loc[
                            grupo,
                            "oferta",
                        ]
                        restante -= oferta_grupo
                        preco_marginal = float(preco_oferta)
                        ofertas_pendentes = ofertas_pendentes.drop(index=grupo)
                    else:
                        if oferta_grupo > 0.0:
                            quantidades_aceitas.loc[grupo] = (
                                ofertas_pendentes.loc[grupo, "oferta"]
                                * max(0.0, restante)
                                / oferta_grupo
                            )
                        preco_marginal = float(preco_oferta)
                        restante = 0.0
                        break
                demanda_firmas.loc[ids] = quantidades_aceitas.loc[ids]
                vendas_importadas[setor] = quantidades_aceitas[id_externo]
                preco_basico[setor] = preco_marginal
                preco_basico_composto[setor] = preco_marginal

            pc_calculado = (
                ci["G"]
                @ preco_basico_composto
            ).rename("preco_comprador")
            residuo_preco = float((pc_calculado - pc).abs().max())
            if residuo_preco <= tolerancia_precos:
                convergiu_preco = True
                break
            pc = (
                amortecimento_precos * pc_calculado
                + (1.0 - amortecimento_precos) * pc
            )

        if not convergiu_preco:
            raise RuntimeError(
                f"Preços do ABM não convergiram no período {t}. "
                f"Resíduo final = {residuo_preco}."
            )

        # --------------------------------------------------------------
        # Produção, vendas e estoques após a formação da demanda
        # --------------------------------------------------------------
        estoque_disponivel = (
            firmas_base["estoque_inicial_real"] + estoque_ciclico
        ).clip(lower=0.0)
        forma_estoque = firmas_base["forma_estoque"]
        oferta_disponivel = producao_planejada.copy()
        oferta_disponivel.loc[forma_estoque] += estoque_disponivel.loc[
            forma_estoque
        ]
        vendas = pd.concat(
            [demanda_firmas, oferta_disponivel],
            axis="columns",
        ).min(axis="columns")
        producao_real = producao_planejada.copy()
        producao_real.loc[~forma_estoque] = pd.concat(
            [
                producao_planejada.loc[~forma_estoque],
                demanda_firmas.loc[~forma_estoque],
            ],
            axis="columns",
        ).min(axis="columns")
        demanda_nao_atendida = (demanda_firmas - vendas).clip(lower=0.0)

        estoque_ciclico_novo = estoque_ciclico.copy()
        estoque_ciclico_novo.loc[forma_estoque] = (
            estoque_ciclico.loc[forma_estoque]
            + producao_real.loc[forma_estoque]
            - vendas.loc[forma_estoque]
            - firmas_base.loc[
                forma_estoque,
                "variacao_estoque_autonoma_real",
            ]
        )
        estoque_ciclico_novo.loc[~forma_estoque] = 0.0
        estoque_ciclico_novo = estoque_ciclico_novo.clip(
            lower=-firmas_base["estoque_inicial_real"]
        )
        estoque_total = (
            firmas_base["estoque_inicial_real"] + estoque_ciclico_novo
        ).clip(lower=0.0)

        preco_transacao = preco_firma.copy()
        for setor in parametros_setoriais.index[
            parametros_setoriais["regime"] == "leilao"
        ]:
            ids = firmas_base.index[firmas_base["setor"] == setor]
            preco_transacao.loc[ids] = preco_basico[setor]

        massa_salarial = salario_unitario * producao_real
        lucro_normal = lucro_normal_unitario * producao_real
        outros_va = outros_va_unitario * producao_real
        custo_intermediario = custo_intermediario_unitario * producao_real
        valor_producao = preco_transacao * producao_real
        lucro_contabil = (
            valor_producao
            - custo_intermediario
            - massa_salarial
            - outros_va
        )
        emprego = firmas_base["ocupacoes_unitario_base"] * producao_real

        periodo_firmas = firmas_base.copy()
        periodo_firmas["periodo"] = t
        periodo_firmas["expectativa_vendas_real"] = expectativa_vendas
        periodo_firmas["producao_planejada_real"] = producao_planejada
        periodo_firmas["producao_real"] = producao_real
        periodo_firmas["demanda_recebida_real"] = demanda_firmas
        periodo_firmas["vendas_real"] = vendas
        periodo_firmas["demanda_nao_atendida_real"] = demanda_nao_atendida
        periodo_firmas["preco_firma"] = preco_firma
        periodo_firmas["preco_transacao"] = preco_transacao
        periodo_firmas["custo_intermediario_unitario"] = (
            custo_intermediario_unitario
        )
        periodo_firmas["salario_unitario"] = salario_unitario
        periodo_firmas["lucro_normal_unitario"] = lucro_normal_unitario
        periodo_firmas["outros_va_unitario"] = outros_va_unitario
        periodo_firmas["massa_salarial_nominal"] = massa_salarial
        periodo_firmas["lucro_normal_nominal"] = lucro_normal
        periodo_firmas["lucro_contabil_nominal"] = lucro_contabil
        periodo_firmas["emprego"] = emprego
        periodo_firmas["estoque_ciclico_real"] = estoque_ciclico_novo
        periodo_firmas["estoque_total_real"] = estoque_total
        firmas_periodos[t] = periodo_firmas

        periodo_setores = pd.DataFrame(index=setores)
        periodo_setores.index.name = "setor"
        periodo_setores["demanda_domestica_real"] = demanda_domestica
        periodo_setores["producao_planejada_real"] = (
            producao_planejada.groupby(firmas_base["setor"]).sum()
        ).reindex(setores)
        periodo_setores["producao_real"] = (
            producao_real.groupby(firmas_base["setor"]).sum()
        ).reindex(setores)
        periodo_setores["vendas_real"] = (
            vendas.groupby(firmas_base["setor"]).sum()
        ).reindex(setores)
        vendas_setoriais = vendas.groupby(firmas_base["setor"]).sum().reindex(
            setores
        )
        periodo_setores["demanda_nao_atendida_real"] = (
            demanda_mercado - vendas_setoriais - vendas_importadas
        ).clip(lower=0.0)
        periodo_setores["vendas_importadas_real"] = vendas_importadas
        periodo_setores["demanda_mercado_real"] = demanda_mercado
        periodo_setores["market_share_importado_real"] = (
            vendas_importadas
            .div((vendas_setoriais + vendas_importadas).replace(0.0, np.nan))
            .fillna(0.0)
        )
        periodo_setores["estoque_total_real"] = (
            estoque_total.groupby(firmas_base["setor"]).sum()
        ).reindex(setores)
        periodo_setores["emprego"] = (
            emprego.groupby(firmas_base["setor"]).sum()
        ).reindex(setores)
        periodo_setores["preco_basico"] = preco_basico
        setores_periodos[t] = periodo_setores

        historico.append({
            "periodo": t,
            "producao_planejada_real": float(producao_planejada.sum()),
            "producao_real": float(producao_real.sum()),
            "vendas_real": float(vendas.sum()),
            "demanda_nao_atendida_real": float(
                periodo_setores["demanda_nao_atendida_real"].sum()
            ),
            "estoque_ciclico_real": float(estoque_ciclico_novo.sum()),
            "emprego": float(emprego.sum()),
            "iteracoes_precos": iteracao_preco,
            "residuo_precos": residuo_preco,
        })
        demanda_intermediaria_periodos[t] = demanda_intermediaria
        precos_comprador_periodos[t] = pc.copy()
        precos_basicos_periodos[t] = preco_basico.copy()
        importacoes_periodos[t] = vendas_importadas.copy()

        expectativa_vendas = (
            (1.0 - velocidade_expectativas) * expectativa_vendas
            + velocidade_expectativas * vendas
        )
        estoque_ciclico = estoque_ciclico_novo
        preco_comprador_anterior = pc.copy()

    firmas_painel = pd.concat(
        firmas_periodos,
        names=["periodo", "firma"],
    )
    setores_painel = pd.concat(
        setores_periodos,
        names=["periodo", "setor"],
    )
    return {
        "historico": pd.DataFrame(historico).set_index("periodo"),
        "firmas": firmas_periodos,
        "firmas_painel": firmas_painel,
        "setores": setores_periodos,
        "setores_painel": setores_painel,
        "demanda_intermediaria_planejada": demanda_intermediaria_periodos,
        "precos_comprador": pd.DataFrame(precos_comprador_periodos).T,
        "precos_basicos": pd.DataFrame(precos_basicos_periodos).T,
        "vendas_importadas": pd.DataFrame(importacoes_periodos).T,
    }
