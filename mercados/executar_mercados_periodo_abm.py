"""Mercados da versão ABM."""

import numpy as np
import pandas as pd


def executar_mercados_periodo(
    setores: list[str],
    firmas: dict,
    importados: dict,
    demanda_real_setorial: pd.Series,
    demanda_nominal_setorial: pd.Series,
    ofertas_reguladas: dict[str, float] | None = None,
) -> dict:
    """Executa os mercados industrial e de leilão do período."""

    # ==========================================================
    # MERCADO INDUSTRIAL
    # ==========================================================

    mercado_industrial = executar_mercados_industriais(
        setores=setores,
        firmas=firmas,
        importados=importados,
        demanda_real_setorial=demanda_real_setorial,
        demanda_nominal_setorial=demanda_nominal_setorial,
    )

    # ==========================================================
    # MERCADO DE LEILÃO
    # ==========================================================

    mercado_leilao = executar_mercados_leilao(
        setores=setores,
        firmas=firmas,
        importados=importados,
        demanda_real_setorial=demanda_real_setorial,
        demanda_nominal_setorial=demanda_nominal_setorial,
        ofertas_reguladas=ofertas_reguladas,
    )

    return {
        "industrial": mercado_industrial,
        "leilao": mercado_leilao,
    }


def executar_mercados_industriais(
    setores: list[str],
    firmas: dict,
    importados: dict,
    demanda_real_setorial: pd.Series,
    demanda_nominal_setorial: pd.Series,
) -> dict:
    """Aloca a demanda entre firmas industriais e importados."""

    resultados_setoriais = {}

    for setor in setores:

        firmas_setor = [
            firma
            for firma in firmas.values()
            if (
                firma.setor == setor
                and firma.regime == "industrial"
            )
        ]

        if not firmas_setor:
            continue

        importado = importados[setor]

        # ==========================================================
        # ATRATIVIDADES
        # ==========================================================

        atratividades_domesticas = np.asarray(
            [
                firma.calcular_atratividade()
                for firma in firmas_setor
            ],
            dtype=float,
        )

        eta_preco = firmas_setor[0].eta_preco
        eta_qualidade = firmas_setor[0].eta_qualidade

        atratividade_importado = (
            importado.qualidade ** eta_qualidade
            * importado.preco ** eta_preco
        )

        atratividades = np.append(
            atratividades_domesticas,
            atratividade_importado,
        )

        if (
            (atratividades < 0.0).any()
            or not np.isfinite(atratividades).all()
        ):
            raise RuntimeError(
                f"Atratividade inválida em {setor}."
            )

        atratividade_total = float(
            atratividades.sum()
        )

        if atratividade_total <= 0.0:
            raise RuntimeError(
                f"Atratividade total não positiva em {setor}."
            )

        # Market shares totais:
        # firmas domésticas + fornecedor importado = 1.
        market_shares = (
            atratividades
            / atratividade_total
        )

        if not np.isclose(
            market_shares.sum(),
            1.0,
        ):
            raise RuntimeError(
                f"Market shares não fecham em {setor}."
            )

        # ==========================================================
        # DEMANDA SETORIAL
        # ==========================================================

        demanda_real = float(
            demanda_real_setorial.at[setor]
        )

        demanda_nominal = float(
            demanda_nominal_setorial.at[setor]
        )

        # ==========================================================
        # FIRMAS DOMÉSTICAS
        # ==========================================================

        for firma, market_share in zip(
            firmas_setor,
            market_shares[:-1],
            strict=True,
        ):

            firma.market_share_desejado = float(
                market_share
            )

            demanda_real_alocada = (
                firma.market_share_desejado
                * demanda_real
            )

            demanda_nominal_alocada = (
                firma.market_share_desejado
                * demanda_nominal
            )

            firma.demanda_recebida_real = (
                demanda_real_alocada
                + demanda_nominal_alocada
                / firma.preco_firma
            )

            firma.vendas_real = min(
                firma.demanda_recebida_real,
                firma.producao_real + firma.estoque,
            )

            firma.preco_transacao = (
                firma.preco_firma
            )

            firma.vendas_nominal = (
                firma.vendas_real
                * firma.preco_transacao
            )

            firma.demanda_nao_atendida_real = max(
                0.0,
                firma.demanda_recebida_real
                - firma.vendas_real,
            )

            firma.producao_nao_vendida_real = max(
                0.0,
                firma.producao_real
                - firma.vendas_real,
            )

            firma.taxa_atendimento = (
                firma.vendas_real
                / firma.demanda_recebida_real
                if firma.demanda_recebida_real > 0.0
                else 1.0
            )

            firma.fator_atendimento = (
                firma.taxa_atendimento
            )

        # ==========================================================
        # FORNECEDOR IMPORTADO
        # ==========================================================

        market_share_importado = float(
            market_shares[-1]
        )

        demanda_real_importado = (
            market_share_importado
            * demanda_real
        )

        demanda_nominal_importado = (
            market_share_importado
            * demanda_nominal
        )

        demanda_importado_real = (
            demanda_real_importado
            + demanda_nominal_importado
            / importado.preco
        )

        importado.vendas_real = min(
            demanda_importado_real,
            importado.oferta_maxima_real,
        )

        importado.vendas_nominal = (
            importado.vendas_real
            * importado.preco
        )

        # ==========================================================
        # MARKET SHARES REALIZADOS
        # ==========================================================

        vendas_nominais = np.asarray(
            [
                firma.vendas_nominal
                for firma in firmas_setor
            ]
            + [
                importado.vendas_nominal
            ],
            dtype=float,
        )

        vendas_nominais_totais = float(
            vendas_nominais.sum()
        )

        if vendas_nominais_totais > 0.0:
            market_shares_realizados = (
                vendas_nominais
                / vendas_nominais_totais
            )
        else:
            market_shares_realizados = (
                np.zeros_like(vendas_nominais)
            )

        for firma, market_share_realizado in zip(
            firmas_setor,
            market_shares_realizados[:-1],
            strict=True,
        ):

            firma.market_share_realizado = float(
                market_share_realizado
            )

            firma.atualizar_market_shares_defasados(
                firma.market_share_realizado
            )

        # ==========================================================
        # RESULTADO SETORIAL
        # ==========================================================

        resultados_setoriais[setor] = {
            "market_share_importado_desejado":
                market_share_importado,

            "market_share_importado_realizado":
                float(market_shares_realizados[-1]),

            "vendas_domesticas_real":
                float(
                    sum(
                        firma.vendas_real
                        for firma in firmas_setor
                    )
                ),

            "vendas_domesticas_nominal":
                float(
                    sum(
                        firma.vendas_nominal
                        for firma in firmas_setor
                    )
                ),

            "importacoes_real":
                importado.vendas_real,

            "importacoes_nominal":
                importado.vendas_nominal,
        }

    return resultados_setoriais


def executar_mercados_leilao_legado_V1(
    setores: list[str],
    firmas: dict,
    importados: dict,
    demanda_real_setorial: pd.Series,
    demanda_nominal_setorial: pd.Series,
) -> dict:
    """Executa os mercados com leilão de preço único.

    A firma deseja ofertar sua demanda esperada, mas sua oferta
    efetiva não pode superar a disponibilidade física:

        O_f = min(D^e_f, Q_f + E_f)

    A oferta importada permanece fixa no nível real calibrado.

    Para cada setor:

        Q = D_real + D_nominal / P

    portanto:

        P = D_nominal / (Q - D_real)

    onde Q é a oferta total efetivamente disponibilizada
    por firmas domésticas e importados.
    """

    resultados_setoriais = []
    participantes = []

    for setor in setores:

        # ==========================================================
        # PARTICIPANTES
        # ==========================================================

        firmas_setor = [
            firma
            for firma in firmas.values()
            if (
                firma.setor == setor
                and firma.regime == "leilao"
            )
        ]

        if not firmas_setor:
            continue

        importado = importados[setor]

        # ==========================================================
        # OFERTA DAS FIRMAS DOMÉSTICAS
        # ==========================================================

        ofertas_firmas = {
            firma.id: min(
                firma.demanda_esperada,
                firma.producao_real + firma.estoque,
            )
            for firma in firmas_setor
        }

        oferta_domestica_real = float(
            sum(ofertas_firmas.values())
        )

        # ==========================================================
        # OFERTA IMPORTADA
        # ==========================================================

        # Nesta versão, a oferta importada permanece constante
        # no nível real calibrado do ano-base.

        oferta_importada_real = float(
            importado.quantidade_ofertada_real
        )

        oferta_total_real = (
            oferta_domestica_real
            + oferta_importada_real
        )

        # ==========================================================
        # DEMANDA
        # ==========================================================

        demanda_real = float(
            demanda_real_setorial.at[setor]
        )

        demanda_nominal = float(
            demanda_nominal_setorial.at[setor]
        )

        if demanda_real < 0.0:
            raise ValueError(
                f"Demanda real negativa no setor {setor}."
            )

        if demanda_nominal <= 0.0:
            raise ValueError(
                f"Demanda nominal deve ser positiva no setor {setor}."
            )

        # ==========================================================
        # PREÇO ÚNICO DE EQUILÍBRIO
        # ==========================================================

        quantidade_disponivel_demanda_nominal = (
            oferta_total_real
            - demanda_real
        )

        if quantidade_disponivel_demanda_nominal <= 0.0:
            raise RuntimeError(
                f"Oferta insuficiente no setor {setor}: "
                f"oferta total={oferta_total_real:.6f}, "
                f"demanda real={demanda_real:.6f}."
            )

        preco_mercado = (
            demanda_nominal
            / quantidade_disponivel_demanda_nominal
        )

        demanda_final_real = (
            demanda_nominal
            / preco_mercado
        )

        demanda_total_real = (
            demanda_real
            + demanda_final_real
        )

        # ==========================================================
        # VENDAS DOMÉSTICAS
        # ==========================================================

        for firma in firmas_setor:

            oferta_firma_real = (
                ofertas_firmas[firma.id]
            )

            # Como o preço foi calculado para limpar o mercado,
            # toda a quantidade efetivamente ofertada é vendida.

            firma.vendas_real = (
                oferta_firma_real
            )

            firma.preco_transacao = (
                preco_mercado
            )

            firma.vendas_nominal = (
                firma.vendas_real
                * preco_mercado
            )

            firma.demanda_recebida_real = (
                firma.vendas_real
            )

            firma.demanda_nao_atendida_real = 0.0

            firma.producao_nao_vendida_real = max(
                0.0,
                firma.producao_real
                - firma.vendas_real,
            )

            firma.taxa_atendimento = 1.0
            firma.fator_atendimento = 1.0

            participantes.append(
                {
                    "participante": firma.id,
                    "setor": setor,
                    "tipo": "domestico",
                    "oferta_real": oferta_firma_real,
                    "vendas_real": firma.vendas_real,
                    "vendas_nominal": firma.vendas_nominal,
                    "preco_transacao": preco_mercado,
                }
            )

        # ==========================================================
        # IMPORTAÇÕES
        # ==========================================================

        importado.vendas_real = (
            oferta_importada_real
        )

        importado.vendas_nominal = (
            importado.vendas_real
            * preco_mercado
        )

        participantes.append(
            {
                "participante": f"importado::{setor}",
                "setor": setor,
                "tipo": "importado",
                "oferta_real": oferta_importada_real,
                "vendas_real": importado.vendas_real,
                "vendas_nominal": importado.vendas_nominal,
                "preco_transacao": preco_mercado,
            }
        )

        # ==========================================================
        # REGISTRO SETORIAL
        # ==========================================================

        resultados_setoriais.append(
            {
                "setor": setor,
                "oferta_domestica_real":
                    oferta_domestica_real,

                "oferta_importada_real":
                    oferta_importada_real,

                "oferta_total_real":
                    oferta_total_real,

                "demanda_real":
                    demanda_real,

                "demanda_nominal":
                    demanda_nominal,

                "demanda_final_real":
                    demanda_final_real,

                "demanda_total_real":
                    demanda_total_real,

                "preco_mercado":
                    preco_mercado,

                "erro_equilibrio": (
                    demanda_total_real
                    - oferta_total_real
                ),
            }
        )

    # ==========================================================
    # RESULTADOS
    # ==========================================================

    if resultados_setoriais:

        resultados_setoriais_df = (
            pd.DataFrame(
                resultados_setoriais
            )
            .set_index("setor")
        )

    else:

        resultados_setoriais_df = pd.DataFrame(
            columns=[
                "oferta_domestica_real",
                "oferta_importada_real",
                "oferta_total_real",
                "demanda_real",
                "demanda_nominal",
                "demanda_final_real",
                "demanda_total_real",
                "preco_mercado",
                "erro_equilibrio",
            ],
            index=pd.Index(
                [],
                name="setor",
            ),
        )

    if participantes:

        participantes_df = (
            pd.DataFrame(
                participantes
            )
            .set_index("participante")
        )

    else:

        participantes_df = pd.DataFrame(
            columns=[
                "setor",
                "tipo",
                "oferta_real",
                "vendas_real",
                "vendas_nominal",
                "preco_transacao",
            ],
            index=pd.Index(
                [],
                name="participante",
            ),
        )

    return {
        "setores": resultados_setoriais_df,
        "participantes": participantes_df,
    }



def executar_mercados_leilao(
    setores: list[str],
    firmas: dict,
    importados: dict,
    demanda_real_setorial: pd.Series,
    demanda_nominal_setorial: pd.Series,
    ofertas_reguladas: dict[str, float] | None = None,
) -> dict:
    """Executa os mercados homogêneos com preço único.

    Cada participante chega ao mercado com:

        - uma quantidade ofertada;
        - um preço de oferta.

    O preço único de mercado é a média dos preços de oferta,
    ponderada pelas quantidades ofertadas:

        P = sum(q_i^of * p_i^of) / sum(q_i^of)

    Depois de determinado o preço, a demanda realizada é:

        D = D_real + D_nominal / P

    A demanda é distribuída entre os participantes segundo
    suas participações na oferta total.
    """

    resultados_setoriais = []
    participantes = []

    for setor in setores:

        # ==========================================================
        # PARTICIPANTES
        # ==========================================================

        firmas_setor = [
            firma
            for firma in firmas.values()
            if (
                firma.setor == setor
                and firma.regime == "leilao"
            )
        ]

        if not firmas_setor:
            continue

        importado = importados[setor]

        # ==========================================================
        # OFERTA DAS FIRMAS DOMÉSTICAS
        # ==========================================================

        # Mantém, por enquanto, exatamente a regra atual de oferta.
        if ofertas_reguladas:
            ofertas_firmas = {
                firma.id: min(
                    max(
                        firma.demanda_esperada,
                        float(ofertas_reguladas.get(firma.id, 0.0)),
                    ),
                    firma.producao_real + firma.estoque,
                )
                for firma in firmas_setor
            }
        else:
            ofertas_firmas = {
                firma.id: min(
                    firma.demanda_esperada,
                    firma.producao_real + firma.estoque,
                )
                for firma in firmas_setor
            }

        oferta_domestica_real = float(
            sum(ofertas_firmas.values())
        )

        # ==========================================================
        # OFERTA IMPORTADA
        # ==========================================================

        oferta_importada_real = float(
            importado.quantidade_ofertada_real
        )

        oferta_total_real = (
            oferta_domestica_real
            + oferta_importada_real
        )

        if oferta_total_real <= 0.0:
            raise RuntimeError(
                f"Oferta total não positiva no setor {setor}."
            )

        # ==========================================================
        # PREÇO ÚNICO DE MERCADO
        # ==========================================================

        # Cada firma já chega ao mercado com seu preço de oferta:
        #
        #     preco_oferta = (1 + markup) * custo_unitario
        #
        # calculado anteriormente em atualizar_custo_e_preco().

        valor_oferta_domestica = float(
            sum(
                ofertas_firmas[firma.id]
                * firma.preco_oferta_leilao
                for firma in firmas_setor
            )
        )

        valor_oferta_importada = (
            oferta_importada_real
            * importado.preco
        )

        preco_mercado = (
            valor_oferta_domestica
            + valor_oferta_importada
        ) / oferta_total_real

        if (
            not np.isfinite(preco_mercado)
            or preco_mercado <= 0.0
        ):
            raise RuntimeError(
                f"Preço de mercado inválido no setor {setor}: "
                f"{preco_mercado}."
            )

        # ==========================================================
        # DEMANDA AO PREÇO DE MERCADO
        # ==========================================================

        demanda_real = float(
            demanda_real_setorial.at[setor]
        )

        demanda_nominal = float(
            demanda_nominal_setorial.at[setor]
        )

        if demanda_real < 0.0:
            raise ValueError(
                f"Demanda real negativa no setor {setor}."
            )

        if demanda_nominal <= 0.0:
            raise ValueError(
                f"Demanda nominal deve ser positiva no setor {setor}."
            )

        demanda_final_real = (
            demanda_nominal
            / preco_mercado
        )

        demanda_total_real = (
            demanda_real
            + demanda_final_real
        )

        # ==========================================================
        # VENDAS DOMÉSTICAS
        # ==========================================================

        for firma in firmas_setor:

            oferta_firma_real = float(
                ofertas_firmas[firma.id]
            )

            # Participação da firma na oferta total.
            share_oferta = (
                oferta_firma_real
                / oferta_total_real
            )

            # A demanda é distribuída segundo a participação
            # de cada participante na oferta.
            firma.demanda_recebida_real = (
                share_oferta
                * demanda_total_real
            )

            # A firma não pode vender mais do que ofertou.
            firma.vendas_real = min(
                firma.demanda_recebida_real,
                oferta_firma_real,
            )

            # Todas as firmas domésticas recebem o mesmo
            # preço efetivamente realizado no mercado.
            firma.preco_transacao = (
                preco_mercado
            )

            firma.vendas_nominal = (
                firma.vendas_real
                * preco_mercado
            )

            firma.demanda_nao_atendida_real = max(
                0.0,
                firma.demanda_recebida_real
                - firma.vendas_real,
            )

            firma.producao_nao_vendida_real = max(
                0.0,
                firma.producao_real
                - firma.vendas_real,
            )

            firma.taxa_atendimento = (
                firma.vendas_real
                / firma.demanda_recebida_real
                if firma.demanda_recebida_real > 0.0
                else 1.0
            )

            firma.fator_atendimento = (
                firma.taxa_atendimento
            )

            participantes.append(
                {
                    "participante": firma.id,
                    "setor": setor,
                    "tipo": "domestico",
                    "oferta_real": oferta_firma_real,
                    "share_oferta": share_oferta,
                    "preco_oferta":
                        firma.preco_oferta_leilao,
                    "vendas_real":
                        firma.vendas_real,
                    "vendas_nominal":
                        firma.vendas_nominal,
                    "preco_transacao":
                        preco_mercado,
                }
            )

        # ==========================================================
        # IMPORTAÇÕES
        # ==========================================================

        share_oferta_importado = (
            oferta_importada_real
            / oferta_total_real
        )

        demanda_importado_real = (
            share_oferta_importado
            * demanda_total_real
        )

        importado.vendas_real = min(
            demanda_importado_real,
            oferta_importada_real,
        )

        importado.vendas_nominal = (
            importado.vendas_real
            * preco_mercado
        )

        participantes.append(
            {
                "participante":
                    f"importado::{setor}",

                "setor":
                    setor,

                "tipo":
                    "importado",

                "oferta_real":
                    oferta_importada_real,

                "share_oferta":
                    share_oferta_importado,

                "preco_oferta":
                    importado.preco,

                "vendas_real":
                    importado.vendas_real,

                "vendas_nominal":
                    importado.vendas_nominal,

                "preco_transacao":
                    preco_mercado,
            }
        )

        # ==========================================================
        # REGISTRO SETORIAL
        # ==========================================================

        resultados_setoriais.append(
            {
                "setor": setor,

                "oferta_domestica_real":
                    oferta_domestica_real,

                "oferta_importada_real":
                    oferta_importada_real,

                "oferta_total_real":
                    oferta_total_real,

                "demanda_real":
                    demanda_real,

                "demanda_nominal":
                    demanda_nominal,

                "demanda_final_real":
                    demanda_final_real,

                "demanda_total_real":
                    demanda_total_real,

                "preco_mercado":
                    preco_mercado,

                "erro_equilibrio": (
                    demanda_total_real
                    - oferta_total_real
                ),
            }
        )

    # ==========================================================
    # RESULTADOS
    # ==========================================================

    if resultados_setoriais:

        resultados_setoriais_df = (
            pd.DataFrame(
                resultados_setoriais
            )
            .set_index("setor")
        )

    else:

        resultados_setoriais_df = pd.DataFrame(
            columns=[
                "oferta_domestica_real",
                "oferta_importada_real",
                "oferta_total_real",
                "demanda_real",
                "demanda_nominal",
                "demanda_final_real",
                "demanda_total_real",
                "preco_mercado",
                "erro_equilibrio",
            ],
            index=pd.Index(
                [],
                name="setor",
            ),
        )

    if participantes:

        participantes_df = (
            pd.DataFrame(
                participantes
            )
            .set_index("participante")
        )

    else:

        participantes_df = pd.DataFrame(
            columns=[
                "setor",
                "tipo",
                "oferta_real",
                "share_oferta",
                "preco_oferta",
                "vendas_real",
                "vendas_nominal",
                "preco_transacao",
            ],
            index=pd.Index(
                [],
                name="participante",
            ),
        )

    return {
        "setores": resultados_setoriais_df,
        "participantes": participantes_df,
    }
