"""Inicialização das firmas domésticas a partir da TRU de 2020.

Esta etapa apenas calibra os objetos ``Firma``. Não resolve mercados, não cria
o player importado e não altera a TRU, a CEI ou o ciclo temporal.
"""

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import VA
from agentes.firma import Firma


PARAMETROS_MARKUP_PADRAO = {
    "parametro_markup": 0.1,
    "markup_min": 0.0,
    "markup_max": 10.0,
    "epsilon_market_share": 1e-12,
}

ETA_ATENDIMENTO_PADRAO = 1.0


def codigo_setor_modelo(setor: str) -> str:
    """Extrai o código CNAE/TRU inicial do rótulo setorial do modelo."""

    codigo = str(setor).split(" - ", 1)[0].strip()
    if len(codigo) != 1 or not codigo.isalpha():
        raise ValueError(f"Rótulo setorial inválido: {setor!r}.")
    return codigo


def _aplicar_heterogeneidade_tecnologica(
    firmas: dict[str, Firma],
    config_abm: dict,
    probabilidades_exposicao: dict[str, float],
) -> None:
    """Sorteia tecnologia e recalibra somente o CI e o preço elétricos."""

    if not bool(config_abm.get("usar_heterogeneidade_tecnologica", False)):
        return

    semente = int(config_abm["semente_exposicao_climatica"])
    sementes_outros_processos = {
        int(config_abm["semente_qualidade"])
        if "semente_qualidade" in config_abm else None,
        int(config_abm.get("demografia_empresas", {}).get("semente"))
        if config_abm.get("demografia_empresas", {}).get("semente") is not None
        else None,
    }
    if semente in sementes_outros_processos:
        raise ValueError(
            "semente_exposicao_climatica deve ser distinta das sementes "
            "de qualidade e demografia."
        )

    # A tecnologia é uma característica estrutural sorteada uma única vez
    # após a calibração inicial e permanece fixa.
    rng_tecnologia = np.random.default_rng(semente)
    setor_eletricidade = next(
        (
            setor
            for setor in probabilidades_exposicao
            if codigo_setor_modelo(setor) == "D"
        ),
        None,
    )
    peso_exposta = float(
        config_abm["peso_relativo_ci_eletricidade_exposta"]
    )
    if peso_exposta <= 0.0:
        raise ValueError(
            "peso_relativo_ci_eletricidade_exposta deve ser positivo."
        )

    for firma in firmas.values():
        probabilidade = probabilidades_exposicao.get(firma.setor, 0.0)
        firma.exposicao_climatica = float(
            rng_tecnologia.random() < probabilidade
        )
        firma.peso_relativo_ci = (
            peso_exposta
            if firma.setor == setor_eletricidade
            and firma.exposicao_climatica == 1.0
            else 1.0
        )

    firmas_eletricidade = [
        firma
        for firma in firmas.values()
        if firma.setor == setor_eletricidade
    ]
    if not firmas_eletricidade:
        return

    # A heterogeneidade elétrica altera somente a intensidade de CI.
    # A normalização preserva o CI agregado calibrado pela TRU.
    peso_medio = sum(
        firma.share_domestico_inicial * firma.peso_relativo_ci
        for firma in firmas_eletricidade
    )
    if peso_medio <= 0.0:
        raise ValueError("O peso médio do CI elétrico deve ser positivo.")

    preco_base_setorial = sum(
        firma.share_domestico_inicial * firma.preco_relativo
        for firma in firmas_eletricidade
    )
    for firma in firmas_eletricidade:
        fator_normalizacao = firma.peso_relativo_ci / peso_medio
        firma.tecnologia *= fator_normalizacao
        firma.coeficientes_demanda_intermediaria *= fator_normalizacao
        firma.custo_intermediario_unitario_base = float(
            firma.tecnologia.sum()
        )
        firma.custo_intermediario_unitario = (
            firma.custo_intermediario_unitario_base
        )
        firma.custo_intermediario_unitario_esperado = (
            firma.custo_intermediario_unitario_base
        )
        firma.custo_intermediario_unitario_realizado = (
            firma.custo_intermediario_unitario_base
        )
        firma.custo_unitario = (
            firma.custo_intermediario_unitario
            + firma.remuneracao_unitaria
            + firma.outros_va_unitario
        )
        firma.custo_unitario_anterior = firma.custo_unitario

    custo_medio = sum(
        firma.share_domestico_inicial * firma.custo_unitario
        for firma in firmas_eletricidade
    )
    markup_comum = preco_base_setorial / custo_medio - 1.0

    # As firmas elétricas mantêm custos e preços individuais distintos.
    # O markup comum faz o preço ponderado reproduzir o mercado do ano-base.
    for firma in firmas_eletricidade:
        if not firma.markup_min <= markup_comum <= firma.markup_max:
            raise ValueError(
                "Os limites de markup não comportam a tecnologia elétrica."
            )
        preco_individual = (1.0 + markup_comum) * firma.custo_unitario
        firma.markup_base = markup_comum
        firma.markup = markup_comum
        firma.markup_leilao_base = markup_comum
        firma.markup_leilao = markup_comum
        firma.preco_minimo_leilao_base = firma.custo_unitario
        firma.preco_relativo = preco_individual
        firma.preco_firma = preco_individual
        firma.preco_oferta_leilao = preco_individual
        firma.preco_transacao = preco_individual
        firma.vendas_nominal = firma.vendas_real * preco_individual
        firma.eob_misto_unitario = preco_individual - firma.custo_unitario

    preco_ponderado = sum(
        firma.share_domestico_inicial * firma.preco_oferta_leilao
        for firma in firmas_eletricidade
    )
    if not np.isclose(preco_ponderado, preco_base_setorial, atol=1e-12):
        raise RuntimeError(
            "A recalibração elétrica não preservou o preço-base setorial."
        )


def inicializar_firmas(
    condicoes_iniciais: dict,
    config_abm: dict,
    config: dict | None = None,
    calibracao_investimento_nf_abm: dict | None = None,
    coortes_demografia: pd.DataFrame | None = None,
    calibracao_investimento_nf_legada: dict | None = None,
    parametros_cei: dict | None = None,
) -> dict[str, Firma]:
    """Abre a produção doméstica da TRU-base em objetos ``Firma``.

    Cada firma recebe a tecnologia e os coeficientes unitários do seu setor.
    Os valores setoriais são repartidos pelo market share doméstico inicial;
    assim, a soma das firmas preserva a TRU de 2020 por construção, sem usar
    qualquer mecanismo de mercado ou de ajuste dinâmico. Quando há coortes da
    Demografia, elas substituem apenas a abertura artificial por shares iguais.
    """

    ci = condicoes_iniciais

    config = ci["config"] if config is None else config

    setores = list(ci["setores"])

    setor_financeiro = setores[config["setor_financeiro"]]

    setores_leilao = set(config_abm.get("setores_leilao", []))

    desconhecidos = setores_leilao.difference(setores)

    probabilidades_exposicao = config_abm[
        "probabilidades_exposicao_climatica"
    ]
    if not isinstance(probabilidades_exposicao, dict):
        raise TypeError(
            "probabilidades_exposicao_climatica deve ser um dicionário."
        )
    setores_exposicao_desconhecidos = set(probabilidades_exposicao).difference(
        setores
    )
    if setores_exposicao_desconhecidos:
        raise KeyError(
            "Setores da exposição climática ausentes na TRU: "
            f"{setores_exposicao_desconhecidos}"
        )
    probabilidades_exposicao = {
        setor: float(probabilidade)
        for setor, probabilidade in probabilidades_exposicao.items()
    }
    for setor, probabilidade in probabilidades_exposicao.items():
        if not np.isfinite(probabilidade) or not 0.0 <= probabilidade <= 1.0:
            raise ValueError(
                "Probabilidade de exposição climática inválida em "
                f"{setor}: {probabilidade}."
            )

    parametros = ci["parametros_cei"] if parametros_cei is None else parametros_cei

    if desconhecidos:
        raise KeyError(f"Setores de leilão ausentes na TRU: {desconhecidos}")

    # Produção doméstica e CI da TRU-base, com Pb = Pc = 1 no período 0.

    producao_setorial = (
        ci["conversao_domestica"] @ ci["demanda_final_base"]
    ).reindex(setores)

    # Somente a diagonal é variação física do próprio produto. Os termos fora
    # dela são serviços correntes de comércio/transporte.

    calibracao_investimento_nf_legada = (
        ci["investimento_nf"]
        if calibracao_investimento_nf_legada is None
        else calibracao_investimento_nf_legada
    )
    estoque_inicial_2020 = (
        calibracao_investimento_nf_legada["estoques_base"]
        .reindex(setores)
        .fillna(0.0)
    )

    forma_estoque = estoque_inicial_2020.abs() > 1e-9

    estoque_inicial_2020 = (
        float(config_abm["parametro_estoque_desejado"])
        * producao_setorial
        * forma_estoque.astype(float)
    )

    diagonal_conversao = np.diag(
        ci["conversao_domestica"].reindex(index=setores, columns=setores)
    )   

    variacao_estoque_setorial = (
        calibracao_investimento_nf_legada["estoques_base"]
        .reindex(setores)
        .fillna(0.0)
        .astype(float)
    )

    estoque_final_2020 = (
        estoque_inicial_2020 + variacao_estoque_setorial
    )

    if calibracao_investimento_nf_abm is None:
        calibracao_nf = calibracao_investimento_nf_legada
        if config["inicializacao_investimento_nf"] == "estacionaria":
            capital_setorial = (
                calibracao_nf["investimento_nf_base_por_investidor"]
                / calibracao_nf["depreciacao"]
            )
        elif config["inicializacao_investimento_nf"] == "historica":
            capital_setorial = calibracao_nf["estoque_capital_nf_base"]
        else:
            raise ValueError(
                "inicializacao_investimento_nf deve ser "
                "'estacionaria' ou 'historica'."
            )
    else:
        capital_setorial = calibracao_investimento_nf_abm[
            "estoque_capital_inicial"
        ]

    capital_setorial = capital_setorial.reindex(setores).fillna(0.0)

    numeros = config_abm.get("numero_firmas_por_setor", {})

    shares_configurados = config_abm.get("market_shares_domesticos", {})

    precos_configurados = config_abm.get("precos_relativos_iniciais", {})

    ajustes_setoriais = config_abm.get("ajustes_setoriais", {})

    parametros_markup = {
        **PARAMETROS_MARKUP_PADRAO,
        **config_abm.get("parametros_markup", {}),
    }

    if coortes_demografia is not None:
        # A demografia determina apenas o número de firmas e seus
        # market shares iniciais.
        colunas_demografia = {
            "setor",
            "id_firma",
            "market_share_domestico",
        }
        faltantes = colunas_demografia.difference(coortes_demografia.columns)
        if faltantes:
            raise KeyError(
                "Coortes demográficas sem colunas obrigatórias: "
                f"{sorted(faltantes)}"
            )
        coortes_demografia = coortes_demografia.copy()
        coortes_demografia["setor"] = (
            coortes_demografia["setor"].astype(str).str.strip()
        )

    firmas: dict[str, Firma] = {}

    for indice_setor, setor in enumerate(setores):

        regime = "leilao" if setor in setores_leilao else "industrial"
        ajustes = ajustes_setoriais.get(setor, {})
        codigo = codigo_setor_modelo(setor)
        coortes_setor = None
        if coortes_demografia is not None and setor != setor_financeiro:
            coortes_setor = coortes_demografia.loc[
                coortes_demografia["setor"] == codigo
            ].copy()
            if coortes_setor.empty:
                raise ValueError(
                    f"Não há coortes demográficas para o setor {setor}."
                )

        numero_firmas = (
            len(coortes_setor)
            if coortes_setor is not None
            else 1
            if setor == setor_financeiro and coortes_demografia is not None
            else int(
                numeros.get(
                    setor,
                    config_abm[
                        "numero_firmas_leilao"
                        if regime == "leilao"
                        else "numero_firmas_industria"
                    ],
                )
            )
        )
        if numero_firmas < 1:
            raise ValueError(f"{setor} deve possuir pelo menos uma firma.")

        eta_preco = float(ajustes.get("eta_preco", config_abm["eta_preco_padrao"]))

        eta_qualidade = float(
            ajustes.get("eta_qualidade", config_abm["eta_qualidade_padrao"])
        )

        eta_atendimento = float(
            ajustes.get(
                "eta_atendimento",
                config_abm.get("eta_atendimento_padrao", ETA_ATENDIMENTO_PADRAO),
            )
        )
        if eta_preco >= 0.0 or eta_qualidade <= 0.0 or eta_atendimento <= 0.0:
            raise ValueError(f"Elasticidades inválidas em {setor}.")

        shares = np.asarray(
            coortes_setor["market_share_domestico"]
            if coortes_setor is not None
            else shares_configurados.get(
                setor, np.full(numero_firmas, 1.0 / numero_firmas)
            ),
            dtype=float,
        )

        precos_relativos = np.asarray(
            precos_configurados.get(setor, np.ones(numero_firmas)),
            dtype=float,
        )
        if len(shares) != numero_firmas or len(precos_relativos) != numero_firmas:
            raise ValueError(f"Dimensão incorreta de shares ou preços em {setor}.")
        if (shares < 0.0).any() or not np.isclose(shares.sum(), 1.0):
            raise ValueError(f"Market shares domésticos inválidos em {setor}.")
        if (precos_relativos <= 0.0).any():
            raise ValueError(f"Preço relativo não positivo em {setor}.")
        if not np.isclose(float(shares @ precos_relativos), 1.0):
            raise ValueError(
                f"Os preços relativos de {setor} não preservam Pb0 = 1."
            )

        # A tecnologia original é calibrada a preços ao comprador e, por isso,
        # incorpora margens de comércio, transporte e impostos sobre produtos.
        # Ela continua sendo usada para calcular o custo intermediário da firma.

        tecnologia = ci["A_precos"][setor].reindex(setores).astype(float)

        # Para gerar a demanda intermediária dirigida aos setores produtores,
        # os coeficientes são convertidos para preços básicos. A conversão
        # redistribui margens para Comércio e Transporte e retira impostos,
        # sem alterar a produção da firma.

        coeficientes_demanda_intermediaria = ci["conversao_de_pm_pb"] @ tecnologia.reindex(setores)

        # A TRU determina os agregados econômicos do setor. Os market shares
        # distribuem esses valores entre as firmas.

        # ==========================================================
        # UTILIZAÇÃO NORMAL DA CAPACIDADE (u*)
        # ==========================================================
        # A nova chave é independente do regime de preço. As chaves antigas
        # permanecem como fallback para reproduzir configurações existentes.
        utilizacao_por_setor = config_abm.get(
            "utilizacao_capacidade_normal_por_setor", {}
        )
        utilizacao_padrao = config_abm.get(
            "utilizacao_capacidade_normal", None
        )
        if isinstance(utilizacao_padrao, dict):
            utilizacao_padrao = utilizacao_padrao.get(setor)

        if utilizacao_padrao is None:
            utilizacao_padrao = config_abm.get(
                "utilizacao_capacidade_inicial_leilao"
                if regime == "leilao"
                else "utilizacao_capacidade_inicial_industrial",
                1.00 if regime == "leilao" else 0.80,
            )

        utilizacao_capacidade_normal = float(
            ajustes.get(
                "utilizacao_capacidade_normal",
                utilizacao_por_setor.get(setor, utilizacao_padrao),
            )
        )



        remuneracao_unitaria = float(
            ci["razoes_va"].at[
                VA["remuneracoes"],
                setor,
            ]
        )

        salario_unitaria = float(
            ci["razoes_va"].at[
                VA["salarios"],
                setor,
            ]
        )

        contribuicao_unitaria = float(
            ci["razoes_va"].at[
                VA["contribuicoes_efetivas"],
                setor,
            ]
        )

        eob_misto_unitario = float(
            ci["razoes_va"].at[
                VA["eob_mais_misto"],
                setor,
            ]
        )

        remuneracao_unitaria = float(ci["razoes_va"].at[VA["remuneracoes"], setor])
        
        salario_unitaria = float(ci["razoes_va"].at[VA["salarios"], setor])

        contribuicao_unitaria = float(ci["razoes_va"].at[VA["contribuicoes_efetivas"], setor])

        eob_misto_unitario = float(ci["razoes_va"].at[VA["eob_mais_misto"], setor])

        outros_va_unitario = float(
            ci["v0"].at[setor]
            - remuneracao_unitaria
            - eob_misto_unitario
        )

        if setor == setor_financeiro:
            parametro_dividendos = parametros["razao_divendos_eob_ff"]
        else:
            parametro_dividendos = parametros["razao_divendos_eob_nf"]

        ocupacoes_unitario = float(ci["razoes_va"].at[VA["ocupacoes"], setor])

        custo_intermediario_unitario = float(tecnologia.sum())

        custo_unitario = (
            custo_intermediario_unitario
            + remuneracao_unitaria
            + outros_va_unitario
        )
        preco_minimo_leilao = (
            custo_unitario if regime == "leilao" else np.nan
        )

        for posicao, (share, preco_relativo) in enumerate(
            zip(shares, precos_relativos, strict=True), start=1
        ):
            exposicao_climatica = float(
                probabilidades_exposicao.get(setor, 0.0) > 0.0
            )
            coorte = (
                coortes_setor.iloc[posicao - 1]
                if coortes_setor is not None
                else None
            )
            id_firma = (
                str(coorte["id_firma"])
                if coorte is not None
                else f"{codigo}_{posicao:03d}"
            )
            qualidade = np.nan
            if regime == "industrial":
                # Calibra só a heterogeneidade doméstica. O importado ainda não
                # existe como player nesta etapa.
                qualidade = float(
                    (
                        max(
                            float(share),
                            float(parametros_markup["epsilon_market_share"]),
                        )
                        / preco_relativo**eta_preco
                    )
                    ** (1.0 / eta_qualidade)
                )

            producao = float(share * producao_setorial.at[setor])

            variacao = float(
                share * variacao_estoque_setorial.at[setor]
            )
            estoque = float(share * estoque_final_2020.at[setor])

            producao_vendida = producao - variacao
            if producao_vendida < 0.0:
                raise ValueError(f"Produção vendida-base negativa em {id_firma}.")


            firmas[id_firma] = Firma(
                id=id_firma,
                setor=setor,
                regime=regime,
                eta_preco=eta_preco,
                eta_qualidade=eta_qualidade,
                eta_atendimento=eta_atendimento,
                share_domestico_inicial=float(share),
                # Participação no mercado total: doméstico relativo vezes a
                # parcela doméstica observada. O share doméstico relativo é
                # mantido separadamente para a agregação transitória de Pb.
                share_total_inicial=float(
                    (1.0 - ci["parcela_importada"].at[setor]) * share
                ),
                preco_relativo=float(preco_relativo),
                qualidade=qualidade,
                tecnologia=tecnologia.copy(),
                exposicao_climatica=exposicao_climatica,
                # Peso neutro na construção. A heterogeneidade tecnológica,
                # quando habilitada, é aplicada e normalizada logo abaixo.
                peso_relativo_ci=1.0,
                remuneracao_unitaria_base=remuneracao_unitaria,
                salario_unitaria_base = salario_unitaria,
                contribuicao_unitaria_base = contribuicao_unitaria,
                eob_misto_unitario_base=eob_misto_unitario,
                outros_va_unitario_base=outros_va_unitario,
                ocupacoes_unitario_base=ocupacoes_unitario,
                custo_intermediario_unitario_base=custo_intermediario_unitario,
                parametro_markup=float(parametros_markup["parametro_markup"]),
                markup_min=float(parametros_markup["markup_min"]),
                markup_max=float(parametros_markup["markup_max"]),
                epsilon_market_share=float(
                    parametros_markup["epsilon_market_share"]
                ),
                preco_minimo_leilao_base=preco_minimo_leilao,
                forma_estoque=bool(forma_estoque.at[setor]),
                estoque = estoque,
                estoque_capital_inicial_real=float(share * capital_setorial.at[setor]),
                producao_base_real=producao,
                producao_vendida_base_real=producao_vendida,
                estoque_capital_real=float(
                    share * capital_setorial.at[setor]
                ),
                utilizacao_capacidade_normal=utilizacao_capacidade_normal,
                producao_anterior=producao,
                demanda_esperada=producao,
                demanda_realizada=producao_vendida,
                parametro_dividendos=parametro_dividendos,
                coeficientes_demanda_intermediaria=(
                    coeficientes_demanda_intermediaria
                ),
                numero_firmas_representadas=(
                    int(coorte.get("numero_firmas_representadas", 1))
                    if coorte is not None else 1
                ),
                pessoal_ocupado_demografia=(
                    float(coorte.get("pessoal_ocupado_demografia", np.nan))
                    if coorte is not None else np.nan
                ),
                pessoal_ocupado_firma=(
                    float(coorte.get("pessoal_ocupado_firma", np.nan))
                    if coorte is not None else np.nan
                ),
                pessoal_ocupado_medio_por_firma=(
                    float(coorte.get("pessoal_ocupado_medio_por_firma", np.nan))
                    if coorte is not None else np.nan
                ),
                pessoal_ocupado_minimo_original=(
                    float(coorte.get("pessoal_ocupado_minimo_original", np.nan))
                    if coorte is not None else np.nan
                ),
                pessoal_ocupado_maximo_original=(
                    float(coorte.get("pessoal_ocupado_maximo_original", np.nan))
                    if coorte is not None else np.nan
                ),
                multiplicador_tru=(
                    float(coorte.get("multiplicador_tru", 1.0))
                    if coorte is not None else 1.0
                ),
                peso_variedade=(
                    float(coorte.get("peso_variedade", 1.0))
                    if coorte is not None else 1.0
                ),
                faixa_pessoal=(
                    str(coorte.get("faixa_pessoal", ""))
                    if coorte is not None else ""
                ),
                faixa_inicial=(
                    str(coorte.get("faixa_inicial", ""))
                    if coorte is not None else ""
                ),
                faixa_final=(
                    str(coorte.get("faixa_final", ""))
                    if coorte is not None else ""
                ),
                tipo_agente=(
                    str(coorte.get("tipo_agente", "firma"))
                    if coorte is not None else "firma"
                ),
            )

    _aplicar_heterogeneidade_tecnologica(
        firmas,
        config_abm,
        probabilidades_exposicao,
    )
    return firmas
