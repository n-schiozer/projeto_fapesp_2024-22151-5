"""Execução de um período econômico da trajetória SFC--IO--ABM."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from macro.demanda_autonoma_abm import calcular_demanda_autonoma

import mercados.calcular_precos_realizados_abm as calcular_precos_realizados_abm

from agentes.agregar_firmas import (
    agregar_firmas,
    agregar_resultados_realizados_firmas,
)
from contabilidade.distribuicao_abm import calcular_distribuicao_pre_mercado_abm
from contabilidade.calcular_fluxos_cei import calcular_fluxos_cei
from contabilidade.montar_cei import montar_cei
from investimento.investimento_abm import (
    SETOR_CONSTRUCAO,
    calcular_fbcf_familias,
)
from macro.ciclo_abm import (
    atualizar_financeiro_periodo,
    calcular_inflacao_periodo,
    calcular_juros_periodo,
    calcular_mercado_trabalho,
)
from mercados.executar_mercados_periodo_abm import executar_mercados_periodo
from mercados.regulacao_producao_abm import calcular_decisoes_regulador


def obter_fator_produtividade_climatica(
    *,
    CONFIG_ABM: dict,
    setor: str,
    periodo: int,
) -> float:
    """Retorna o multiplicador climático da produtividade do capital."""

    if not CONFIG_ABM.get("choques_climaticos", {}).get("ativo", False):
        return 1.0

    choque = CONFIG_ABM.get("choques_climaticos", {}).get("setores", {}).get(
        setor
    )
    if choque is None:
        return 1.0

    periodo_choque = int(choque["periodo_choque"])
    multiplicador = float(choque["multiplicador_produtividade"])
    choque_permanente = bool(choque.get("choque_permanente", False))
    if not 0.0 <= multiplicador <= 1.0:
        raise ValueError("multiplicador_produtividade deve estar entre 0 e 1.")

    if choque_permanente:
        if periodo >= periodo_choque:
            return multiplicador
    elif periodo == periodo_choque:
        return multiplicador
    return 1.0


def construir_diagnostico_capacidade_setorial(
    *,
    firmas: dict[str, Any],
    setores: list[str],
    periodo: int,
    depreciacao: float | None,
) -> list[dict]:
    """Calcula as identidades setoriais do período sem manter histórico."""

    linhas = []
    for setor in setores:
        firmas_setor = [
            firma for firma in firmas.values() if firma.setor == setor
        ]
        if not firmas_setor:
            continue

        capital = float(sum(firma.estoque_capital_real for firma in firmas_setor))
        producao_normal = float(
            sum(firma.producao_normal_real for firma in firmas_setor)
        )
        capacidade_estrutural = float(
            sum(
                firma.capacidade_produtiva_estrutural_real
                for firma in firmas_setor
            )
        )
        capacidade_efetiva = float(
            sum(firma.capacidade_produtiva_real for firma in firmas_setor)
        )
        capacidade_pos_clima = float(
            sum(
                firma.capacidade_produtiva_estrutural_real
                * firma.fator_produtividade_climatica
                for firma in firmas_setor
            )
        )
        producao_planejada = float(
            sum(firma.producao_planejada_real for firma in firmas_setor)
        )
        producao_real = float(sum(firma.producao_real for firma in firmas_setor))
        investimento_liquido = float(
            sum(firma.investimento_liquido for firma in firmas_setor)
        )
        investimento_reposicao = float(
            sum(firma.investimento_reposicao for firma in firmas_setor)
        )
        investimento_bruto = float(
            sum(firma.investimento_bruto for firma in firmas_setor)
        )

        if depreciacao is None:
            capital_seguinte = capital
        else:
            capital_seguinte = float(
                sum(
                    (1.0 - depreciacao) * firma.estoque_capital_real
                    + firma.investimento_bruto
                    for firma in firmas_setor
                )
            )

        utilizacao_normal = (
            producao_normal / capacidade_estrutural
            if capacidade_estrutural > 0.0
            else np.nan
        )
        fator_clima = (
            capacidade_pos_clima / capacidade_estrutural
            if capacidade_estrutural > 0.0
            else np.nan
        )
        fator_capacidade_total = (
            capacidade_efetiva / capacidade_estrutural
            if capacidade_estrutural > 0.0
            else np.nan
        )
        capitais_desejados = [
            firma.capital_desejado
            for firma in firmas_setor
            if np.isfinite(firma.capital_desejado)
        ]
        gaps_capital = [
            firma.gap_capital
            for firma in firmas_setor
            if np.isfinite(firma.gap_capital)
        ]
        linhas.append(
            {
                "periodo": periodo,
                "setor": setor,
                "produtividade_capital_normal": (
                    producao_normal / capital if capital > 0.0 else np.nan
                ),
                "utilizacao_capacidade_normal": utilizacao_normal,
                "producao_normal": producao_normal,
                "capacidade_estrutural": capacidade_estrutural,
                "capacidade_efetiva": capacidade_efetiva,
                "fator_clima": fator_clima,
                "fator_capacidade_total": fator_capacidade_total,
                "producao_planejada": producao_planejada,
                "producao_real": producao_real,
                "utilizacao_planejada": (
                    producao_planejada / capacidade_efetiva
                    if capacidade_efetiva > 0.0
                    else np.nan
                ),
                "demanda_esperada": float(
                    sum(firma.demanda_esperada for firma in firmas_setor)
                ),
                "capital": capital,
                "capital_desejado": (
                    float(sum(capitais_desejados))
                    if capitais_desejados
                    else np.nan
                ),
                "gap_capital": (
                    float(sum(gaps_capital)) if gaps_capital else np.nan
                ),
                "investimento_liquido": investimento_liquido,
                "investimento_reposicao": investimento_reposicao,
                "investimento_bruto": investimento_bruto,
                "capital_periodo_seguinte": capital_seguinte,
            }
        )
    return linhas


def executar_periodo(
    *,
    periodo: int,
    firmas: dict[str, Any],
    importados: dict[str, Any],
    estado: dict[str, Any],
    condicoes_iniciais: dict[str, Any],
    calibracoes: dict[str, Any],
    CONFIG: dict[str, Any],
    CONFIG_ABM: dict[str, Any],
) -> dict[str, Any]:
    """Realiza a economia de ``periodo`` e prepara o estado herdado por t+1."""

    # ==========================================================
    # 0. VALIDAÇÃO E LEITURA DO ESTADO HERDADO
    # ==========================================================
    # Até o bloco final, ``estado`` representa exclusivamente informação
    # recebida de t-1. Resultados correntes permanecem em variáveis locais.

    if periodo < 1:
        raise ValueError("executar_periodo representa somente períodos t >= 1.")
    t = periodo
    setores = list(condicoes_iniciais["setores"])
    lambda_expectativa_precos = float(
        CONFIG_ABM.get("lambda_expectativa_precos", 1.0)
    )
    if lambda_expectativa_precos < 0.0:
        raise ValueError("lambda_expectativa_precos não pode ser negativo.")
    periodo_choque = int(CONFIG["periodo_choque"])
    choque_permanente = CONFIG["choque_permanente"]
    if not isinstance(choque_permanente, bool):
        raise TypeError("choque_permanente deve ser True ou False.")
    if periodo_choque < 1 or periodo_choque > CONFIG["periodos"]:
        raise ValueError(
            "periodo_choque deve estar entre 1 e o número de períodos."
        )
    inertia_pm = float(CONFIG["inertia_pm"])
    taxa_juros_real = float(CONFIG["taxa_juros_real"])
    taxa_crescimento_populacional = float(
        CONFIG["taxa_crescimento_populacional"]
    )
    if taxa_crescimento_populacional < 0.0:
        raise ValueError(
            "taxa_crescimento_populacional não pode ser negativa."
        )
    fracao_reavaliacao_financeira = float(
        CONFIG["fracao_reavaliacao_financeira"]
    )
    # O índice populacional corrente é calculado a partir do valor herdado,
    # mas permanece local durante t. O estado só é atualizado no fechamento.
    indice_populacao_anterior = float(
        estado["macro"]["indice_populacao"]
    )
    indice_populacao = indice_populacao_anterior * (
        1.0 + taxa_crescimento_populacional
    )
    pea_base = float(calibracoes["cei"]["parametros"]["pea"])
    # A PEA do ano-base cresce com a população, ampliando a oferta potencial
    # de trabalho sem determinar diretamente o nível de emprego.
    pea_periodo = pea_base * indice_populacao

    # ==========================================================
    # 1. EXPECTATIVAS DE PREÇOS
    # ==========================================================
    # A expectativa de Pc usa somente os dois vetores herdados. Nenhuma
    # demanda ou transação corrente foi observada neste ponto.

    inflacao_pc_anterior = (
        estado["precos"]["pc_anterior"]
        / estado["precos"]["pc_anterior_2"]
        - 1.0
    )

    pc_esperado = estado["precos"]["pc_anterior"] * (
        1.0 + lambda_expectativa_precos * inflacao_pc_anterior
    )

    if np.any(pc_esperado <= 0.0):
        raise RuntimeError(f"Preço esperado não positivo no período {t}.")

    # ==========================================================
    # 2. CAPACIDADE, CLIMA E DECISÃO DE PRODUÇÃO
    # ==========================================================
    # O clima altera a capacidade efetiva, mas não destrói capital. Com essa
    # capacidade e a demanda esperada herdada, cada firma decide sua produção
    # antes de conhecer a demanda corrente.

    decisoes_producao = {}
    fatores_clima_firmas = {}
    # O despacho é uma instituição permanente dos setores regulados. O clima
    # altera somente os limites operacionais das firmas, nunca a regra usada
    # pelo regulador no benchmark ou antes/depois de um choque.
    for firma in firmas.values():

        # Choque climático do período.
        fator_clima_setorial = obter_fator_produtividade_climatica(
            CONFIG_ABM=CONFIG_ABM,
            setor=firma.setor,
            periodo=t,
        )
        # O cenário determina o choque climático do setor. A exposição
        # estrutural determina se ele chega à produtividade/capacidade da firma.
        fator_clima_firma = 1.0 + firma.exposicao_climatica * (
            fator_clima_setorial - 1.0
        )
        fatores_clima_firmas[firma.id] = fator_clima_firma

        # Uma única inovação idiossincrática por firma e período, realizada
        # antes do cálculo da capacidade. O fluxo aleatório é independente do
        # processo de qualidade.
        firma.atualizar_produtividade_idiossincratica(
            rng=estado["aleatoriedade"][
                "rng_produtividade_idiossincratica"
            ],
            rho=CONFIG_ABM["rho_produtividade_idiossincratica"],
            sigma=CONFIG_ABM["sigma_produtividade_idiossincratica"],
        )


        # Capacidade efetivamente disponível em t:
        #
        # industrial:
        #   cap = cap_normal * fator_clima_firma
        #
        # todas as firmas:
        #   cap = cap_normal * fator_clima * fator_produtividade_idiossincratica

        firma.atualizar_capacidade_produtiva(
            fator_produtividade_climatica=fator_clima_firma,
        )


        # Demanda esperada e decisão descentralizada de produção.
        firma.calcular_demanda_esperada(
            beta=CONFIG_ABM["velocidade_ajuste_expectativa_demanda"],
        )

        # Decisão de produção dado expectativa de demanda e capacidade de produção:
        decisoes_producao[firma.id] = firma.calcular_producao_desejada(
            parametro_estoque_desejado=CONFIG_ABM[
                "parametro_estoque_desejado"
            ],
            velocidade_ajuste_estoques=CONFIG_ABM[
                "velocidade_ajuste_estoques_firmas"
            ],
        )

    # ==========================================================
    # 3. REGULAÇÃO E PRODUÇÃO REALIZADA
    # ==========================================================
    # Somente setores configurados sofrem coordenação. A quantidade decidida
    # após a regulação é limitada pela capacidade e não será reaberta pelo
    # mercado: o mercado determinará vendas, não produção.
    decisoes_regulador = calcular_decisoes_regulador(
        firmas=firmas,
        decisoes_producao=decisoes_producao,
        setores_regulados=CONFIG_ABM["setores_regulados"],
        fatores_clima=fatores_clima_firmas,
        usar_despacho_por_atratividade=True,
    )

    for firma in firmas.values():
        quantidade_final = decisoes_regulador.get(
            firma.id,
            decisoes_producao[firma.id],
        )
        firma.realizar_producao(quantidade_final)

    # ==========================================================
    # 4. CI, TRABALHO, INVESTIMENTO E PREÇOS DAS FIRMAS
    # ==========================================================
    # Com a produção de t fechada, as firmas determinam insumos, trabalho,
    # investimento e preços de oferta. Custos e expectativas ainda usam preços,
    # salários e inflação herdados, nunca resultados futuros do mercado.
    for firma in firmas.values():
        firma.calcular_demanda_intermediaria()
        firma.calcular_demanda_trabalho()
        if firma.setor in calibracoes["investimento"]["setores_nf"]:
            firma.decidir_investimento(
                v=calibracoes["investimento"]["v"],
                depreciacao=calibracoes["investimento"]["depreciacao"],
                gamma_retorno=CONFIG_ABM[
                    "gamma_investimento_retorno"
                ],
                gamma_investimento_capacidade=CONFIG_ABM[
                    "gamma_investimento_capacidade"
                ],
            )

        else:
            firma.investimento_liquido = 0.0
            firma.investimento_reposicao = 0.0
            firma.investimento_bruto = 0.0
        firma.atualizar_custo_e_preco(
            precos_insumos=estado["precos"]["pc_anterior"],
            indice_salarios=estado["macro"]["indice_salarios"],
            inflacao=estado["macro"]["inflacao"],
        )
        firma.calcular_eob_recorrente_esperado()
        firma.calcular_dividendos()

    # ==========================================================
    # 5. AGREGAÇÃO MICRO E DEMANDAS AUTÔNOMAS
    # ==========================================================
    # Os objetos continuam sendo a fonte de verdade; a agregação apenas converte
    # decisões micro já fechadas em demandas setoriais. A FBCF familiar usa a
    # poupança de t-1, enquanto governo, FBCF fixa e exportações seguem os choques
    # exógenos configurados para t.

    agregados = agregar_firmas(firmas, setores)
    demanda_intermediaria_real = agregados["demanda_intermediaria_real"]
    investimento_nf_total = (
        agregados["investimento_bruto"]
        .reindex(calibracoes["investimento"]["setores_nf"])
        .fillna(0.0)
        .sum()
    )

    demanda_investimento_real = (
        calibracoes["investimento"]["pesos_bens_capital_nf"]
        .reindex(setores)
        .fillna(0.0)
        * investimento_nf_total
    ).rename("demanda_investimento_real")

    investimento_familias = calcular_fbcf_familias(
        estado["familias"]["poupanca_familias_anterior"],
        estado["precos"]["pc_anterior"],
        pc_esperado,
        SETOR_CONSTRUCAO,
        calibracoes["cei"]["parametros"]["prop_invest_fbcf_familias"],
    )
    fbcf_familias = investimento_familias["fbcf_familias_nominal"]

    fator_governo = 1.0
    fator_investimento = 1.0
    fator_exportacoes = 1.0
    if choque_permanente:
        if t >= periodo_choque:
            fator_governo = CONFIG["multiplicador_governo"]
            fator_investimento = CONFIG["multiplicador_investimento"]
            fator_exportacoes = CONFIG["multiplicador_exportacoes"]
    else:
        if t == periodo_choque:
            fator_governo = CONFIG["multiplicador_governo"]
            fator_investimento = CONFIG["multiplicador_investimento"]
            fator_exportacoes = CONFIG["multiplicador_exportacoes"]

    demandas_autonomas = calcular_demanda_autonoma(
        governo_base=condicoes_iniciais["governo_base"],
        fbcf_fixa_base=calibracoes["investimento"]["fbcf_fixa_base"],
        exportacoes_base=condicoes_iniciais["exportacoes_base"],
        precos_setoriais=pc_esperado,
        periodo=t,
        taxa_crescimento_demanda_autonoma=CONFIG.get(
            "taxa_crescimento_demanda_autonoma",
            0.0,
        ),
        fator_governo=fator_governo,
        fator_investimento=fator_investimento,
        fator_exportacoes=fator_exportacoes,
    )
    governo_nominal = demandas_autonomas["governo_nominal"]
    fbcf_fixa_nominal = demandas_autonomas["fbcf_fixa_nominal"]
    exportacoes_nominais = demandas_autonomas["exportacoes_nominais"]

    # ==========================================================
    # 6. CONDIÇÕES MACRO PRÉ-MERCADO
    # ==========================================================
    # Pc esperado determina inflação e juros provisórios usados na distribuição
    # pré-mercado. Esses valores são correntes de t, mas ainda não são gravados
    # no estado persistente.

    precos_pre_mercado = calcular_inflacao_periodo(
        condicoes_iniciais["consumo_base"],
        pc_esperado,
        estado["precos"]["indice_precos_anterior"],
    )
    indice_precos_pre_mercado = precos_pre_mercado["indice_precos"]
    inflacao_pre_mercado = precos_pre_mercado["inflacao"]
    juros = calcular_juros_periodo(
        estado["financeiro"]["ativos_financeiros"],
        estado["financeiro"]["passivos_financeiros"],
        indice_precos_pre_mercado,
        estado["precos"]["indice_precos_anterior"],
        taxa_juros_real,
        estado["macro"]["taxa_juros_nominal"],
        inertia_pm,
        t,
    )
    ativos_financeiros_corrigidos = juros["ativos_corrigidos"]
    passivos_financeiros_corrigidos = juros["passivos_corrigidos"]
    taxa_juros_nominal_periodo = juros["taxa_juros_nominal"]
    juros_recebidos = juros["juros_recebidos"]
    juros_pagos = juros["juros_pagos"]
    juros_liquidos = juros["juros_liquidos"]

    impostos_produtos = (
        condicoes_iniciais["taxa_impostos"]
        .reindex(setores)
        .fillna(0.0)
        * agregados["producao_nominal"]
        .reindex(setores)
        .fillna(0.0)
    ).rename("impostos_produtos")

    indice_setor_financeiro = CONFIG["setor_financeiro"]

    dados_firmas_cei_pre_mercado = {
        "ff": {
            "valor_adicionado": float(
                agregados["valor_adicionado"].iloc[indice_setor_financeiro]
            ),
            "salarios": float(
                agregados["salarios"].iloc[indice_setor_financeiro]
            ),
            "contribuicoes_efetivas": float(
                agregados["contribuicoes"].iloc[indice_setor_financeiro]
            ),
            "dividendos": float(
                agregados["dividendos"].iloc[indice_setor_financeiro]
            ),
            "outros_va": float(
                agregados["outros_va"].iloc[indice_setor_financeiro]
            ),
        },
        "nf": {
            "valor_adicionado": float(
                agregados["valor_adicionado"].sum()
                - agregados["valor_adicionado"].iloc[indice_setor_financeiro]
            ),
            "salarios": float(
                agregados["salarios"].sum()
                - agregados["salarios"].iloc[indice_setor_financeiro]
            ),
            "contribuicoes_efetivas": float(
                agregados["contribuicoes"].sum()
                - agregados["contribuicoes"].iloc[indice_setor_financeiro]
            ),
            "dividendos": float(
                agregados["dividendos"].sum()
                - agregados["dividendos"].iloc[indice_setor_financeiro]
            ),
            "outros_va": float(
                agregados["outros_va"].sum()
                - agregados["outros_va"].iloc[indice_setor_financeiro]
            ),
        },
        "ocupacoes": float(agregados["ocupacoes"].sum()),
    }
    distribuicao_pre_mercado = calcular_distribuicao_pre_mercado_abm(
        p=calibracoes["cei"]["parametros"],
        dados_firmas=dados_firmas_cei_pre_mercado,
        impostos_produtos=impostos_produtos,
        juros_recebidos=juros_recebidos,
        juros_pagos=juros_pagos,
        indice_salarios=estado["macro"]["indice_salarios"],
        indice_precos=indice_precos_pre_mercado,
        setor_financeiro=indice_setor_financeiro,
        outras_transferencias_base=condicoes_iniciais[
            "outras_transferencias_base"
        ],
    )

    # ==========================================================
    # 7. DEMANDA SETORIAL E MERCADOS
    # ==========================================================
    # A distribuição pré-mercado determina o consumo. Demandas reais de CI e
    # investimento e demandas finais nominais são então convertidas para PB.
    # O mercado aloca essas demandas entre firmas domésticas e importados.

    consumo_nominal = distribuicao_pre_mercado["consumo_nominal"]

    demanda_real_setorial = (
        demanda_intermediaria_real
        + demanda_investimento_real
    ).rename("demanda_real_setorial")

    demanda_final_pm_nominal = (
        consumo_nominal * calibracoes["consumo"]["pesos_consumo"]
        + governo_nominal
        + exportacoes_nominais
        + fbcf_familias
        * calibracoes["investimento"]["pesos_investimento_familias"]
        + fbcf_fixa_nominal
    ).rename("demanda_final_pm_nominal")

    demanda_nominal_setorial = (
        condicoes_iniciais["conversao_de_pm_pb"] @ demanda_final_pm_nominal
    ).rename("demanda_nominal_setorial")

    for importado in importados.values():
        importado.atualizar_preco(
            indice_cambio=estado["macro"]["indice_cambio"],
        )

    mercados = executar_mercados_periodo(
        setores=setores,
        firmas=firmas,
        importados=importados,
        demanda_real_setorial=demanda_real_setorial,
        demanda_nominal_setorial=demanda_nominal_setorial,
        ofertas_reguladas=decisoes_regulador,
    )

    # ==========================================================
    # 8. PREÇOS E RESULTADOS REALIZADOS
    # ==========================================================
    # Vendas e preços de transação já incorporam a alocação do mercado. A partir
    # daqui, custos, rentabilidade e estoques usam somente valores realizados;
    # a produção permanece a decisão ex ante já fechada.

    precos_realizados = (
        calcular_precos_realizados_abm
        .calcular_precos_realizados_abm(
            setores=setores,
            firmas=firmas,
            importados=importados,
            G=condicoes_iniciais["G"],
        )
    )

    pb_realizado = precos_realizados["pb"]
    pm_realizado = precos_realizados["pm"]
    pc_realizado = precos_realizados["pc"]

    pesos_capital = (
        calibracoes["investimento"]["pesos_bens_capital_nf"]
        .reindex(setores)
        .fillna(0.0)
    )
    pesos_capital /= pesos_capital.sum()
    preco_capital_realizado = float(
        (pesos_capital * pc_realizado.reindex(setores)).sum()
    )
    conversao_pm_pb = (
        condicoes_iniciais["conversao_de_pm_pb"]
        .reindex(
            index=setores,
            columns=setores,
        )
    )
    matriz_pm_pb = conversao_pm_pb.to_numpy(dtype=float)

    vendas_domesticas_setoriais = pd.Series(
        {
            setor: float(
                sum(
                    firma.vendas_nominal
                    for firma in firmas.values()
                    if firma.setor == setor
                )
            )
            for setor in setores
        },
        index=setores,
        dtype=float,
    )
    importacoes_basicas_nominais = pd.Series(
        {
            setor: float(importados[setor].vendas_nominal)
            for setor in setores
        },
        index=setores,
        dtype=float,
        name="importacoes_basicas_nominais",
    )
    vendas_totais_pb = vendas_domesticas_setoriais + importacoes_basicas_nominais

    # A demanda final já entrou no mercado convertida para PB. O resíduo das
    # vendas totais valoriza os usos reais de CI e investimento das firmas.
    valor_bloco_real_pb = vendas_totais_pb - demanda_nominal_setorial

    preco_efetivo_bloco_real = (
        valor_bloco_real_pb
        .div(
            demanda_real_setorial.replace(
                0.0,
                np.nan,
            )
        )
        .fillna(0.0)
    )

    for firma in firmas.values():
        ci_firma_pb = (
            firma.demanda_intermediaria_real
            .reindex(setores)
            .fillna(0.0)
            * preco_efetivo_bloco_real
        )

        ci_firma_pm = pd.Series(
            np.linalg.solve(
                matriz_pm_pb,
                ci_firma_pb.to_numpy(
                    dtype=float
                ),
            ),
            index=setores,
        )

        consumo_intermediario_firma = float(ci_firma_pm.sum())

        if firma.producao_real > 0.0:
            firma.custo_intermediario_unitario_realizado = (
                consumo_intermediario_firma
                / firma.producao_real
            )
        else:
            firma.custo_intermediario_unitario_realizado = 0.0

        firma.calcular_resultado_realizado()

        firma.calcular_taxa_retorno_observada(
            preco_capital=preco_capital_realizado,
            depreciacao=calibracoes["investimento"]["depreciacao"],
            taxa_juros_real=taxa_juros_real,
        )
        firma.estoque_final()

    agregados_realizados = agregar_resultados_realizados_firmas(
        firmas=firmas,
        setores=setores,
    )

    # ==========================================================
    # 9. ESTOQUES, IMPORTAÇÕES, FBCF E IMPOSTOS REALIZADOS
    # ==========================================================
    # Produção menos vendas gera estoques ou excedentes, sem impor equilíbrio.
    # Importações e usos realizados valorizam FBCF e a base tributável ex post.

    variacao_estoques_real = float(
        sum(
            firma.variacao_estoque_real
            for firma in firmas.values()
        )
    )

    variacao_estoques_nominal = float(
        sum(
            firma.variacao_estoque_real
            * firma.preco_transacao
            for firma in firmas.values()
        )
    )
    importacoes_nominais = float(importacoes_basicas_nominais.sum())

    # O investimento NF foi decidido em quantidade real; seu valor nominal
    # aparece somente agora, aos preços efetivamente realizados.
    investimento_nf_pb_realizado = (
        demanda_investimento_real
        .reindex(setores)
        .fillna(0.0)
        * preco_efetivo_bloco_real
    )

    investimento_nf_nominal_realizado = pd.Series(
        np.linalg.solve(
            matriz_pm_pb,
            investimento_nf_pb_realizado.to_numpy(
                dtype=float
            ),
        ),
        index=setores,
        name="investimento_nf_nominal_realizado",
    )

    fbcf_nf_nominal_realizada = float(
        investimento_nf_nominal_realizado.sum()
    )
    ci_pb_realizado = (
        demanda_intermediaria_real
        .reindex(setores)
        .fillna(0.0)
        * preco_efetivo_bloco_real
    )

    ci_pm_realizado = pd.Series(
        np.linalg.solve(
            matriz_pm_pb,
            ci_pb_realizado.to_numpy(
                dtype=float
            ),
        ),
        index=setores,
        name="ci_pm_realizado",
    )

    demanda_total_pm_nominal_realizada = (
        ci_pm_realizado
        + demanda_final_pm_nominal
        .reindex(setores)
        .fillna(0.0)
        + investimento_nf_nominal_realizado
    ).rename(
        "demanda_total_pm_nominal_realizada"
    )

    impostos_produtos_realizados = (
        condicoes_iniciais["taxa_impostos"]
        .reindex(setores)
        .fillna(0.0)
        * demanda_total_pm_nominal_realizada
    ).rename(
        "impostos_produtos_realizados"
    )

    impostos_produtos_total_realizado = float(
        impostos_produtos_realizados.sum()
    )

    # ==========================================================
    # 10. CEI, DISTRIBUIÇÃO E SISTEMA FINANCEIRO
    # ==========================================================
    # A CEI só é montada após os mercados porque VA, impostos, importações e
    # estoques devem refletir fluxos realizados. Sua capacidade de financiamento
    # fecha a aquisição de ativos e a emissão de passivos do período.
    fluxos_cei = calcular_fluxos_cei(
        distribuicao_pre_mercado=distribuicao_pre_mercado,
        agregados_realizados=agregados_realizados,
        impostos_produtos_total_realizado=impostos_produtos_total_realizado,
        fbcf_familias=fbcf_familias,
        fbcf_nf_nominal_realizada=fbcf_nf_nominal_realizada,
        fbcf_fixa_nominal=fbcf_fixa_nominal,
        importacoes_nominais=importacoes_nominais,
        exportacoes_nominais=exportacoes_nominais,
        governo_nominal=governo_nominal,
        variacao_estoques_nominal=variacao_estoques_nominal,
        condicoes_iniciais=condicoes_iniciais,
        calibracoes=calibracoes,
        CONFIG=CONFIG,
    )
    resultado_cei = montar_cei(
        estrutura_cei=condicoes_iniciais["cei_original"],
        fluxos_cei=fluxos_cei,
        teste_flag=CONFIG["executar_testes"],
    )
    cei_periodo = resultado_cei["cei"]
    capacidade_financiamento = resultado_cei["capacidade_financiamento"]
    poupanca_familias = float(
        distribuicao_pre_mercado["poupanca_familias"]
    )
    # Emprego corrente determina o salário usado no próximo período.
    mercado_trabalho = calcular_mercado_trabalho(
        agregados["ocupacoes"].sum(),
        pea_periodo,
        CONFIG["taxa_desemprego_base"],
        CONFIG["a0"],
        CONFIG["a1"],
        CONFIG["a3"],
    )

    emprego = mercado_trabalho["emprego"]
    taxa_desemprego = mercado_trabalho["taxa_desemprego"]
    variacao_salarios = max(
        0.0,
        mercado_trabalho["variacao_salarios"],
    )
    financeiro_periodo = atualizar_financeiro_periodo(
        capacidade_financiamento,
        fracao_reavaliacao_financeira,
        ativos_financeiros_corrigidos,
        passivos_financeiros_corrigidos,
    )

    reavaliacao_financeira = financeiro_periodo["reavaliacao_financeira"]
    aquisicao_ativos = financeiro_periodo["aquisicao_ativos"]
    emissao_passivos = financeiro_periodo["emissao_passivos"]
    ativos_financeiros_periodo = financeiro_periodo[
        "ativos_financeiros_periodo"
    ]
    passivos_financeiros_periodo = financeiro_periodo[
        "passivos_financeiros_periodo"
    ]
    estoque_financeiro_periodo = financeiro_periodo[
        "estoque_financeiro_periodo"
    ]

    # ==========================================================
    # 11. AGREGADOS MACRO REALIZADOS
    # ==========================================================
    # Com mercados e contabilidade fechados, os agregados de t são calculados
    # uma única vez. A inflação realizada usa Pc de transação, ao passo que os
    # juros do período foram determinados anteriormente com Pc esperado.
    inflacao_realizada_periodo = calcular_inflacao_periodo(
        condicoes_iniciais["consumo_base"],
        pc_realizado,
        estado["precos"]["indice_precos_anterior"],
    )
    indice_precos_realizado = inflacao_realizada_periodo["indice_precos"]
    inflacao_realizada = inflacao_realizada_periodo["inflacao"]
    pib_nominal = (
        float(agregados_realizados["valor_adicionado"].sum())
        + impostos_produtos_total_realizado
    )
    pib_real = pib_nominal/indice_precos_realizado
    producao_real_total = float(
        sum(firma.producao_real for firma in firmas.values())
    )
    vendas_real_total = float(
        sum(firma.vendas_real for firma in firmas.values())
    )
    governo_real = float((governo_nominal / pc_realizado).sum())
    fbcf_fixa_real = float((fbcf_fixa_nominal / pc_realizado).sum())
    exportacoes_real = float((exportacoes_nominais / pc_realizado).sum())
    consumo_real = float(
        (
            consumo_nominal
            * calibracoes["consumo"]["pesos_consumo"]
            / pc_realizado
        ).sum()
    )
    renda_disponivel_real = float(
        distribuicao_pre_mercado["renda_disponivel_familias"]
        / indice_precos_realizado
    )
    consumo_intermediario_real_total = float(demanda_intermediaria_real.sum())
    investimento_nf_real_total = float(demanda_investimento_real.sum())
    demanda_esperada_total = float(
        sum(firma.demanda_esperada for firma in firmas.values())
    )

    # ==========================================================
    # 12. DIAGNÓSTICOS ECONÔMICOS DE SAÍDA
    # ==========================================================
    # Os mecanismos já foram realizados. Estes diagnósticos apenas fotografam
    # capacidade, clima e regulação; identidades de implementação pertencem aos
    # testes e não são recalculadas no caminho normal da simulação.
    setores_regulados = set(CONFIG_ABM["setores_regulados"])
    diagnostico_clima_periodo = []
    diagnostico_regulacao_periodo = []
    for firma in firmas.values():
        producao_antes_regulacao = decisoes_producao[firma.id]
        producao_final = decisoes_regulador.get(
            firma.id,
            producao_antes_regulacao,
        )
        diagnostico_clima_periodo.append(
            {
                "periodo": t,
                "firma": firma.id,
                "setor": firma.setor,
                "fator_clima": firma.fator_produtividade_climatica,
                "fator_produtividade_idiossincratica": (
                    firma.fator_produtividade_idiossincratica
                ),
                "capital": firma.estoque_capital_real,
                "capacidade_estrutural": (
                    firma.capacidade_produtiva_estrutural_real
                ),
                "capacidade_efetiva": firma.capacidade_produtiva_real,
                "producao_desejada": firma.producao_desejada_real,
                "producao_antes_regulacao": producao_antes_regulacao,
                "redistribuicao_regulador": (
                    producao_final - producao_antes_regulacao
                ),
                "producao_planejada": firma.producao_planejada_real,
                "producao_real": firma.producao_real,
                "restricao_capacidade": (
                    firma.producao_restringida_capacidade_real
                ),
            }
        )
        diagnostico_regulacao_periodo.append(
            {
                "periodo": t,
                "setor": firma.setor,
                "firma": firma.id,
                "regulado": firma.setor in setores_regulados,
                "producao_desejada": firma.producao_desejada_real,
                "producao_final": producao_final,
                "capacidade_efetiva": firma.capacidade_produtiva_real,
            }
        )
    diagnostico_capacidade_periodo = construir_diagnostico_capacidade_setorial(
        firmas=firmas,
        setores=setores,
        periodo=t,
        depreciacao=calibracoes["investimento"]["depreciacao"],
    )

    # ==========================================================
    # 13. SAÍDA DO PERÍODO
    # ==========================================================
    # O payload registra a economia fechada de t. O histórico será criado fora
    # desta função por ``registrar_resultados_periodo``.
    resultado_macro_periodo = {
        "periodo": t,
        "ano": CONFIG["ano"] + t,
        "indice_populacao": indice_populacao,
        "pea": pea_periodo,
        "indice_precos": indice_precos_realizado,
        "inflacao": inflacao_realizada,
        "indice_salarios": estado["macro"]["indice_salarios"],
        "indice_cambio": estado["macro"]["indice_cambio"],
        "taxa_juros_nominal": taxa_juros_nominal_periodo,
        "pib_nominal": pib_nominal,
        "pib_real": pib_real,
        "producao_real": producao_real_total,
        "vendas_real": vendas_real_total,
        "emprego": emprego,
        "taxa_desemprego": taxa_desemprego,
        "consumo_nominal": float(consumo_nominal),
        "poupanca_familias_nominal": float(poupanca_familias),
        "fbcf_familias_nominal": float(fbcf_familias),
        "fbcf_nf_real": investimento_nf_real_total,
        "fbcf_nf_nominal": fbcf_nf_nominal_realizada,
        "fbcf_fixa_nominal": float(fbcf_fixa_nominal.sum()),
        "variacao_estoques_real": variacao_estoques_real,
        "variacao_estoques_nominal": variacao_estoques_nominal,
        "importacoes_nominais": importacoes_nominais,
        "exportacoes_nominais": float(exportacoes_nominais.sum()),
        "discrepancia_cei": float(resultado_cei["discrepancia"]),
        "governo_real": governo_real,
        "fbcf_fixa_real": fbcf_fixa_real,
        "exportacoes_real": exportacoes_real,
        "consumo_real": consumo_real,
        "renda_disponivel_real": renda_disponivel_real,
        "consumo_intermediario_real": consumo_intermediario_real_total,
        "investimento_nf_real": investimento_nf_real_total,
        "demanda_esperada_total": demanda_esperada_total,
        "vendas_real_total": vendas_real_total,
    }
    resultado_setorial_periodo = {
        "agregados_firmas": agregados,
        "agregados_firmas_realizados": agregados_realizados,
        "precos": {
            "precos_comprador": pc_realizado,
            "precos_basicos": pb_realizado,
            "precos_importacoes": pm_realizado,
            "precos_comprador_esperados": pc_esperado,
            "inflacao_precos_setorial": (
                pc_realizado / estado["precos"]["pc_anterior"] - 1.0
            ),
        },
        "investimento_capital_estoques": {
            "investimento_nf_real": demanda_investimento_real,
            "investimento_nf_nominal": investimento_nf_nominal_realizado,
            "fbcf_fixa_nominal": fbcf_fixa_nominal,
            "variacao_estoques_real": variacao_estoques_real,
            "variacao_estoques_nominal": variacao_estoques_nominal,
        },
        "mercados": mercados,
    }

    resultado_cei_periodo = {"cei": cei_periodo}

    resultado_financeiro_periodo = {
        "capacidade_financiamento": capacidade_financiamento,
        "estoque_financeiro": estoque_financeiro_periodo,
        "aquisicao_ativos_financeiros": aquisicao_ativos,
        "emissao_passivos_financeiros": emissao_passivos,
        "juros_liquidos": juros_liquidos,
        "juros_recebidos": juros_recebidos,
        "juros_pagos": juros_pagos,
        "reavaliacao_financeira": reavaliacao_financeira,
    }
    # ==========================================================
    # FIM DA ECONOMIA DO PERÍODO t
    # ==========================================================

    # ==========================================================
    # 14. ÚNICA ATUALIZAÇÃO DO ESTADO HERDADO POR t + 1
    # ==========================================================
    # Nenhuma linha abaixo calcula novamente a economia de t. Apenas deslocamos
    # resultados já fechados para os estados persistentes das firmas e do modelo.

    for firma in firmas.values():
        firma.atualizar_estado(calibracoes["investimento"]["depreciacao"])

        # Qualidade que será observada no período seguinte.
        firma.atualizar_qualidade(
            rng=estado["aleatoriedade"]["rng_qualidade"],
            rho_qualidade=CONFIG_ABM["rho_qualidade"],
            sigma_qualidade=CONFIG_ABM["sigma_qualidade"],
        )

    estado["macro"]["inflacao"] = inflacao_pre_mercado
    estado["macro"]["taxa_juros_nominal"] = taxa_juros_nominal_periodo
    estado["macro"]["indice_populacao"] = indice_populacao
    estado["macro"]["indice_salarios"] *= 1.0 + variacao_salarios
    estado["macro"]["indice_cambio"] *= (
        1.0
        + CONFIG["repasse_inflacao_cambio"] * inflacao_realizada
    )
    estado["precos"]["pc_anterior_2"] = estado["precos"][
        "pc_anterior"
    ].copy()
    estado["precos"]["pc_anterior"] = pc_realizado.copy()
    estado["precos"]["indice_precos_anterior"] = indice_precos_realizado
    estado["familias"]["poupanca_familias_anterior"] = float(
        poupanca_familias
    )
    estado["financeiro"]["ativos_financeiros"] = ativos_financeiros_periodo.copy()
    estado["financeiro"]["passivos_financeiros"] = passivos_financeiros_periodo.copy()

    return {
        "macro": resultado_macro_periodo,
        "setores": resultado_setorial_periodo,
        "cei": resultado_cei_periodo,
        "financeiro": resultado_financeiro_periodo,
        "diagnosticos": {
            "clima": diagnostico_clima_periodo,
            "regulacao": diagnostico_regulacao_periodo,
            "capacidade_setorial": diagnostico_capacidade_periodo,
        },
        "preco_capital": preco_capital_realizado,
    }
