"""Distribuição de renda pré-mercado do ABM, sem construir a CEI."""

import numpy as np
import pandas as pd

from contabilidade.estrutura_cei import C, L


CONTAS_BASE_IR_FIRMAS = (
    "valor_adicionado",
    "salarios",
    "contribuicoes_efetivas",
    "impostos_produtos",
    "outros_impostos",
    "juros",
    "contribuicoes_sociais",
    "beneficios",
    "aposentadorias",
    "outras_transferencias",
)
CONTAS_ENTRADAS_DIVIDENDOS = CONTAS_BASE_IR_FIRMAS[:6]
CONTAS_SAIDAS_DIVIDENDOS = (
    "salarios",
    "contribuicoes_efetivas",
    "impostos_produtos",
    "outros_impostos",
    "juros",
    "ir",
    "contribuicoes_sociais",
    "beneficios",
    "aposentadorias",
    "outras_transferencias",
)
CONTAS_BASE_IR_FAMILIAS = CONTAS_ENTRADAS_DIVIDENDOS
CONTAS_ENTRADAS_RENDA_DISPONIVEL = (
    "valor_adicionado",
    "salarios",
    "contribuicoes_efetivas",
    "impostos_produtos",
    "outros_impostos",
    "juros",
    "dividendos",
    "ir",
    "contribuicoes_sociais",
    "beneficios",
    "aposentadorias",
    "outras_transferencias",
)
CONTAS_SAIDAS_RENDA_DISPONIVEL = (
    "valor_adicionado",
    "salarios",
    "contribuicoes_efetivas",
    "impostos_produtos",
    "outros_impostos",
    "juros",
    "dividendos",
    "ir",
)


def extrair_fluxos_template_distribuicao(
    valores_cei: pd.DataFrame,
    inflation_index: float,
) -> dict:
    """Expõe, por nome econômico, os fluxos-base ainda lidos pela distribuição."""

    contas = {
        "valor_adicionado": L["va"],
        "salarios": L["salarios"],
        "contribuicoes_efetivas": L["contribuicoes_efetivas"],
        "impostos_produtos": L["impostos_produtos"],
        "outros_impostos": L["outros_impostos"],
        "juros": L["juros"],
        "dividendos": L["dividendos"],
        "ir": L["ir"],
        "contribuicoes_sociais": L["contribuicoes_sociais"],
        "beneficios": L["beneficios"],
        "aposentadorias": L["aposentadorias"],
        "outras_transferencias": L["outras_transferencias"],
    }
    colunas = {
        "familias": (C["familias_e"], C["familias_s"]),
        "firmas_financeiras": (C["ff_e"], C["ff_s"]),
        "firmas_nao_financeiras": (C["nf_e"], C["nf_s"]),
    }
    return {
        setor: {
            "entrada": {
                conta: float(valores_cei.iat[linha, entrada]) * inflation_index
                for conta, linha in contas.items()
            },
            "saida": {
                conta: float(valores_cei.iat[linha, saida]) * inflation_index
                for conta, linha in contas.items()
            },
        }
        for setor, (entrada, saida) in colunas.items()
    }


def calcular_distribuicao_pre_mercado(
    parametros: dict,
    dados_firmas: dict,
    impostos_produtos: pd.Series,
    juros_recebidos: pd.Series,
    juros_pagos: pd.Series,
    indice_salarios: float,
    inflation_index: float,
    setor_financeiro: int,
    fluxos_template: dict,
) -> dict:
    """Calcula a renda, o consumo e a previdência antes do mercado corrente."""

    ff = dados_firmas["ff"]
    nf = dados_firmas["nf"]
    va_planejado_ff = ff["valor_adicionado"]
    va_planejado_nf = nf["valor_adicionado"]
    impostos_produtos_ff = float(impostos_produtos.iloc[setor_financeiro])
    impostos_produtos_nf = float(impostos_produtos.sum() - impostos_produtos_ff)

    ff_entrada = fluxos_template["firmas_financeiras"]["entrada"].copy()
    ff_saida = fluxos_template["firmas_financeiras"]["saida"].copy()
    nf_entrada = fluxos_template["firmas_nao_financeiras"]["entrada"].copy()
    nf_saida = fluxos_template["firmas_nao_financeiras"]["saida"].copy()
    familias_entrada = fluxos_template["familias"]["entrada"].copy()
    familias_saida = fluxos_template["familias"]["saida"].copy()

    ff_entrada["valor_adicionado"] = va_planejado_ff + impostos_produtos_ff
    nf_entrada["valor_adicionado"] = va_planejado_nf + impostos_produtos_nf
    ff_saida["salarios"] = ff["salarios"]
    nf_saida["salarios"] = nf["salarios"]
    familias_entrada["salarios"] = ff["salarios"] + nf["salarios"]
    ff_saida["contribuicoes_efetivas"] = ff["contribuicoes_efetivas"]
    nf_saida["contribuicoes_efetivas"] = nf["contribuicoes_efetivas"]
    familias_entrada["contribuicoes_efetivas"] = (
        ff["contribuicoes_efetivas"] + nf["contribuicoes_efetivas"]
    )
    ff_saida["impostos_produtos"] = impostos_produtos_ff
    nf_saida["impostos_produtos"] = impostos_produtos_nf
    ff_saida["outros_impostos"] = ff["outros_va"]
    nf_saida["outros_impostos"] = nf["outros_va"]
    ff_entrada["juros"] = float(juros_recebidos.loc["firmas_financeiras"])
    ff_saida["juros"] = float(juros_pagos.loc["firmas_financeiras"])
    nf_entrada["juros"] = float(juros_recebidos.loc["firmas_nao_financeiras"])
    nf_saida["juros"] = float(juros_pagos.loc["firmas_nao_financeiras"])
    familias_entrada["juros"] = float(juros_recebidos.loc["familias"])
    familias_saida["juros"] = float(juros_pagos.loc["familias"])

    base_ir_ff = float(
        np.asarray(
            [ff_entrada[conta] for conta in CONTAS_BASE_IR_FIRMAS], dtype=float
        ).sum()
        - np.asarray(
            [ff_saida[conta] for conta in CONTAS_BASE_IR_FIRMAS], dtype=float
        ).sum()
    )
    base_ir_nf = float(
        np.asarray(
            [nf_entrada[conta] for conta in CONTAS_BASE_IR_FIRMAS], dtype=float
        ).sum()
        - np.asarray(
            [nf_saida[conta] for conta in CONTAS_BASE_IR_FIRMAS], dtype=float
        ).sum()
    )
    ir_ff = parametros["taxa_ir_ff"] * base_ir_ff
    ir_nf = parametros["taxa_ir_nf"] * base_ir_nf
    ff_saida["ir"] = ir_ff
    nf_saida["ir"] = ir_nf

    base_dividendos_ff = float(
        np.asarray(
            [ff_entrada[conta] for conta in CONTAS_ENTRADAS_DIVIDENDOS],
            dtype=float,
        ).sum()
        - np.asarray(
            [ff_saida[conta] for conta in CONTAS_SAIDAS_DIVIDENDOS],
            dtype=float,
        ).sum()
    )
    base_dividendos_nf = float(
        np.asarray(
            [nf_entrada[conta] for conta in CONTAS_ENTRADAS_DIVIDENDOS],
            dtype=float,
        ).sum()
        - np.asarray(
            [nf_saida[conta] for conta in CONTAS_SAIDAS_DIVIDENDOS],
            dtype=float,
        ).sum()
    )
    dividendos_ff = parametros["razao_dividendos_ff"] * base_dividendos_ff
    dividendos_nf = parametros["razao_dividendos_nf"] * base_dividendos_nf
    dividendos_total = dividendos_ff + dividendos_nf
    dividendos_familias = (
        parametros["parcela_dividendos_familias"] * dividendos_total
    )
    dividendos_exterior = (
        parametros["parcela_dividendos_exterior"] * dividendos_total
    )
    familias_entrada["dividendos"] = dividendos_familias

    base_ir_familias = float(
        np.asarray(
            [familias_entrada[conta] for conta in CONTAS_BASE_IR_FAMILIAS],
            dtype=float,
        ).sum()
    )
    ir_familias = parametros["taxa_ir_familias"] * base_ir_familias
    familias_saida["ir"] = ir_familias

    emprego = float(dados_firmas["ocupacoes"])
    desempregados = parametros["pea"] - emprego
    beneficios = (
        parametros["beneficio_fixo"] * inflation_index
        + parametros["beneficio_por_desempregado"]
        * indice_salarios
        * desempregados
        / 1_000_000.0
    )
    aposentadorias = (
        parametros["aposentadoria_por_pessoa"]
        * inflation_index
        * parametros["aposentados"]
    )
    familias_entrada["beneficios"] = beneficios
    familias_entrada["aposentadorias"] = aposentadorias

    renda_disponivel_familias = float(
        np.asarray(
            [
                familias_entrada[conta]
                for conta in CONTAS_ENTRADAS_RENDA_DISPONIVEL
            ],
            dtype=float,
        ).sum()
        - np.asarray(
            [familias_saida[conta] for conta in CONTAS_SAIDAS_RENDA_DISPONIVEL],
            dtype=float,
        ).sum()
    )
    consumo_cei = parametros["propensao_consumir"] * renda_disponivel_familias
    poupanca_familias = float(
        pd.Series(
            [familias_entrada[conta] for conta in CONTAS_ENTRADAS_RENDA_DISPONIVEL],
            dtype=object,
        ).sum()
        - pd.Series(
            [familias_saida[conta] for conta in CONTAS_SAIDAS_RENDA_DISPONIVEL],
            dtype=object,
        ).sum()
        - consumo_cei
    )
    previdencia_familias = (
        parametros["prop_invest_prev_familias"] * poupanca_familias
    )

    return {
        "va_planejado_ff": va_planejado_ff,
        "va_planejado_nf": va_planejado_nf,
        "salarios_ff": ff["salarios"],
        "salarios_nf": nf["salarios"],
        "contribuicoes_efetivas_ff": ff["contribuicoes_efetivas"],
        "contribuicoes_efetivas_nf": nf["contribuicoes_efetivas"],
        "outros_impostos_ff": ff["outros_va"],
        "outros_impostos_nf": nf["outros_va"],
        "impostos_produtos_ff": impostos_produtos_ff,
        "impostos_produtos_nf": impostos_produtos_nf,
        "juros_recebidos": juros_recebidos,
        "juros_pagos": juros_pagos,
        "base_ir_ff": base_ir_ff,
        "base_ir_nf": base_ir_nf,
        "ir_ff": ir_ff,
        "ir_nf": ir_nf,
        "base_dividendos_ff": base_dividendos_ff,
        "base_dividendos_nf": base_dividendos_nf,
        "dividendos_ff": dividendos_ff,
        "dividendos_nf": dividendos_nf,
        "dividendos_total": dividendos_total,
        "dividendos_familias": dividendos_familias,
        "dividendos_exterior": dividendos_exterior,
        "beneficios": beneficios,
        "aposentadorias": aposentadorias,
        "aposentadorias_governo": parametros["parcela_governo_aposentadoria"]
        * aposentadorias,
        "aposentadorias_ff": parametros["parcela_ff_aposentadoria"]
        * aposentadorias,
        "base_ir_familias": base_ir_familias,
        "ir_familias": ir_familias,
        "renda_disponivel_familias": renda_disponivel_familias,
        "consumo_cei": consumo_cei,
        "poupanca_familias": poupanca_familias,
        "previdencia_familias": previdencia_familias,
        "previdencia_publica": parametros["prop_prev_publica"]
        * previdencia_familias,
        "previdencia_privada": parametros["prop_prev_privada"]
        * previdencia_familias,
        "emprego": emprego,
    }



def calcular_distribuicao_pre_mercado_abm(
    p: dict,
    dados_firmas: dict,
    impostos_produtos: pd.Series,
    juros_recebidos: pd.Series,
    juros_pagos: pd.Series,
    indice_salarios: float,
    indice_precos: float,
    setor_financeiro: int,
    outras_transferencias_base: dict,
) -> dict:
    """
    Calcula a distribuição de renda antes do mercado.

    Os fluxos das firmas vêm das decisões produtivas do ABM.
    Benefícios, aposentadorias e outras transferências correntes
    seguem regras calibradas a partir da CEI do ano-base.

    As outras transferências correntes são mantidas constantes
    em termos reais e atualizadas pelo índice geral de preços.
    """

    if indice_salarios <= 0.0:
        raise ValueError(
            "indice_salarios deve ser positivo."
        )

    if indice_precos <= 0.0:
        raise ValueError(
            "indice_precos deve ser positivo."
        )

    ff = dados_firmas["ff"]
    nf = dados_firmas["nf"]


    # ==========================================================
    # VALOR ADICIONADO E REMUNERAÇÕES
    # ==========================================================

    va_planejado_ff = float(
        ff["valor_adicionado"]
    )

    va_planejado_nf = float(
        nf["valor_adicionado"]
    )

    salarios_ff = float(
        ff["salarios"]
    )

    salarios_nf = float(
        nf["salarios"]
    )

    contribuicoes_efetivas_ff = float(
        ff["contribuicoes_efetivas"]
    )

    contribuicoes_efetivas_nf = float(
        nf["contribuicoes_efetivas"]
    )

    outros_impostos_ff = float(
        ff["outros_va"]
    )

    outros_impostos_nf = float(
        nf["outros_va"]
    )


    # ==========================================================
    # IMPOSTOS SOBRE PRODUTOS
    # ==========================================================

    impostos_produtos_total = float(
        impostos_produtos.sum()
    )

    impostos_produtos_ff = (
        p["parcela_impostos_produtos_ff"]
        * impostos_produtos_total
    )

    impostos_produtos_nf = (
        impostos_produtos_total
        - impostos_produtos_ff
    )


    # ==========================================================
    # JUROS
    # ==========================================================
    # A ordem dos vetores financeiros é:
    # famílias, governo, FF, NF e exterior.

    if len(juros_recebidos) != 5:
        raise ValueError(
            "juros_recebidos deve ter cinco setores institucionais."
        )

    if len(juros_pagos) != 5:
        raise ValueError(
            "juros_pagos deve ter cinco setores institucionais."
        )

    juros_familias_recebidos = float(
        juros_recebidos.iloc[0]
    )

    juros_familias_pagos = float(
        juros_pagos.iloc[0]
    )

    juros_ff_recebidos = float(
        juros_recebidos.iloc[2]
    )

    juros_ff_pagos = float(
        juros_pagos.iloc[2]
    )

    juros_nf_recebidos = float(
        juros_recebidos.iloc[3]
    )

    juros_nf_pagos = float(
        juros_pagos.iloc[3]
    )


    # ==========================================================
    # OUTRAS TRANSFERÊNCIAS CORRENTES
    # ==========================================================
    # Por enquanto, esta conta não possui comportamento próprio.
    # Conserva o valor real do ano-base e acompanha o nível
    # geral de preços.

    transferencias_familias = (
        float(
            outras_transferencias_base[
                "familias_recebidas"
            ]
        )
        * indice_precos
    )

    transferencias_governo = (
        float(
            outras_transferencias_base[
                "governo_recebidas"
            ]
        )
        * indice_precos
    )

    transferencias_ff = (
        float(
            outras_transferencias_base[
                "ff_pagas"
            ]
        )
        * indice_precos
    )

    transferencias_nf = (
        float(
            outras_transferencias_base[
                "nf_pagas"
            ]
        )
        * indice_precos
    )

    transferencias_exterior = (
        float(
            outras_transferencias_base[
                "exterior_pagas"
            ]
        )
        * indice_precos
    )

    transferencias_recebidas = (
        transferencias_familias
        + transferencias_governo
    )

    transferencias_pagas = (
        transferencias_ff
        + transferencias_nf
        + transferencias_exterior
    )

    if not np.isclose(
        transferencias_recebidas,
        transferencias_pagas,
        atol=1e-6,
    ):
        raise RuntimeError(
            "As outras transferências correntes não fecham: "
            f"recebidas={transferencias_recebidas}, "
            f"pagas={transferencias_pagas}."
        )

    # ==========================================================
    # DIVIDENDOS
    # ==========================================================
    # Os dividendos totais pagos são decididos pelas próprias
    # firmas. A CEI apenas distribui o fluxo entre famílias
    # e setor externo.

    base_dividendos_ff = float(
        ff["dividendos"]
    )

    base_dividendos_nf = float(
        nf["dividendos"]
    )

    dividendos_ff = base_dividendos_ff
    dividendos_nf = base_dividendos_nf

    dividendos_total = (
        dividendos_ff
        + dividendos_nf
    )

    dividendos_familias = (
        p["parcela_dividendos_familias"]
        * dividendos_total
    )

    dividendos_exterior = (
        p["parcela_dividendos_exterior"]
        * dividendos_total
    )


    # ==========================================================
    # BENEFÍCIOS
    # ==========================================================

    emprego = float(
        dados_firmas["ocupacoes"]
    )

    desempregados = (
        p["pea"]
        - emprego
    )

    beneficios = (
        p["beneficio_fixo"]
        * indice_precos
        + p["beneficio_por_desempregado"]
        * indice_salarios
        * desempregados
        / 1_000_000.0
    )


    # ==========================================================
    # APOSENTADORIAS
    # ==========================================================

    aposentadorias = (
        p["aposentadoria_por_pessoa"]
        * indice_precos
        * p["aposentados"]
    )

    aposentadorias_governo = (
        p["parcela_governo_aposentadoria"]
        * aposentadorias
    )

    aposentadorias_ff = (
        p["parcela_ff_aposentadoria"]
        * aposentadorias
    )


    # ==========================================================
    # IMPOSTO DE RENDA DAS FAMÍLIAS
    # ==========================================================
    # A base calibrada do IR familiar contém as entradas
    # primárias das linhas 1--6 da CEI.
    #
    # Outras transferências correntes não entram nessa base.

    base_ir_familias = (
        salarios_ff
        + salarios_nf
        + contribuicoes_efetivas_ff
        + contribuicoes_efetivas_nf
        + juros_familias_recebidos
    )

    ir_familias = (
        p["taxa_ir_familias"]
        * base_ir_familias
    )


    # ==========================================================
    # RENDA DISPONÍVEL DAS FAMÍLIAS
    # ==========================================================
    # Aqui, ao contrário da base do IR, entram as outras
    # transferências correntes recebidas pelas famílias.

    renda_disponivel_familias = (
        salarios_ff
        + salarios_nf
        + contribuicoes_efetivas_ff
        + contribuicoes_efetivas_nf
        + juros_familias_recebidos
        - juros_familias_pagos
        + dividendos_familias
        + beneficios
        + aposentadorias
        + transferencias_familias
        - ir_familias
    )


    # ==========================================================
    # CONSUMO E POUPANÇA DAS FAMÍLIAS
    # ==========================================================

    consumo_nominal = (
        p["propensao_consumir"]
        * renda_disponivel_familias
    )

    poupanca_familias = (
        renda_disponivel_familias
        - consumo_nominal
    )


    # ==========================================================
    # CONTRIBUIÇÕES SOCIAIS / PREVIDÊNCIA
    # ==========================================================
    # A previdência é uma alocação da poupança familiar.

    previdencia_familias = (
        p["prop_invest_prev_familias"]
        * poupanca_familias
    )

    previdencia_publica = (
        p["prop_prev_publica"]
        * previdencia_familias
    )

    previdencia_privada = (
        p["prop_prev_privada"]
        * previdencia_familias
    )


        # ==========================================================
    # IMPOSTO DE RENDA DAS FIRMAS
    # ==========================================================
    # A base reproduz as linhas da CEI usadas na calibração.
    #
    # O valor adicionado das firmas está sem impostos sobre
    # produtos. Na CEI, esses impostos entram na linha 1 e saem
    # novamente na linha 4, portanto se cancelam na base do IR.

    base_ir_ff = (
        va_planejado_ff
        - salarios_ff
        - contribuicoes_efetivas_ff
        - outros_impostos_ff
        + juros_ff_recebidos
        - juros_ff_pagos
        + previdencia_privada
        - aposentadorias_ff
        - transferencias_ff
    )

    base_ir_nf = (
        va_planejado_nf
        - salarios_nf
        - contribuicoes_efetivas_nf
        - outros_impostos_nf
        + juros_nf_recebidos
        - juros_nf_pagos
        - transferencias_nf
    )

    ir_ff = (
        p["taxa_ir_ff"]
        * base_ir_ff
    )

    ir_nf = (
        p["taxa_ir_nf"]
        * base_ir_nf
    )



    # ==========================================================
    # RETORNO
    # ==========================================================

    return {
        "va_planejado_ff": va_planejado_ff,
        "va_planejado_nf": va_planejado_nf,

        "salarios_ff": salarios_ff,
        "salarios_nf": salarios_nf,

        "contribuicoes_efetivas_ff": contribuicoes_efetivas_ff,
        "contribuicoes_efetivas_nf": contribuicoes_efetivas_nf,

        "outros_impostos_ff": outros_impostos_ff,
        "outros_impostos_nf": outros_impostos_nf,

        "impostos_produtos_ff": impostos_produtos_ff,
        "impostos_produtos_nf": impostos_produtos_nf,

        "juros_recebidos": juros_recebidos,
        "juros_pagos": juros_pagos,

        "base_ir_ff": base_ir_ff,
        "base_ir_nf": base_ir_nf,
        "ir_ff": ir_ff,
        "ir_nf": ir_nf,

        "base_dividendos_ff": base_dividendos_ff,
        "base_dividendos_nf": base_dividendos_nf,
        "dividendos_ff": dividendos_ff,
        "dividendos_nf": dividendos_nf,
        "dividendos_total": dividendos_total,
        "dividendos_familias": dividendos_familias,
        "dividendos_exterior": dividendos_exterior,

        "beneficios": beneficios,

        "aposentadorias": aposentadorias,
        "aposentadorias_governo": aposentadorias_governo,
        "aposentadorias_ff": aposentadorias_ff,

        "outras_transferencias_familias": transferencias_familias,
        "outras_transferencias_governo": transferencias_governo,
        "outras_transferencias_ff": transferencias_ff,
        "outras_transferencias_nf": transferencias_nf,
        "outras_transferencias_exterior": transferencias_exterior,

        "base_ir_familias": base_ir_familias,
        "ir_familias": ir_familias,

        "renda_disponivel_familias": renda_disponivel_familias,
        "consumo_nominal": consumo_nominal,
        "poupanca_familias": poupanca_familias,

        "previdencia_familias": previdencia_familias,
        "previdencia_publica": previdencia_publica,
        "previdencia_privada": previdencia_privada,

        "emprego": emprego,
    }