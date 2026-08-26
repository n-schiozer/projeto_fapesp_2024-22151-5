"""Compatibilidade temporária para simuladores anteriores ao laboratório atual.

Este módulo preserva o contrato público antigo enquanto os outros simuladores
versionados ainda não passam pela etapa de unificação de resultados. O
laboratório principal não deve importar este módulo.
"""

from __future__ import annotations

import pandas as pd


def inicializar_resultados_abm(base: dict) -> dict:
    """Cria os contêineres públicos do contrato anterior."""

    return {
        "firmas": base["firmas"],
        "importados": base["importados"],
        "ids_firmas_por_periodo": {},
        "agregados_firmas": {},
        "dados_firmas_cei": {},
        "mercados_industriais": {},
        "mercados_leilao": {},
        "diagnostico_importacoes": {},
        "fluxos_transitorios": {},
        "inicializacao_investimento_nf": base["inicializacao_investimento_nf"],
        "historico": [base["historico_zero"]],
        "precos_comprador": {0: base["pc_zero"]},
        "precos_basicos": {0: base["pb_zero"]},
        "precos_importacoes": {0: base["pm_zero"]},
        "precos_comprador_esperados": {0: base["pc_esperado_zero"]},
        "inflacao_precos_setorial": {0: base["inflacao_pc_zero"]},
        "cei": {0: base["cei_zero"]},
        "capacidade_financiamento": {0: base["capacidade_zero"]},
        "ativos_financeiros": {0: base["ativos_zero"].copy()},
        "passivos_financeiros": {0: base["passivos_zero"].copy()},
        "estoque_financeiro": {0: base["estoque_financeiro_zero"].copy()},
        "aquisicao_ativos_financeiros": {0: base["aquisicao_ativos_zero"]},
        "emissao_passivos_financeiros": {0: base["emissao_passivos_zero"]},
        "juros_liquidos": {0: base["juros_liquidos_zero"].copy()},
        "juros_recebidos": {0: base["juros_recebidos_zero"].copy()},
        "juros_pagos": {0: base["juros_pagos_zero"].copy()},
        "reavaliacao_financeira": {0: base["reavaliacao_zero"]},
        "investimento_nf_real": {0: base["investimento_nf_real_zero"].copy()},
        "investimento_nf_nominal": {0: base["investimento_nf_nominal_zero"].copy()},
        "fbcf_fixa_nominal": {0: base["fbcf_fixa_zero"].copy()},
        "variacao_estoques_real": {0: base["estoques_zero"].copy()},
        "variacao_estoques_nominal": {0: base["estoques_zero"].copy()},
        "variacao_autonoma_estoques_real": {0: base["estoques_zero"].copy()},
        "variacao_ciclica_estoques_real": {0: base["estoques_ciclicos_zero"]},
        "estoque_real": {0: base["estoque_real_zero"].copy()},
        "estoque_referencia_real": {0: base["estoque_referencia_zero"].copy()},
        "estoque_ciclico_real": {0: base["estoque_ciclico_zero"].copy()},
        "investimento_nf_por_setor_investidor": {
            0: base["investimento_nf_investidor_zero"].copy()
        },
        "estoque_capital_nf_real": {0: base["capital_nf_zero"].copy()},
    }


def armazenar_resultados_periodo(resultados: dict, t: int, dados: dict) -> None:
    """Guarda uma rodada conforme o contrato anterior."""

    resultados["agregados_firmas"][t] = dados["agregados_firmas"]
    resultados["dados_firmas_cei"][t] = dados["dados_firmas_cei"]
    resultados["ids_firmas_por_periodo"][t] = dados["ids_firmas"]
    resultados["fluxos_transitorios"][t] = {
        nome: valor.copy() if hasattr(valor, "copy") else valor
        for nome, valor in dados["fluxos_transitorios"].items()
    }
    resultados["mercados_industriais"][t] = dados["mercado_industrial"]
    resultados["mercados_leilao"][t] = dados["mercado_leilao"]
    resultados["diagnostico_importacoes"][t] = dados["diagnostico_importacoes"]
    resultados["historico"].append(dados["historico"])

    for nome, valor in {
        "precos_comprador": dados["pc"],
        "precos_basicos": dados["pb"],
        "precos_importacoes": dados["pm"],
        "precos_comprador_esperados": dados["pc_esperado"],
        "inflacao_precos_setorial": dados["inflacao_pc_setorial"],
        "ativos_financeiros": dados["ativos_financeiros"],
        "passivos_financeiros": dados["passivos_financeiros"],
        "estoque_financeiro": dados["estoque_financeiro"],
        "aquisicao_ativos_financeiros": dados["aquisicao_ativos"],
        "emissao_passivos_financeiros": dados["emissao_passivos"],
        "juros_liquidos": dados["juros_liquidos"],
        "juros_recebidos": dados["juros_recebidos"],
        "juros_pagos": dados["juros_pagos"],
        "reavaliacao_financeira": dados["reavaliacao_financeira"],
        "investimento_nf_real": dados["investimento_nf_real"],
        "investimento_nf_nominal": dados["investimento_nf_nominal"],
        "fbcf_fixa_nominal": dados["fbcf_fixa_nominal"],
        "variacao_estoques_real": dados["variacao_estoques_real"],
        "variacao_estoques_nominal": dados["variacao_estoques_nominal"],
        "variacao_autonoma_estoques_real": dados["variacao_autonoma_estoques"],
        "variacao_ciclica_estoques_real": dados["variacao_ciclica_estoques"],
        "estoque_real": dados["estoque_real"],
        "estoque_referencia_real": dados["estoque_referencia"],
        "estoque_ciclico_real": dados["estoque_ciclico"],
        "investimento_nf_por_setor_investidor": dados["investimento_nf_investidor"],
        "estoque_capital_nf_real": dados["estoque_capital_nf"],
    }.items():
        resultados[nome][t] = valor.copy()
    resultados["cei"][t] = dados["cei"].copy(deep=True)
    resultados["capacidade_financiamento"][t] = dados["capacidade"].copy()


def finalizar_resultados(resultados: dict, setores_com_estoques: pd.Series) -> dict:
    """Converte os contêineres do contrato anterior em sua API pública."""

    resultado = resultados.copy()
    resultado["historico"] = pd.DataFrame(resultado["historico"]).set_index("periodo")
    for nome in (
        "precos_comprador", "precos_basicos", "precos_importacoes",
        "precos_comprador_esperados", "inflacao_precos_setorial",
        "ativos_financeiros", "passivos_financeiros", "estoque_financeiro",
        "aquisicao_ativos_financeiros", "emissao_passivos_financeiros",
        "juros_liquidos", "juros_recebidos", "juros_pagos",
        "reavaliacao_financeira",
    ):
        resultado[nome] = pd.DataFrame(resultado[nome]).T
    resultado["setores_com_estoques"] = setores_com_estoques
    return resultado
