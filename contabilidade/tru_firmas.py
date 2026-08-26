"""Adaptador do bloco produtivo da TRU para os objetos ``Firma``.

Este módulo não resolve demanda, preços ou mercado. Ele apenas reorganiza os
agregados correntes das firmas no formato mínimo da TRU utilizado pela versão
ABM. As linhas desagregadas do valor adicionado preservam, explicitamente, a
decomposição proporcional observada no ano-base.
"""

import numpy as np
import pandas as pd

from agentes.agregar_firmas import agregar_firmas
from contabilidade.estrutura_cei import VA


def _distribuir_composicao_base(
    total: pd.Series,
    razoes_base: pd.DataFrame,
    linhas: list[str],
) -> pd.DataFrame:
    """Distribui um agregado atual segundo as proporções base de suas linhas."""

    pesos = razoes_base.loc[linhas].copy()
    denominador = pesos.sum(axis="index")
    if ((denominador == 0.0) & (total != 0.0)).any():
        setores = list(total.index[(denominador == 0.0) & (total != 0.0)])
        raise ValueError(
            "Não há decomposição-base para componentes positivos do VA em: "
            f"{setores}."
        )
    pesos = pesos.div(denominador.replace(0.0, np.nan), axis="columns").fillna(0.0)
    return pesos.mul(total, axis="columns")


def construir_tru_a_partir_firmas(
    firmas: dict,
    setores: list[str],
    condicoes_iniciais: dict,
) -> dict:
    """Reconstrói o bloco produtivo corrente da TRU a partir das firmas.

    ``agregar_firmas`` é a única fonte dos agregados correntes. A decomposição
    interna de remunerações, EOB/rendimento misto e outros impostos/subsídios
    somente preserva os pesos do ano-base: ela não cria comportamento novo.
    """

    setores = list(setores)
    agregados = agregar_firmas(firmas, setores)
    razoes_va = condicoes_iniciais["razoes_va"].reindex(columns=setores)
    quadro_va = pd.DataFrame(0.0, index=razoes_va.index, columns=setores)

    # Remunerações: salários e cada contribuição mantêm sua parcela-base.
    linhas_remuneracao = [
        VA["salarios"],
        VA["previdencia_oficial"],
        VA["previdencia_privada"],
        VA["contribuicoes_imputadas"],
    ]
    componentes_remuneracao = _distribuir_composicao_base(
        agregados["remuneracoes"], razoes_va, linhas_remuneracao
    )
    quadro_va.loc[linhas_remuneracao] = componentes_remuneracao
    quadro_va.loc[VA["contribuicoes_efetivas"]] = (
        quadro_va.loc[VA["previdencia_oficial"]]
        + quadro_va.loc[VA["previdencia_privada"]]
    )
    quadro_va.loc[VA["remuneracoes"]] = agregados["remuneracoes"]

    # EOB e rendimento misto preservam a composição observada no ano-base.
    linhas_eob = [VA["rendimento_misto"], VA["eob"]]
    quadro_va.loc[linhas_eob] = _distribuir_composicao_base(
        agregados["eob_misto"], razoes_va, linhas_eob
    )
    quadro_va.loc[VA["eob_mais_misto"]] = agregados["eob_misto"]

    # O resíduo do VA das firmas é explicitamente decomposto entre impostos e
    # subsídios sobre a produção pelas participações do ano-base.
    linhas_outros = [VA["outros_impostos"], VA["outros_subsidios"]]
    quadro_va.loc[linhas_outros] = _distribuir_composicao_base(
        agregados["outros_va"], razoes_va, linhas_outros
    )
    quadro_va.loc[VA["total"]] = agregados["valor_adicionado"]
    quadro_va.loc[VA["producao"]] = agregados["producao_nominal"]
    quadro_va.loc[VA["ocupacoes"]] = agregados["ocupacoes"]

    # Nesta etapa, os objetos já carregam quantidades e valores ao preço
    # vigente. A mesma estrutura é exposta como real e nominal para manter a
    # compatibilidade mínima com a TRU legada até a Etapa 7.
    real = {
        "producao_domestica": agregados["producao_real"].copy(),
        "consumo_intermediario": agregados["consumo_intermediario"].copy(),
        "valor_adicionado": quadro_va.copy(),
        "ocupacoes": agregados["ocupacoes"].copy(),
    }
    nominal = {
        "producao_domestica": agregados["producao_nominal"].copy(),
        "consumo_intermediario": agregados["consumo_intermediario"].copy(),
        "valor_adicionado": quadro_va.copy(),
        "ocupacoes": agregados["ocupacoes"].copy(),
    }
    return {"real": real, "nominal": nominal, "agregados_firmas": agregados}
