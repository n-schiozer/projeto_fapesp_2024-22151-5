"""Agregação contábil das firmas domésticas por setor."""

import pandas as pd

from contabilidade.estrutura_cei import VA
from agentes.firma import Firma


def agregar_firmas(
    firmas: dict[str, Firma],
    setores: list[str],
) -> dict:
    """Agrega exclusivamente os estados e fluxos registrados nas firmas.

    A matriz de consumo intermediário mantém fornecedores nas linhas e setores
    compradores nas colunas.

    Portanto:

        consumo_intermediario.loc[i, j]

    representa a quantidade do produto do setor i demandada pelas firmas do
    setor j.

    A soma das colunas fornece a demanda intermediária total dirigida a cada
    setor fornecedor:

        demanda_intermediaria_real[i]
            = soma_j consumo_intermediario[i, j]

    O investimento, por sua vez, é agregado inicialmente pelo setor da firma
    investidora. Sua posterior transformação em demanda por setores produtores
    de bens de capital é realizada fora desta função.
    """

    setores = list(setores)

    if len(setores) != len(set(setores)):
        raise ValueError(
            "A lista de setores possui rótulos repetidos."
        )

    # ==========================================================
    # PRODUÇÃO E VALOR ADICIONADO
    # ==========================================================

    producao_real = pd.Series(
        0.0, index=setores, name="producao_real"
    )

    producao_nominal = pd.Series(
        0.0, index=setores, name="producao_nominal"
    )

    remuneracoes = pd.Series(
        0.0, index=setores, name="remuneracoes"
    )

    salarios = pd.Series(
        0.0, index=setores, name="salarios"
    )

    contribuicoes = pd.Series(
        0.0, index=setores, name="contribuicoes"
    )

    eob_misto = pd.Series(
        0.0, index=setores, name="eob_misto"
    )

    dividendos = pd.Series(
        0.0, index=setores, name="dividendos"
    )

    outros_va = pd.Series(
        0.0, index=setores, name="outros_va"
    )

    valor_adicionado = pd.Series(
        0.0, index=setores, name="valor_adicionado"
    )

    ocupacoes = pd.Series(
        0.0, index=setores, name="ocupacoes"
    )

    # ==========================================================
    # INVESTIMENTO — POR SETOR INVESTIDOR
    # ==========================================================

    investimento_liquido = pd.Series(
        0.0,
        index=setores,
        name="investimento_liquido",
    )

    investimento_reposicao = pd.Series(
        0.0,
        index=setores,
        name="investimento_reposicao",
    )

    investimento_bruto = pd.Series(
        0.0,
        index=setores,
        name="investimento_bruto",
    )

    # ==========================================================
    # MERCADO
    # ==========================================================

    demanda_nominal_desejada = pd.Series(
        0.0, index=setores
    )

    demanda_real_recebida = pd.Series(
        0.0, index=setores
    )

    vendas_real = pd.Series(
        0.0, index=setores
    )

    vendas_nominal = pd.Series(
        0.0, index=setores
    )

    producao_nao_vendida = pd.Series(
        0.0, index=setores
    )

    demanda_nao_atendida = pd.Series(
        0.0, index=setores
    )

    market_share_desejado = pd.Series(
        0.0, index=setores
    )

    market_share_realizado = pd.Series(
        0.0, index=setores
    )

    # Linhas = fornecedores
    # Colunas = setores compradores
    consumo_intermediario = pd.DataFrame(
        0.0,
        index=setores,
        columns=setores,
    )

    # ==========================================================
    # AGREGAÇÃO DAS FIRMAS
    # ==========================================================

    for id_firma, firma in firmas.items():

        if not isinstance(firma, Firma):
            raise TypeError(
                f"{id_firma} não é uma instância de Firma."
            )

        if firma.id != id_firma:
            raise ValueError(
                f"A chave {id_firma} não coincide com firma.id."
            )

        if firma.setor not in producao_real.index:
            raise KeyError(
                f"Setor da firma ausente da agregação: {firma.setor}"
            )

        # A demanda intermediária já foi calculada pela firma no
        # bloco de decisões ex ante. O agregador apenas lê o estado.
        demanda_intermediaria = (
            firma.demanda_intermediaria_real
        )

        fornecedores_desconhecidos = (
            demanda_intermediaria.index.difference(setores)
        )

        if not fornecedores_desconhecidos.empty:
            raise KeyError(
                f"Tecnologia de {id_firma} usa setores ausentes: "
                f"{list(fornecedores_desconhecidos)}"
            )

        setor = firma.setor

        # Produção
        producao_real.at[setor] += firma.producao_real

        producao_nominal.at[setor] += (
            firma.preco_transacao
            * firma.producao_real
        )

        # Consumo intermediário:
        # cada vetor da firma entra na coluna de seu setor comprador.
        consumo_intermediario[setor] += (
            demanda_intermediaria.reindex(
                setores,
                fill_value=0.0,
            )
        )

        # Valor adicionado
        remuneracoes.at[setor] += (
            firma.calcular_remuneracoes()
        )

        salarios.at[setor] += (
            firma.calcular_salarios()
        )

        contribuicoes.at[setor] += (
            firma.calcular_contribuicoes()
        )

        eob_misto.at[setor] += (
            firma.calcular_eob_misto()
        )

        dividendos.at[setor] += (
            firma.calcular_dividendos()
        )

        outros_va.at[setor] += (
            firma.calcular_outros_va()
        )

        valor_adicionado.at[setor] += (
            firma.calcular_valor_adicionado()
        )

        ocupacoes.at[setor] += (
            firma.demanda_trabalho
        )

        # Investimento por setor investidor
        investimento_liquido.at[setor] += (
            firma.investimento_liquido
        )

        investimento_reposicao.at[setor] += (
            firma.investimento_reposicao
        )

        investimento_bruto.at[setor] += (
            firma.investimento_bruto
        )

        # Mercado
        demanda_nominal_desejada.at[setor] += (
            firma.demanda_nominal_desejada
        )

        demanda_real_recebida.at[setor] += (
            firma.demanda_recebida_real
        )

        vendas_real.at[setor] += (
            firma.vendas_real
        )

        vendas_nominal.at[setor] += (
            firma.vendas_nominal
        )

        producao_nao_vendida.at[setor] += (
            firma.producao_nao_vendida_real
        )

        demanda_nao_atendida.at[setor] += (
            firma.demanda_nao_atendida_real
        )

        market_share_desejado.at[setor] += (
            firma.market_share_desejado
        )

        market_share_realizado.at[setor] += (
            firma.market_share_realizado
        )

    # ==========================================================
    # DEMANDA INTERMEDIÁRIA POR SETOR FORNECEDOR
    # ==========================================================

    demanda_intermediaria_real = (
        consumo_intermediario
        .sum(axis="columns")
        .rename("demanda_intermediaria_real")
    )

    return {
        "producao_real": producao_real,
        "producao_nominal": producao_nominal,

        # CI completa: fornecedor × comprador
        "consumo_intermediario": consumo_intermediario,

        # CI total recebida por setor fornecedor
        "demanda_intermediaria_real": demanda_intermediaria_real,

        "remuneracoes": remuneracoes,
        "salarios": salarios,
        "contribuicoes": contribuicoes,
        "eob_misto": eob_misto,
        "dividendos":dividendos,
        "outros_va": outros_va,
        "valor_adicionado": valor_adicionado,
        "ocupacoes": ocupacoes,

        # Investimento por setor investidor
        "investimento_liquido": investimento_liquido,
        "investimento_reposicao": investimento_reposicao,
        "investimento_bruto": investimento_bruto,

        "demanda_nominal_desejada": demanda_nominal_desejada,
        "demanda_real_recebida": demanda_real_recebida,
        "vendas_real": vendas_real,
        "vendas_nominal": vendas_nominal,
        "producao_nao_vendida": producao_nao_vendida,
        "demanda_nao_atendida": demanda_nao_atendida,
        "market_share_desejado": market_share_desejado,
        "market_share_realizado": market_share_realizado,
    }

def separar_agregados_firmas_cei(
    agregados: dict,
    setores: list[str],
    razoes_va: pd.DataFrame,
    setor_financeiro: int,
) -> dict:
    """Separa os fluxos produtivos das firmas em FF e NF para a CEI.

    A decomposição interna preserva somente as participações do ano-base. Os
    totais de remuneração, EOB+misto, outros componentes, VA e ocupações são
    os que já vieram de ``agregar_firmas``; nenhuma produção é recalculada.
    """

    setores = list(setores)
    if setor_financeiro < 0 or setor_financeiro >= len(setores):
        raise IndexError("setor_financeiro está fora da lista de setores.")
    razoes_va = razoes_va.reindex(columns=setores)

    def decompor(total: pd.Series, linhas: list[str]) -> pd.DataFrame:
        pesos = razoes_va.loc[linhas]
        denominador = pesos.sum(axis="index")
        sem_base = (denominador == 0.0) & (total != 0.0)
        if sem_base.any():
            raise ValueError(
                "Falta decomposição-base do VA em "
                f"{list(total.index[sem_base])}."
            )
        pesos = pesos.div(denominador.mask(denominador == 0.0), axis="columns")
        return pesos.fillna(0.0).mul(total, axis="columns")

    remuneracoes = decompor(
        agregados["remuneracoes"],
        [
            VA["salarios"],
            VA["contribuicoes_efetivas"],
            VA["contribuicoes_imputadas"],
        ],
    )
    eob_misto = decompor(
        agregados["eob_misto"], [VA["eob"], VA["rendimento_misto"]]
    )
    outros_va = decompor(
        agregados["outros_va"], [VA["outros_impostos"], VA["outros_subsidios"]]
    )
    detalhado = pd.concat([remuneracoes, eob_misto, outros_va])
    detalhado.loc[VA["valor_adicionado"] if "valor_adicionado" in VA else VA["total"]] = (
        agregados["valor_adicionado"]
    )

    setor_ff = setores[setor_financeiro]

    def institucional(selecionados: list[str]) -> dict:
        return {
            "valor_adicionado": float(
                agregados["valor_adicionado"].loc[selecionados].sum()
            ),
            "remuneracoes": float(agregados["remuneracoes"].loc[selecionados].sum()),
            "salarios": float(detalhado.loc[VA["salarios"], selecionados].sum()),
            "contribuicoes_efetivas": float(
                detalhado.loc[VA["contribuicoes_efetivas"], selecionados].sum()
            ),
            "contribuicoes_imputadas": float(
                detalhado.loc[VA["contribuicoes_imputadas"], selecionados].sum()
            ),
            "eob": float(detalhado.loc[VA["eob"], selecionados].sum()),
            "rendimento_misto": float(
                detalhado.loc[VA["rendimento_misto"], selecionados].sum()
            ),
            "eob_mais_misto": float(agregados["eob_misto"].loc[selecionados].sum()),
            "outros_impostos": float(
                detalhado.loc[VA["outros_impostos"], selecionados].sum()
            ),
            "outros_subsidios": float(
                detalhado.loc[VA["outros_subsidios"], selecionados].sum()
            ),
            "outros_va": float(agregados["outros_va"].loc[selecionados].sum()),
        }

    setores_nf = [setor for setor in setores if setor != setor_ff]

    return {
        "ff": institucional([setor_ff]),
        "nf": institucional(setores_nf),
        "ocupacoes": float(agregados["ocupacoes"].sum()),
        "setorial": {
            "producao_real": agregados["producao_real"].copy(),
            "consumo_intermediario": agregados["consumo_intermediario"].copy(),
            "valor_adicionado": detalhado,
            "ocupacoes": agregados["ocupacoes"].copy(),
        },
    }



def agregar_resultados_realizados_firmas(
    firmas: dict,
    setores: list,
) -> dict:

    valor_producao = pd.Series(
        0.0,
        index=setores,
        name="valor_producao_realizado",
    )

    consumo_intermediario = pd.Series(
        0.0,
        index=setores,
        name="consumo_intermediario_realizado",
    )

    valor_adicionado = pd.Series(
        0.0,
        index=setores,
        name="valor_adicionado_realizado",
    )

    remuneracoes = pd.Series(
        0.0,
        index=setores,
        name="remuneracoes_realizadas",
    )

    salarios = pd.Series(
        0.0,
        index=setores,
        name="salarios_realizados",
    )

    contribuicoes = pd.Series(
        0.0,
        index=setores,
        name="contribuicoes_realizadas",
    )

    outros_va = pd.Series(
        0.0,
        index=setores,
        name="outros_va_realizados",
    )

    eob_misto = pd.Series(
        0.0,
        index=setores,
        name="eob_misto_realizado",
    )


    for firma in firmas.values():

        setor = firma.setor

        valor_producao.at[setor] += (
            firma.valor_producao_nominal_realizado
        )

        consumo_intermediario.at[setor] += (
            firma.consumo_intermediario_nominal_realizado
        )

        valor_adicionado.at[setor] += (
            firma.valor_adicionado_realizado
        )

        remuneracoes.at[setor] += (
            firma.remuneracoes_realizadas
        )

        salarios.at[setor] += (
            firma.salarios_realizados
        )

        contribuicoes.at[setor] += (
            firma.contribuicoes_realizadas
        )

        outros_va.at[setor] += (
            firma.outros_va_realizados
        )

        eob_misto.at[setor] += (
            firma.eob_misto_realizado
        )


    return {
        "valor_producao": valor_producao,
        "consumo_intermediario": consumo_intermediario,
        "valor_adicionado": valor_adicionado,
        "remuneracoes": remuneracoes,
        "salarios": salarios,
        "contribuicoes": contribuicoes,
        "outros_va": outros_va,
        "eob_misto": eob_misto,
    }