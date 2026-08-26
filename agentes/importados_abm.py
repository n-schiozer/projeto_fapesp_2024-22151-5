"""Player importado calibrado no ano-base da TRU."""

from dataclasses import dataclass

import numpy as np

from agentes.firma import Firma


@dataclass
class FornecedorImportado:
    """Concorrente externo sem produção ou fluxos domésticos próprios."""

    setor: str
    preco_importado_base: float
    preco_importado: float
    qualidade_importado: float
    atratividade_importado: float
    market_share_importado_base: float
    market_share_importado: float
    fator_atendimento: float = 1.0
    market_share_desejado: float = 0.0
    market_share_realizado: float = 0.0
    demanda_nominal_desejada: float = 0.0
    demanda_real_desejada: float = 0.0
    vendas_real: float = 0.0
    vendas_nominal: float = 0.0
    custo_unitario_importado_base: float = np.nan
    custo_unitario_importado: float = np.nan
    markup_importado: float = np.nan
    preco_oferta_importado: float = np.nan
    oferta_maxima_importado_real: float = 0.0
    quantidade_importada_real: float = 0.0
    valor_importado_nominal: float = 0.0
    demanda_nao_atendida_real: float = 0.0


def inicializar_importados(
    condicoes_iniciais: dict,
    firmas: dict[str, Firma],
    multiplicador_capacidade_importada: float = 1.5,
) -> dict[str, FornecedorImportado]:
    """Cria e calibra os importados contra as firmas industriais no ano-base.

    A participação observada da TRU é ``parcela_importada``. A qualidade do
    importado é a incógnita que faz o multilogit reproduzir essa participação,
    sem ainda executar qualquer mercado ou dinâmica cambial.
    """

    if multiplicador_capacidade_importada < 0.0:
        raise ValueError("multiplicador_capacidade_importada não pode ser negativo.")

    setores = list(condicoes_iniciais["setores"])
    market_share_base = condicoes_iniciais["parcela_importada"].reindex(setores)
    importacoes_reais_base = (
        condicoes_iniciais["conversao_de_pm_pb"]
        @ condicoes_iniciais["demanda_final_base"]
    ).mul(market_share_base).clip(lower=0.0)
    importados: dict[str, FornecedorImportado] = {}

    for setor in setores:
        firmas_setor = [firma for firma in firmas.values() if firma.setor == setor]
        if not firmas_setor:
            raise ValueError(f"Não há firma doméstica calibrada em {setor}.")

        share_base = float(market_share_base.at[setor])
        if not 0.0 <= share_base < 1.0:
            raise ValueError(
                f"Market share importado fora de [0, 1) em {setor}: {share_base}."
            )

        # Leilões não usam multilogit. O custo-sombra externo é calibrado uma
        # única vez pela estrutura doméstica-base e depois só acompanha câmbio.
        if firmas_setor[0].regime == "leilao":
            producao_base = np.asarray(
                [firma.producao_base_real for firma in firmas_setor], dtype=float
            )
            if producao_base.sum() <= 0.0:
                raise ValueError(f"Produção-base não positiva em {setor}.")
            custos_base = np.asarray(
                [
                    firma.custo_intermediario_unitario_base
                    + firma.remuneracao_unitaria_base
                    + firma.outros_va_unitario_base
                    for firma in firmas_setor
                ],
                dtype=float,
            )
            markups = np.asarray(
                [firma.markup_leilao_base for firma in firmas_setor], dtype=float
            )
            custo_base = float(np.average(custos_base, weights=producao_base))
            markup = float(np.average(markups, weights=producao_base))
            preco_oferta_base = (1.0 + markup) * custo_base
            importados[setor] = FornecedorImportado(
                setor=setor,
                preco_importado_base=preco_oferta_base,
                preco_importado=preco_oferta_base,
                qualidade_importado=np.nan,
                atratividade_importado=np.nan,
                market_share_importado_base=share_base,
                market_share_importado=share_base,
                custo_unitario_importado_base=custo_base,
                custo_unitario_importado=custo_base,
                markup_importado=markup,
                preco_oferta_importado=preco_oferta_base,
                oferta_maxima_importado_real=float(
                    multiplicador_capacidade_importada
                    * importacoes_reais_base.at[setor]
                ),
            )
            continue

        eta_preco = firmas_setor[0].eta_preco
        eta_qualidade = firmas_setor[0].eta_qualidade
        if any(
            firma.regime != "industrial"
            or firma.eta_preco != eta_preco
            or firma.eta_qualidade != eta_qualidade
            for firma in firmas_setor
        ):
            raise ValueError(f"Parâmetros domésticos inconsistentes em {setor}.")

        atratividade_domestica = sum(
            firma.calcular_atratividade() for firma in firmas_setor
        )
        if atratividade_domestica <= 0.0:
            raise ValueError(f"Atratividade doméstica não positiva em {setor}.")

        atratividade_importado = (
            share_base / (1.0 - share_base) * atratividade_domestica
            if share_base > 0.0
            else 0.0
        )
        qualidade_importado = float(
            (atratividade_importado / 1.0**eta_preco) ** (1.0 / eta_qualidade)
        )
        share_calculado = atratividade_importado / (
            atratividade_domestica + atratividade_importado
        )

        importados[setor] = FornecedorImportado(
            setor=setor,
            preco_importado_base=1.0,
            preco_importado=1.0,
            qualidade_importado=qualidade_importado,
            atratividade_importado=atratividade_importado,
            market_share_importado_base=share_base,
            market_share_importado=float(share_calculado),
        )

    return importados
