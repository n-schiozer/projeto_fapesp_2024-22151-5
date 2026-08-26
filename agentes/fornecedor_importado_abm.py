"""Fornecedor importado da versão ABM."""

from dataclasses import dataclass

import numpy as np

from agentes.firma import Firma


@dataclass
class FornecedorImportadoABM:
    """Fornecedor externo que participa dos mercados domésticos."""

    setor: str
    regime: str

    preco_base: float
    preco: float

    quantidade_ofertada_real: float

    qualidade: float = np.nan

    oferta_maxima_real: float = np.inf   

    vendas_real: float = 0.0
    vendas_nominal: float = 0.0

    

    def atualizar_preco(
        self,
        indice_cambio: float,
    ) -> None:
        """Atualiza o preço doméstico do bem importado pelo câmbio."""

        if indice_cambio <= 0.0:
            raise ValueError(
                "indice_cambio deve ser positivo."
            )

        self.preco = (
            self.preco_base
            * indice_cambio
        )


# ==========================================================
# INICIALIZAÇÃO DOS FORNECEDORES IMPORTADOS
# ==========================================================

def inicializar_importados_abm(
    condicoes_iniciais: dict,
    firmas: dict[str, Firma],
) -> dict[str, FornecedorImportadoABM]:
    """Inicializa um fornecedor importado por setor."""

    ci = condicoes_iniciais
    setores = list(ci["setores"])

    parcela_importada = (
        ci["parcela_importada"]
        .reindex(setores)
        .astype(float)
    )

    importados = {}

    importacoes_base = (
        ci["conversao_de_pm_pb"]
        @ ci["demanda_final_base"]
    ).mul(
        parcela_importada
    ).clip(
        lower=0.0
    )

    for setor in setores:

        firmas_setor = [
            firma
            for firma in firmas.values()
            if firma.setor == setor
        ]

        if not firmas_setor:
            raise ValueError(
                f"Não há firmas domésticas em {setor}."
            )

        regime = firmas_setor[0].regime

        if any(
            firma.regime != regime
            for firma in firmas_setor
        ):
            raise ValueError(
                f"Regimes inconsistentes em {setor}."
            )

        share_importado = float(
            parcela_importada.at[setor]
        )

        if not 0.0 <= share_importado < 1.0:
            raise ValueError(
                f"Parcela importada inválida em {setor}: "
                f"{share_importado}."
            )

        # ==========================================================
        # MERCADO INDUSTRIAL
        # ==========================================================

        if regime == "industrial":

            eta_preco = firmas_setor[0].eta_preco
            eta_qualidade = firmas_setor[0].eta_qualidade

            atratividade_domestica = sum(
                firma.calcular_atratividade()
                for firma in firmas_setor
            )

            if atratividade_domestica <= 0.0:
                raise ValueError(
                    f"Atratividade doméstica não positiva "
                    f"em {setor}."
                )

            atratividade_importado = (
                share_importado
                / (1.0 - share_importado)
                * atratividade_domestica
            )

            preco_base = 1.0

            qualidade = (
                (
                    atratividade_importado
                    / preco_base ** eta_preco
                )
                ** (1.0 / eta_qualidade)
                if share_importado > 0.0
                else 0.0
            )

            importados[setor] = (
                FornecedorImportadoABM(
                    setor=setor,
                    regime=regime,
                    preco_base=preco_base,
                    preco=preco_base,
                    quantidade_ofertada_real=np.nan,
                    qualidade=float(qualidade),
                )
            )

        # ==========================================================
        # MERCADO DE LEILÃO
        # ==========================================================

        elif regime == "leilao":

            preco_base = 1.0

            quantidade_ofertada_real = float(
                importacoes_base.at[setor]
            )

            importados[setor] = (
                FornecedorImportadoABM(
                    setor=setor,
                    regime=regime,
                    preco_base=preco_base,
                    preco=preco_base,
                    quantidade_ofertada_real=quantidade_ofertada_real,
                )
            )

        else:
            raise ValueError(
                f"Regime desconhecido em {setor}: {regime}."
            )

    return importados