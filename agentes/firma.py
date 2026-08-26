"""Agente firma do modelo SFC-IO-ABM.

O objeto guarda o estado econômico corrente da firma. DataFrames devem ser
construídos a partir desses objetos apenas para agregação, histórico e análise.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Firma:
    """Firma doméstica com tecnologia, custos e estados produtivos próprios."""

    # As variáveis do ano-base são condições iniciais observadas ou calibradas
    # e precisam ser fornecidas como input na criação da firma. Elas permanecem como referência ao longo da simulação. 

    # Já os atributos com field(init=False) são criados automaticamente em __post_init__ e atualizados ao longo do tempo.

    # Identificação
    id: str
    setor: str
    regime: str

    # Mercado
    eta_preco: float
    eta_qualidade: float
    eta_atendimento: float = field(default=1.0, kw_only=True)
    share_domestico_inicial: float
    share_total_inicial: float
    preco_relativo: float
    qualidade: float

    # Tecnologia
    tecnologia: pd.Series
    exposicao_climatica: float = field(kw_only=True)
    peso_relativo_ci: float = field(default=1.0, kw_only=True)

    coeficientes_demanda_intermediaria: pd.Series

    # Demanda inicial
    demanda_esperada: float
    demanda_realizada: float

    # Componentes do valor adicionado por unidade de produção
    remuneracao_unitaria_base: float
    salario_unitaria_base: float
    contribuicao_unitaria_base: float
    eob_misto_unitario_base: float
    outros_va_unitario_base: float
    ocupacoes_unitario_base: float
    custo_intermediario_unitario_base: float

    # Regra de markup industrial
    parametro_markup: float
    markup_min: float
    markup_max: float
    epsilon_market_share: float

    # Leilão
    preco_minimo_leilao_base: float

    # Estoques
    forma_estoque: bool
    estoque: float

    # Capital no ano-base
    estoque_capital_inicial_real: float
    estoque_capital_real: float

    # Produção do ano-base
    producao_base_real: float
    producao_vendida_base_real: float
    producao_anterior: float


    # ==========================================================
    # ESTADOS CORRENTES DA FIRMA
    # ==========================================================

    # Produção e mercado
    expectativa_vendas_real: float = field(init=False)
    producao_desejada_real: float = field(init=False)
    producao_planejada_real: float = field(init=False)
    producao_real: float = field(init=False)

    demanda_recebida_real: float = field(init=False)
    vendas_real: float = field(init=False)
    demanda_nao_atendida_real: float = field(init=False)
    producao_nao_vendida_real: float = field(init=False)
    excedente_nao_estocavel_real: float = field(init=False)

    variacao_estoque_real: float = field(init=False)
   

    # Capital e investimento
    investimento_liquido: float = field(init=False)
    investimento_reposicao: float = field(init=False)
    investimento_bruto: float = field(init=False)

    # Capacidade produtiva
    # A produtividade normal e a utilização normal são primitivas distintas:
    # a primeira converte capital em produção normal; a segunda transforma
    # produção normal em capacidade física máxima.
    produtividade_capital_normal: float = field(init=False)
    produtividade_capital_capacidade: float = field(init=False)
    produtividade_capital_capacidade_efetiva: float = field(init=False)

    fator_produtividade_climatica: float = field(init=False)

    producao_normal_real: float = field(init=False)
    capacidade_produtiva_estrutural_real: float = field(init=False)
    capacidade_produtiva_real: float = field(init=False)


    # Choque idiossincrático de produtividade de cada firma.
    desvio_produtividade_idiossincratica: float = field(init=False)
    variancia_produtividade_idiossincratica: float = field(init=False)
    fator_produtividade_idiossincratica: float = field(init=False)

    # Capacidade de produção: 
    taxa_utilizacao_capacidade: float = field(init=False)
    utilizacao_demanda_base: float = field(init=False)
    producao_restringida_capacidade_real: float = field(init=False)

    # Diagnóstico da decisão de investimento industrial.
    capital_desejado: float = field(init=False)
    gap_capital: float = field(init=False)

    # Rentabilidade observada
    preco_capital_observado: float = field(init=False)
    taxa_retorno_bruta_observada: float = field(init=False)
    taxa_retorno_observada: float = field(init=False)

    taxa_retorno_parametro: float = field(init=False)
    taxa_retorno_ajustada: float = field(init=False)
    taxa_retorno_ajustada_anterior: float = field(init=False)

    # Utilização normal (u*) que transforma produção normal em capacidade.
    # ``utilizacao_capacidade_inicial`` é mantida apenas como alias de
    # compatibilidade para chamadas antigas.
    utilizacao_capacidade_normal: float | None = field(
        default=None,
        kw_only=True,
    )
    utilizacao_capacidade_inicial: float | None = field(default=None, kw_only=True)

    # Metadados opcionais da Demografia das Empresas. Eles descrevem a coorte
    # representada, sem alterar a dinâmica econômica da firma nesta etapa.
    numero_firmas_representadas: int = field(default=1, kw_only=True)
    pessoal_ocupado_demografia: float = field(default=np.nan, kw_only=True)
    pessoal_ocupado_firma: float = field(default=np.nan, kw_only=True)
    pessoal_ocupado_medio_por_firma: float = field(default=np.nan, kw_only=True)
    pessoal_ocupado_minimo_original: float = field(default=np.nan, kw_only=True)
    pessoal_ocupado_maximo_original: float = field(default=np.nan, kw_only=True)
    multiplicador_tru: float = field(default=1.0, kw_only=True)
    peso_variedade: float = field(default=1.0, kw_only=True)
    faixa_pessoal: str = field(default="", kw_only=True)
    faixa_inicial: str = field(default="", kw_only=True)
    faixa_final: str = field(default="", kw_only=True)
    tipo_agente: str = field(default="firma", kw_only=True)

    # Trabalho
    demanda_trabalho: float = field(init=False)

    # Preços e custos
    preco_firma: float = field(init=False)
    preco_transacao: float = field(init=False)

    demanda_intermediaria_real: pd.Series = field(init=False) 

    custo_intermediario_unitario: float = field(init=False)
    custo_intermediario_unitario_esperado: float = field(init=False)
    custo_intermediario_unitario_realizado: float = field(init=False)

    remuneracao_unitaria: float = field(init=False)
    salario_unitaria: float = field(init=False)
    contribuicao_unitaria: float = field(init=False)
    eob_misto_unitario: float = field(init=False)
    outros_va_unitario: float = field(init=False)
    custo_unitario: float = field(init=False)

    # Resultado
    eob_misto_recorrente_esperado: float = field(init=False)
    eob_misto_realizado: float = field(init=False)
    eob_misto_realizado_anterior: float = field(init=False)
    custo_unitario_anterior: float = field(init=False)
    parametro_dividendos: float

    # Markup
    markup_base: float = field(init=False)
    markup: float = field(init=False)
    markup_leilao_base: float = field(init=False)
    markup_leilao: float = field(init=False)

    # Mercado
    market_share_t_1: float = field(init=False)
    market_share_t_2: float = field(init=False)
    preco_oferta_leilao: float = field(init=False)
    fator_atendimento: float = field(init=False)
    taxa_atendimento: float = field(init=False)
    market_share_desejado: float = field(init=False)
    market_share_realizado: float = field(init=False)
    demanda_nominal_desejada: float = field(init=False)
    vendas_nominal: float = field(init=False)
    qualidade_base: float = field(init=False)
    desvio_qualidade: float = field(init=False)
    variancia_qualidade: float = field(init=False)


    # Resultado realizado
    valor_producao_nominal_realizado: float = field(init=False)
    consumo_intermediario_nominal_realizado: float = field(init=False)
    valor_adicionado_realizado: float = field(init=False)

    remuneracoes_realizadas: float = field(init=False)
    salarios_realizados: float = field(init=False)
    contribuicoes_realizadas: float = field(init=False)
    outros_va_realizados: float = field(init=False)
        

    def __post_init__(self) -> None:
        """Valida parâmetros e inicializa os estados da firma no ano-base."""

        # ==========================================================
        # VALIDAÇÕES
        # ==========================================================

        if self.regime not in {"industrial", "leilao"}:
            raise ValueError("regime deve ser 'industrial' ou 'leilao'.")

        if self.eta_preco >= 0.0:
            raise ValueError("eta_preco deve ser negativa.")

        if self.eta_qualidade <= 0.0:
            raise ValueError("eta_qualidade deve ser positiva.")

        if self.eta_atendimento <= 0.0:
            raise ValueError("eta_atendimento deve ser positiva.")

        if self.preco_relativo <= 0.0:
            raise ValueError("preco_relativo deve ser positivo.")

        if self.markup_min <= -1.0 or self.markup_max < self.markup_min:
            raise ValueError("Limites de markup inválidos.")

        if self.epsilon_market_share <= 0.0:
            raise ValueError("epsilon_market_share deve ser positivo.")

        if self.share_domestico_inicial < 0.0 or self.share_total_inicial < 0.0:
            raise ValueError("Market shares não podem ser negativos.")

        if self.producao_base_real < 0.0:
            raise ValueError("producao_base_real não pode ser negativa.")

        if self.producao_vendida_base_real < 0.0:
            raise ValueError(
                "producao_vendida_base_real não pode ser negativa."
            )

        if self.estoque_capital_inicial_real < 0.0:
            raise ValueError(
                "estoque_capital_inicial_real não pode ser negativo."
            )

        self.exposicao_climatica = float(self.exposicao_climatica)
        if not 0.0 <= self.exposicao_climatica <= 1.0:
            raise ValueError(
                "exposicao_climatica deve estar no intervalo [0, 1]."
            )
        self.peso_relativo_ci = float(self.peso_relativo_ci)
        if self.peso_relativo_ci <= 0.0:
            raise ValueError("peso_relativo_ci deve ser positivo.")

        if not isinstance(self.tecnologia, pd.Series):
            self.tecnologia = pd.Series(self.tecnologia, dtype=float)
        else:
            self.tecnologia = self.tecnologia.astype(float).copy()
        
        if not isinstance(self.coeficientes_demanda_intermediaria, pd.Series):
            self.coeficientes_demanda_intermediaria = pd.Series(self.coeficientes_demanda_intermediaria, dtype=float)
        else:
            self.coeficientes_demanda_intermediaria = self.coeficientes_demanda_intermediaria.astype(float).copy()

        if self.tecnologia.index.has_duplicates:
            raise ValueError(
                "A tecnologia possui setores fornecedores repetidos."
            )

        if self.tecnologia.isna().any() or (self.tecnologia < 0.0).any():
            raise ValueError(
                "A tecnologia deve conter coeficientes não negativos."
            )

        if self.utilizacao_capacidade_normal is None:
            self.utilizacao_capacidade_normal = (
                self.utilizacao_capacidade_inicial
                if self.utilizacao_capacidade_inicial is not None
                else 0.85
            )
        # Quando ambos os nomes aparecem (por exemplo, em uma sessão de
        # notebook que ainda conserva configuração antiga), o parâmetro novo
        # é deliberadamente soberano. O nome antigo é somente compatibilidade
        # e não deve impedir a execução nem reintroduzir vínculo com preço.

        self.utilizacao_capacidade_normal = float(
            self.utilizacao_capacidade_normal
        )
        # Alias transitório para código externo que ainda lê o nome antigo.
        self.utilizacao_capacidade_inicial = (
            self.utilizacao_capacidade_normal
        )

        if not 0.0 < self.utilizacao_capacidade_normal <= 1.0:
            raise ValueError(
                "utilizacao_capacidade_normal deve estar no intervalo (0, 1]."
            )

        if self.numero_firmas_representadas < 1:
            raise ValueError("numero_firmas_representadas deve ser positivo.")
        if self.peso_variedade <= 0.0:
            raise ValueError("peso_variedade deve ser positivo.")


        # ==========================================================
        # PRODUÇÃO E DEMANDA NO ANO-BASE
        # ==========================================================

        self.expectativa_vendas_real = self.producao_vendida_base_real

        # A decisão descentralizada coincide com a produção no período-base.
        # Nos períodos seguintes, ela é preservada mesmo quando um regulador
        # altera a quantidade planejada que será fisicamente realizada.
        self.producao_desejada_real = self.producao_base_real
        self.producao_planejada_real = self.producao_base_real
        self.producao_real = self.producao_base_real

        self.demanda_intermediaria_real = pd.Series(
            0.0,
            index=self.tecnologia.index,
            name=self.id,
        )

        self.variacao_estoque_real = 0.0

        self.demanda_recebida_real = self.producao_vendida_base_real
        self.vendas_real = self.producao_vendida_base_real

        self.demanda_nao_atendida_real = 0.0

        self.producao_nao_vendida_real = max(
            0.0,
            self.producao_real - self.vendas_real,
        )

        self.excedente_nao_estocavel_real = 0.0

        # O estado estocástico começa neutro no ano-base. A variância
        # acumulada permite preservar média unitária nos multiplicadores.
        self.desvio_produtividade_idiossincratica = 0.0
        self.variancia_produtividade_idiossincratica = 0.0
        self.fator_produtividade_idiossincratica = 1.0


        self.preco_capital_observado = np.nan
        self.taxa_retorno_bruta_observada = np.nan
        self.taxa_retorno_observada = np.nan

        self.taxa_retorno_parametro = np.nan
        self.taxa_retorno_ajustada = np.nan
        self.taxa_retorno_ajustada_anterior = np.nan

        if self.qualidade <= 0.0:
            raise ValueError("qualidade deve ser positiva.")

        self.qualidade_base = float(self.qualidade)
        self.desvio_qualidade = 0.0
        self.variancia_qualidade = 0.0

        # ==========================================================
        # CAPITAL E INVESTIMENTO
        # ==========================================================

        # Ainda não há decisão de investimento no período 0.
        self.investimento_liquido = 0.0
        self.investimento_reposicao = 0.0
        self.investimento_bruto = 0.0



        if (
            self.estoque_capital_inicial_real > 0.0
            and self.producao_base_real > 0.0
        ):
            # Produtividade normal do capital: Y0 / K0. A utilização normal
            # não é embutida nesta relação tecnológica.
            self.produtividade_capital_normal = (
                self.producao_base_real
                / self.estoque_capital_inicial_real
            )
            self.producao_normal_real = self.producao_base_real

            # Produtividade apenas derivada, útil para diagnóstico de
            # capacidade máxima. Não é um parâmetro de calibração primário.
            self.produtividade_capital_capacidade = (
                self.produtividade_capital_normal
                / self.utilizacao_capacidade_normal
            )

            # No ano-base não existe choque climático.
            self.fator_produtividade_climatica = 1.0

            self.produtividade_capital_capacidade_efetiva = (
                self.produtividade_capital_capacidade
            )

            self.capacidade_produtiva_estrutural_real = (
                self.producao_normal_real
                / self.utilizacao_capacidade_normal
            )

            self.capacidade_produtiva_real = (
                self.capacidade_produtiva_estrutural_real
            )

            self.taxa_utilizacao_capacidade = (
                self.producao_base_real
                / self.capacidade_produtiva_real
            )
            self.utilizacao_demanda_base = (
                self.utilizacao_capacidade_normal
                * self.producao_vendida_base_real
                / self.producao_normal_real
            )

        else:
            self.produtividade_capital_normal = np.nan
            self.produtividade_capital_capacidade = np.nan
            self.produtividade_capital_capacidade_efetiva = np.nan

            self.fator_produtividade_climatica = 1.0

            self.producao_normal_real = np.nan
            self.capacidade_produtiva_estrutural_real = np.inf
            self.capacidade_produtiva_real = np.inf

            self.taxa_utilizacao_capacidade = np.nan
            self.utilizacao_demanda_base = np.nan

        self.producao_restringida_capacidade_real = 0.0
        self.capital_desejado = self.estoque_capital_real
        self.gap_capital = 0.0




        # ==========================================================
        # DEMANDA POR TRABALHO
        # ==========================================================

        self.demanda_trabalho = 0.0

        # ==========================================================
        # ESTADOS DE MERCADO
        # ==========================================================

        self.fator_atendimento = 1.0
        self.taxa_atendimento = 1.0

        self.market_share_desejado = 0.0
        self.market_share_realizado = 0.0

        self.demanda_nominal_desejada = 0.0

        # ==========================================================
        # PREÇOS E CUSTOS NO ANO-BASE
        # ==========================================================

        self.preco_firma = self.preco_relativo
        self.preco_transacao = self.preco_relativo

        self.vendas_nominal = (
            self.vendas_real * self.preco_transacao
        )

        self.custo_intermediario_unitario = (
            self.custo_intermediario_unitario_base
        )

        self.custo_intermediario_unitario_esperado = (
            self.custo_intermediario_unitario_base
        )

        self.custo_intermediario_unitario_realizado = (
            self.custo_intermediario_unitario_base
        )

        self.remuneracao_unitaria = self.remuneracao_unitaria_base
        self.salario_unitaria = self.salario_unitaria_base
        self.contribuicao_unitaria = self.contribuicao_unitaria_base
        self.eob_misto_unitario = self.eob_misto_unitario_base
        self.outros_va_unitario = self.outros_va_unitario_base

        self.custo_unitario = (
            self.custo_intermediario_unitario
            + self.remuneracao_unitaria
            + self.outros_va_unitario
        )

        # ==========================================================
        # RESULTADO NO ANO-BASE
        # ==========================================================

        self.custo_unitario_anterior = self.custo_unitario

        self.eob_misto_realizado = (
            self.eob_misto_unitario_base
            * self.producao_base_real
        )

        self.eob_misto_realizado_anterior = (
            self.eob_misto_realizado
        )

        self.eob_misto_recorrente_esperado = (
            self.eob_misto_realizado
        )

        if self.custo_unitario <= 0.0:
            raise ValueError(
                "custo_unitario_base deve ser positivo."
            )

        # ==========================================================
        # MARKUP
        # ==========================================================

        self.markup_base = (
            self.eob_misto_unitario_base
            / self.custo_unitario
        )

        self.markup = float(
            np.clip(
                self.markup_base,
                self.markup_min,
                self.markup_max,
            )
        )

        if not np.isclose(self.markup, self.markup_base):
            raise ValueError(
                "Os limites de markup não comportam o ano-base."
            )

        self.market_share_t_1 = self.share_total_inicial
        self.market_share_t_2 = self.share_total_inicial

        self.preco_oferta_leilao = np.nan

        self.markup_leilao_base = (
            self.markup_base
            if self.regime == "leilao"
            else np.nan
        )

        self.markup_leilao = self.markup_leilao_base

    def calcular_demanda_esperada(self, beta: float) -> None:
        """Atualiza a expectativa de demanda de forma adaptativa."""

        if not 0.0 <= beta <= 1.0:
            raise ValueError(
                "velocidade_ajuste_expectativa_demanda deve estar em [0, 1]."
            )

        self.demanda_esperada = (
            self.demanda_esperada
            + beta
            * (self.demanda_realizada - self.demanda_esperada)
        )

    def atualizar_capacidade_produtiva(
        self,
        fator_produtividade_climatica: float = 1.0,
    ) -> None:
        """Atualiza a capacidade estrutural e a capacidade disponível."""

        if not 0.0 <= fator_produtividade_climatica <= 1.0:
            raise ValueError(
                "fator_produtividade_climatica deve estar entre 0 e 1."
            )

        self.fator_produtividade_climatica = float(
            fator_produtividade_climatica
        )

        if np.isfinite(self.produtividade_capital_normal):

            self.producao_normal_real = (
                self.produtividade_capital_normal
                * self.estoque_capital_real
            )
            self.produtividade_capital_capacidade = (
                self.produtividade_capital_normal
                / self.utilizacao_capacidade_normal
            )
            # Capacidade estrutural associada somente ao capital físico e u*.
            self.capacidade_produtiva_estrutural_real = (
                self.producao_normal_real
                / self.utilizacao_capacidade_normal
            )

            # Produtividade efetiva após o choque climático.
            self.produtividade_capital_capacidade_efetiva = (
                self.fator_produtividade_climatica
                * self.produtividade_capital_capacidade
            )

            # A produtividade idiossincrática afeta todas as firmas. Como o
            # multiplicador possui média unitária, não altera a tendência
            # produtiva calibrada quando se considera o ensemble.
            self.produtividade_capital_capacidade_efetiva = (
                self.produtividade_capital_capacidade_efetiva
                * self.fator_produtividade_idiossincratica
            )


            # Capacidade efetivamente disponível no período.
            self.capacidade_produtiva_real = (
                self.produtividade_capital_capacidade_efetiva
                * self.estoque_capital_real
            )

        else:
            self.producao_normal_real = np.nan
            self.produtividade_capital_capacidade_efetiva = np.nan

            self.capacidade_produtiva_estrutural_real = np.inf
            self.capacidade_produtiva_real = np.inf

    def calcular_producao_desejada(
        self,
        parametro_estoque_desejado: float,
        velocidade_ajuste_estoques: float,
    ) -> float:
        """Calcula a decisão descentralizada, sem impor capacidade física."""

        if self.forma_estoque:
            estoque_desejado = (
                parametro_estoque_desejado
                * self.demanda_esperada
            )

            ajuste_estoques = (
                velocidade_ajuste_estoques
                * (estoque_desejado - self.estoque)
            )

        else:
            ajuste_estoques = 0.0

        self.producao_desejada_real = max(
            0.0,
            self.demanda_esperada + ajuste_estoques,
        )
        return float(self.producao_desejada_real)

    def realizar_producao(
        self,
        quantidade_planejada_real: float,
    ) -> None:
        """Realiza a produção final, respeitando a capacidade física atual."""

        self.producao_planejada_real = max(
            0.0,
            float(quantidade_planejada_real),
        )

        # A intervenção do regulador já deve respeitar capacidade, mas esta
        # proteção física permanece na firma e impede produção impossível.
        if np.isfinite(self.capacidade_produtiva_real):
            self.producao_real = min(
                self.producao_planejada_real,
                self.capacidade_produtiva_real,
            )

            # Quantidade que a firma desejava produzir, mas não conseguiu
            # devido à restrição física de capacidade.
            self.producao_restringida_capacidade_real = max(
                0.0,
                self.producao_planejada_real
                - self.capacidade_produtiva_real,
            )

        else:

            # Firmas sem capacidade calibrada continuam sem restrição.
            self.producao_real = self.producao_planejada_real

            self.producao_restringida_capacidade_real = 0.0

        if (
            np.isfinite(self.capacidade_produtiva_real)
            and self.capacidade_produtiva_real > 0.0
        ):
            self.taxa_utilizacao_capacidade = (
                self.producao_real
                / self.capacidade_produtiva_real
            )

        else:
            self.taxa_utilizacao_capacidade = np.nan

    def decidir_producao(
        self,
        parametro_estoque_desejado: float,
        velocidade_ajuste_estoques: float,
    ) -> None:
        """Compatibilidade: decide descentralizadamente e realiza sem regulador."""

        producao_desejada = self.calcular_producao_desejada(
            parametro_estoque_desejado=parametro_estoque_desejado,
            velocidade_ajuste_estoques=velocidade_ajuste_estoques,
        )
        self.realizar_producao(producao_desejada)


    def calcular_taxa_retorno_observada(
        self,
        preco_capital: float,
        depreciacao: float,
        taxa_juros_real: float,
    ) -> float:
        """Calcula a rentabilidade realizada e a rentabilidade ajustada."""

        if preco_capital <= 0.0:
            raise ValueError(
                "preco_capital deve ser positivo."
            )

        if not 0.0 <= depreciacao < 1.0:
            raise ValueError(
                "depreciacao deve estar no intervalo [0, 1)."
            )

        self.preco_capital_observado = float(
            preco_capital
        )

        if self.estoque_capital_real <= 0.0:

            self.taxa_retorno_bruta_observada = np.nan
            self.taxa_retorno_observada = np.nan
            self.taxa_retorno_ajustada = np.nan

            return np.nan

        valor_capital_nominal = (
            self.preco_capital_observado
            * self.estoque_capital_real
        )

        self.taxa_retorno_bruta_observada = (
            self.eob_misto_realizado
            / valor_capital_nominal
        )

        self.taxa_retorno_observada = (
            self.taxa_retorno_bruta_observada
            - depreciacao
        )

        if np.isfinite(
            self.taxa_retorno_parametro
        ):

            self.taxa_retorno_ajustada = (
                self.taxa_retorno_observada
                - self.taxa_retorno_parametro
                - taxa_juros_real
            )

        else:

            self.taxa_retorno_ajustada = np.nan

        return self.taxa_retorno_observada


    def estoque_final(self) -> float:
        """Atualiza o estoque físico após a realização das vendas."""

        if self.forma_estoque:

            self.variacao_estoque_real = (
                self.producao_real
                - self.vendas_real
            )

            self.estoque = (
                self.estoque
                + self.variacao_estoque_real
            )

        else:

            self.variacao_estoque_real = 0.0
            self.estoque = 0.0

        return self.estoque
    
    def calcular_demanda_trabalho(self) -> None:
        """Calcula a demanda de trabalho necessária à produção planejada."""

        self.demanda_trabalho = (
            self.ocupacoes_unitario_base
            * self.producao_real
        )

    def calcular_demanda_intermediaria(self) -> pd.Series:
        """Calcula e registra a demanda real de insumos da firma."""

        self.demanda_intermediaria_real = (
            self.coeficientes_demanda_intermediaria
            * self.producao_real
        ).rename(self.id)

        return self.demanda_intermediaria_real


    # Métodos de componentes estocásticos:
        # Produtividade das commodities:

    def atualizar_produtividade_idiossincratica(
        self,
        rng: np.random.Generator,
        rho: float,
        sigma: float,
    ) -> float:
        """Atualiza a produtividade idiossincrática AR(1) da firma.

        O choque segue um AR(1) em log:

            z_t = rho * z_(t-1) + epsilon_t
            epsilon_t ~ N(0, sigma²)

        e o fator multiplicativo de produtividade é:

            psi_t = exp(z_t)

        A correção pela metade da variância acumulada garante que o fator
        multiplicativo tenha média unitária em cada período do ensemble.
        """

        if not 0.0 <= rho < 1.0:
            raise ValueError(
                "rho_produtividade_idiossincratica deve estar em [0, 1)."
            )

        if sigma < 0.0:
            raise ValueError(
                "sigma_produtividade_idiossincratica não pode ser negativo."
            )

        if sigma == 0.0:
            self.desvio_produtividade_idiossincratica = (
                rho * self.desvio_produtividade_idiossincratica
            )
            self.variancia_produtividade_idiossincratica = (
                rho**2 * self.variancia_produtividade_idiossincratica
            )
        else:
            choque = rng.normal(loc=0.0, scale=sigma)
            self.desvio_produtividade_idiossincratica = (
                rho * self.desvio_produtividade_idiossincratica + choque
            )
            self.variancia_produtividade_idiossincratica = (
                rho**2 * self.variancia_produtividade_idiossincratica
                + sigma**2
            )

        self.fator_produtividade_idiossincratica = float(
            np.exp(
                self.desvio_produtividade_idiossincratica
                - 0.5 * self.variancia_produtividade_idiossincratica
            )
        )
        return self.fator_produtividade_idiossincratica

        # Qualidade (vale para todos):
    def atualizar_qualidade(
        self,
        rng: np.random.Generator,
        rho_qualidade: float,
        sigma_qualidade: float,
    ) -> None:
        """Atualiza a qualidade em torno do nível calibrado no ano-base."""

        if not 0.0 <= rho_qualidade < 1.0:
            raise ValueError(
                "rho_qualidade deve estar no intervalo [0, 1)."
            )

        if sigma_qualidade < 0.0:
            raise ValueError(
                "sigma_qualidade não pode ser negativo."
            )

        if sigma_qualidade == 0.0:
            self.desvio_qualidade = rho_qualidade * self.desvio_qualidade
            self.variancia_qualidade = (
                rho_qualidade**2 * self.variancia_qualidade
            )
        else:
            choque = rng.normal(loc=0.0, scale=sigma_qualidade)
            self.desvio_qualidade = (
                rho_qualidade * self.desvio_qualidade + choque
            )
            self.variancia_qualidade = (
                rho_qualidade**2 * self.variancia_qualidade
                + sigma_qualidade**2
            )

        self.qualidade = (
            self.qualidade_base
            * np.exp(
                self.desvio_qualidade - 0.5 * self.variancia_qualidade
            )
        )



    def decidir_investimento(
        self,
        v: float,
        depreciacao: float,
        gamma_retorno: float,
        gamma_investimento_capacidade: float | None = None,
    ) -> None:
        """Decide o investimento conforme o regime da firma.

        Para firmas industriais, ``v`` não participa mais da regra: o ajuste
        líquido segue o hiato entre o capital desejado para a produção
        planejada e o estoque atual. A regra por retorno do leilão é mantida.
        """

        # ==========================================================
        # INVESTIMENTO LÍQUIDO
        # ==========================================================

        if self.regime == "industrial":
            gamma_capacidade = (
                gamma_retorno
                if gamma_investimento_capacidade is None
                else gamma_investimento_capacidade
            )
            if gamma_capacidade < 0.0:
                raise ValueError(
                    "gamma_investimento_capacidade não pode ser negativo."
                )

            if (
                np.isfinite(self.produtividade_capital_normal)
                and self.produtividade_capital_normal > 0.0
            ):
                self.capital_desejado = (
                    self.producao_planejada_real
                    / self.produtividade_capital_normal
                )
                self.gap_capital = (
                    self.capital_desejado
                    - self.estoque_capital_real
                )
                self.investimento_liquido = (
                    gamma_capacidade
                    * self.gap_capital
                )
            else:
                # Setores sem relação capital-produto calibrável não recebem
                # investimento pela regra industrial de capacidade.
                self.capital_desejado = np.nan
                self.gap_capital = np.nan
                self.investimento_liquido = 0.0

        elif self.regime == "leilao":

            if not np.isfinite(
                self.taxa_retorno_ajustada_anterior
            ):
                raise ValueError(
                    f"Taxa de retorno ajustada anterior inválida "
                    f"para {self.id}."
                )

            self.investimento_liquido = (
                gamma_retorno
                * self.estoque_capital_real
                * self.taxa_retorno_ajustada_anterior
            )
            self.capital_desejado = np.nan
            self.gap_capital = np.nan

        else:

            raise ValueError(
                f"Regime de investimento inválido: {self.regime}"
            )

        # ==========================================================
        # REPOSIÇÃO
        # ==========================================================

        self.investimento_reposicao = (
            depreciacao
            * self.estoque_capital_real
        )

        # ==========================================================
        # INVESTIMENTO BRUTO
        # ==========================================================

        self.investimento_bruto = max(
            0.0,
            self.investimento_liquido
            + self.investimento_reposicao,
        )


    def calcular_eob_recorrente_esperado(self) -> float:
        """Projeta o EOB recorrente da firma antes da realização do mercado."""

        if self.producao_anterior <= 0.0:
            # Uma coorte que não vende pode planejar produção nula no período
            # seguinte. Sem uma base física para a projeção, o EOB recorrente
            # esperado é nulo até a atividade voltar a ocorrer.
            self.eob_misto_recorrente_esperado = 0.0
            return self.eob_misto_recorrente_esperado

        if self.custo_unitario_anterior <= 0.0:
            raise ValueError(
                "custo_unitario_anterior deve ser positivo para calcular "
                "o EOB recorrente esperado."
            )

        crescimento_producao = (
            self.producao_planejada_real
            / self.producao_anterior
        )

        crescimento_custo_unitario = (
            self.custo_unitario
            / self.custo_unitario_anterior
        )

        self.eob_misto_recorrente_esperado = (
            self.eob_misto_realizado_anterior
            * crescimento_producao
            * crescimento_custo_unitario
        )

        return self.eob_misto_recorrente_esperado

    def calcular_dividendos(self) -> float:

        self.dividendos = (
            self.parametro_dividendos
            * max(0.0, self.eob_misto_recorrente_esperado)
        )

        return self.dividendos



    def atualizar_custo_e_preco(
        self,
        precos_insumos: pd.Series,
        indice_salarios: float,
        inflacao: float,
    ) -> float:
        """Atualiza custo e preço, sem tratar EOB+misto como custo."""

        if indice_salarios <= 0.0:
            raise ValueError("indice_salarios deve ser positivo.")
        if not isinstance(precos_insumos, pd.Series):
            precos_insumos = pd.Series(
                precos_insumos, index=self.tecnologia.index, dtype=float
            )
        precos_insumos = precos_insumos.reindex(self.tecnologia.index)
        if precos_insumos.isna().any() or (precos_insumos <= 0.0).any():
            raise ValueError(
                "precos_insumos deve conter preço positivo para todo insumo."
            )

        self.custo_intermediario_unitario_esperado = float(
            self.tecnologia @ precos_insumos
        )
        # Correção de custo pelo valor da inflação (inércia como expectativa):
        self.custo_intermediario_unitario_esperado = (1 + inflacao) * self.custo_intermediario_unitario_esperado


        # Nome anterior: custo previsto empregado na formação do preço.
        self.custo_intermediario_unitario = self.custo_intermediario_unitario_esperado
        self.remuneracao_unitaria = self.remuneracao_unitaria_base * indice_salarios
        self.salario_unitaria  = self.salario_unitaria_base * indice_salarios
        self.contribuicao_unitaria  = self.contribuicao_unitaria_base * indice_salarios
        self.outros_va_unitario = self.outros_va_unitario_base * indice_salarios
        
        self.custo_unitario = (
            self.custo_intermediario_unitario
            + self.remuneracao_unitaria
            + self.outros_va_unitario
        )

        if self.regime == "industrial":
            variacao_share = (
                self.market_share_t_1 - self.market_share_t_2
            ) / max(self.market_share_t_2, self.epsilon_market_share)
            self.markup = float(
                np.clip(
                    self.markup * (1.0 + self.parametro_markup * variacao_share),
                    self.markup_min,
                    self.markup_max,
                )
            )
            self.preco_firma = (1.0 + self.markup) * self.custo_unitario
            self.preco_transacao = self.preco_firma
            self.eob_misto_unitario = self.preco_firma - self.custo_unitario
        else:
            # O retorno normal do leilão é calibrado no ano-base. Ele não usa
            # K+S nem reage a market share até existir o mecanismo de despacho.
            self.preco_oferta_leilao = (
                (1.0 + self.markup_leilao) * self.custo_unitario
            )
            self.preco_transacao = self.preco_oferta_leilao
            self.eob_misto_unitario = (
                self.preco_oferta_leilao - self.custo_unitario
            )
        return self.preco_transacao

    def registrar_custo_intermediario_realizado(
        self,
        precos_comprador: pd.Series,
    ) -> float:
        """Registra o custo observado após Pc_t, sem redefinir o preço."""

        if not isinstance(precos_comprador, pd.Series):
            precos_comprador = pd.Series(
                precos_comprador, index=self.tecnologia.index, dtype=float
            )
        precos_comprador = precos_comprador.reindex(self.tecnologia.index)
        if precos_comprador.isna().any() or (precos_comprador <= 0.0).any():
            raise ValueError(
                "precos_comprador deve conter preço positivo para todo insumo."
            )
        self.custo_intermediario_unitario_realizado = float(
            self.tecnologia @ precos_comprador
        )
        return self.custo_intermediario_unitario_realizado

    def atualizar_market_shares_defasados(self, market_share_realizado: float) -> None:
        """Desloca o share realizado para uso defasado no markup futuro."""

        if market_share_realizado < 0.0:
            raise ValueError("market_share_realizado não pode ser negativo.")
        self.market_share_t_2 = self.market_share_t_1
        self.market_share_t_1 = float(market_share_realizado)

    def calcular_atratividade(self) -> float:
        """Calcula a atratividade de uma firma industrial no multilogit."""

        if self.regime != "industrial":
            return np.nan
        if self.qualidade < 0.0:
            raise ValueError("qualidade não pode ser negativa.")
        if self.preco_firma <= 0.0:
            raise ValueError("preco_firma deve ser positivo.")
        return float(
            self.qualidade**self.eta_qualidade
            * self.preco_firma**self.eta_preco
            * self.fator_atendimento**self.eta_atendimento
        )

    def calcular_consumo_intermediario_total(self) -> float:
        return self.custo_intermediario_unitario * self.producao_real

    def calcular_remuneracoes(self) -> float:
        return self.remuneracao_unitaria * self.producao_real
    
    def calcular_salarios(self) -> float:
        return self.salario_unitaria  * self.producao_real
    
    def calcular_contribuicoes(self) -> float:
        return self.contribuicao_unitaria  * self.producao_real

    def calcular_eob_misto(self) -> float:
        return self.eob_misto_unitario * self.producao_real

    def calcular_outros_va(self) -> float:
        return self.outros_va_unitario * self.producao_real

    def calcular_valor_adicionado(self) -> float:
        return (
            self.remuneracao_unitaria
            + self.eob_misto_unitario
            + self.outros_va_unitario
        ) * self.producao_real

    def calcular_custos_totais(self) -> float:
        return self.custo_unitario * self.producao_real

    def calcular_valor_producao_nominal(self) -> float:
        return self.preco_transacao * self.producao_real

    def calcular_valor_producao(self) -> float:
        """Compatibilidade: valor contábil da produção a preço de transação."""

        return self.calcular_valor_producao_nominal()

    def calcular_custos_e_preco(
        self,
        precos_comprador: pd.Series,
        indice_salarios: float,
        inflacao: float = 0.0,
    ) -> float:
        """Alias temporário para o nome anterior da atualização de preços."""

        return self.atualizar_custo_e_preco(
            precos_comprador,
            indice_salarios,
            inflacao,
        )


    def atualizar_estado(self, depreciacao: float) -> None:
        """Atualiza as variáveis defasadas para o período seguinte."""


        # Atualiza propdução realizada:
        self.producao_anterior = self.producao_real

        # Atualiza estoque de capital:
        self.estoque_capital_real = (
            (1.0 - depreciacao)
            * self.estoque_capital_real
            + self.investimento_bruto
        )

        self.taxa_retorno_ajustada_anterior = (
            self.taxa_retorno_ajustada
        )

        # Atualiza a capacidade que estará disponível no próximo período.
        self.atualizar_capacidade_produtiva(
            fator_produtividade_climatica=(
                self.fator_produtividade_climatica
            )
        )

        # Atualiza demanda realizada:

        self.demanda_realizada = self.vendas_real







    def calcular_resultado_realizado(self) -> None:
        """Calcula a contabilidade da firma após a realização do mercado."""

        # ==========================================================
        # VALOR DA PRODUÇÃO
        # ==========================================================

        # A variação de estoques funciona como a ponte entre produção e vendas:
        #
        #     producao = vendas + variacao_estoques
        #
        # Portanto, quando a variação de estoques é negativa, a firma vende mais
        # do que produziu no período, pois parte das vendas é atendida com estoques
        # acumulados anteriormente.
        #
        # Por esse motivo, para firmas que formam estoques, o valor da produção
        # deve ser calculado com base na produção corrente:
        #
        #     valor_producao = preco * producao
        #
        # e não com base nas vendas:
        #
        #     valor_vendas = preco * vendas
        #
        # A diferença entre produção e vendas já é contabilizada separadamente
        # como variação de estoques na CEI. Se o valor da produção fosse calculado
        # pelas vendas, os bens retirados dos estoques seriam tratados como se
        # tivessem sido produzidos no período corrente e, posteriormente, a redução
        # dos estoques seria registrada novamente na CEI, gerando dupla contagem.
        #
        # Para firmas que não formam estoques, mantém-se o valor das vendas como
        # quantidade contabilizada, pois a produção não vendida não é incorporada
        # aos estoques.

        if self.forma_estoque:
            quantidade_contabilizada = self.producao_real
        else:
            quantidade_contabilizada = self.vendas_real



        self.valor_producao_nominal_realizado = (
            self.preco_transacao
            * quantidade_contabilizada
        )

        # ==========================================================
        # CONSUMO INTERMEDIÁRIO
        # ==========================================================

        self.consumo_intermediario_nominal_realizado = (
            self.custo_intermediario_unitario_realizado
            * self.producao_real
        )

        # ==========================================================
        # VALOR ADICIONADO
        # ==========================================================

        self.valor_adicionado_realizado = (
            self.valor_producao_nominal_realizado
            - self.consumo_intermediario_nominal_realizado
        )

        # ==========================================================
        # REMUNERAÇÕES E OUTROS COMPONENTES DO VA
        # ==========================================================

        self.remuneracoes_realizadas = (
            self.remuneracao_unitaria
            * self.producao_real
        )

        self.salarios_realizados = (
            self.salario_unitaria
            * self.producao_real
        )

        self.contribuicoes_realizadas = (
            self.contribuicao_unitaria
            * self.producao_real
        )

        self.outros_va_realizados = (
            self.outros_va_unitario
            * self.producao_real
        )

        # ==========================================================
        # EOB + RENDIMENTO MISTO REALIZADO
        # ==========================================================

        self.eob_misto_realizado = (
            self.valor_adicionado_realizado
            - self.remuneracoes_realizadas
            - self.outros_va_realizados
        )


    # Aliases temporários somente para o código antigo continuar legível durante
    # a refatoração por etapas. Código novo deve usar a nomenclatura acima.
    @property
    def salario_unitario_base(self) -> float:
        return self.remuneracao_unitaria_base

    @property
    def lucro_normal_unitario_base(self) -> float:
        return self.eob_misto_unitario_base

    @property
    def salario_unitario(self) -> float:
        return self.remuneracao_unitaria

    @property
    def lucro_normal_unitario(self) -> float:
        return self.eob_misto_unitario

    def calcular_lucro_normal(self) -> float:
        """Alias temporário de :meth:`calcular_eob_misto`."""

        return self.calcular_eob_misto()
