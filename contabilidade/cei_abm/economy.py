from __future__ import annotations

from dataclasses import dataclass, field

from .agents import (
    AgenteInstitucional,
    EmpresaFinanceira,
    EmpresaNaoFinanceira,
    Familias,
    Governo,
    SetorExterno,
)
from .accounts import VALORES_INICIAIS_CEI
from .balance import BalancoCEI


@dataclass(frozen=True)
class SimulationConfig:
    periods: int = 10
    n_familias: int = 100
    n_empresas_nao_financeiras: int = 20
    n_empresas_financeiras: int = 5


@dataclass
class Economy:
    config: SimulationConfig
    period: int = 0
    history: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.familias = [
            Familias(id=i + 1)
            for i in range(self.config.n_familias)
        ]
        self.empresas_nao_financeiras = [
            EmpresaNaoFinanceira(id=i + 1)
            for i in range(self.config.n_empresas_nao_financeiras)
        ]
        self.empresas_financeiras = [
            EmpresaFinanceira(id=i + 1)
            for i in range(self.config.n_empresas_financeiras)
        ]
        self.governo = Governo(id=1)
        self.setor_externo = SetorExterno(id=1)
        self._distribuir_valores_iniciais()
        self.balanco = BalancoCEI(self.agentes)

    @property
    def agentes(self) -> list[AgenteInstitucional]:
        return [
            *self.familias,
            *self.empresas_nao_financeiras,
            *self.empresas_financeiras,
            self.governo,
            self.setor_externo,
        ]

    def _agentes_por_setor(self, setor: str) -> list[AgenteInstitucional]:
        return [agente for agente in self.agentes if agente.setor == setor]

    def _distribuir_valores_iniciais(self) -> None:
        for setor, valores_setor in VALORES_INICIAIS_CEI.items():
            agentes_setor = self._agentes_por_setor(setor)
            if not agentes_setor:
                continue

            for rubrica, valor_total in valores_setor.items():
                valor_por_agente = valor_total / len(agentes_setor)

                for agente in agentes_setor:
                    agente.contas[rubrica] = valor_por_agente

    def step(self) -> None:
        self.balanco = BalancoCEI(self.agentes)
        self.history.append(
            {
                "period": self.period,
                "checks_invalidos": self.balanco.checks_invalidos(),
                "capacidade_financiamento": self.balanco.capacidade_financiamento(),
            }
        )
        self.period += 1

    def run(self, verbose: bool = False) -> BalancoCEI:
        for _ in range(self.config.periods):
            self.step()
            if verbose:
                print(f"Step {self.period} completa")
        return self.balanco
