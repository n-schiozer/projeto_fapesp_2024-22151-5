from __future__ import annotations

from dataclasses import dataclass

from .accounts import RUBRICAS_CEI, RUBRICAS_SEM_CHECK_ZERO, SETORES_CEI
from .agents import AgenteInstitucional


@dataclass
class BalancoCEI:
    agentes: list[AgenteInstitucional]

    def tabela(self) -> list[dict[str, float | str]]:
        linhas: list[dict[str, float | str]] = []

        for rubrica in RUBRICAS_CEI:
            linha: dict[str, float | str] = {"Rubrica": rubrica}
            check = 0.0

            for setor in SETORES_CEI:
                valor_setor = sum(
                    agente.contas[rubrica]
                    for agente in self.agentes
                    if agente.setor == setor
                )
                linha[f"{setor} Entrada"] = max(valor_setor, 0.0)
                linha[f"{setor} Saida"] = max(-valor_setor, 0.0)
                check += valor_setor

            linha["Check"] = check
            linhas.append(linha)

        return linhas

    def checks(self) -> dict[str, float]:
        return {
            str(linha["Rubrica"]): float(linha["Check"])
            for linha in self.tabela()
        }

    def checks_invalidos(
        self,
        tolerancia: float = 1e-9,
        ignorar_sem_check_zero: bool = True,
    ) -> dict[str, float]:
        rubricas_ignoradas = (
            set(RUBRICAS_SEM_CHECK_ZERO) if ignorar_sem_check_zero else set()
        )
        return {
            rubrica: check
            for rubrica, check in self.checks().items()
            if rubrica not in rubricas_ignoradas and abs(check) > tolerancia
        }

    def capacidade_financiamento(self) -> dict[str, float]:
        resultado = {}

        for setor in SETORES_CEI:
            resultado[setor] = sum(
                valor
                for agente in self.agentes
                if agente.setor == setor
                for valor in agente.contas.values()
            )

        return resultado
