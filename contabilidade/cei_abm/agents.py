from __future__ import annotations

from dataclasses import dataclass, field

from .accounts import RUBRICAS_CEI


def contas_cei_zeradas() -> dict[str, float]:
    return {rubrica: 0.0 for rubrica in RUBRICAS_CEI}


@dataclass
class AgenteInstitucional:
    id: int
    setor: str
    contas: dict[str, float] = field(default_factory=contas_cei_zeradas)


@dataclass
class Familias(AgenteInstitucional):
    setor: str = "S.14 Familias"


@dataclass
class Governo(AgenteInstitucional):
    setor: str = "S.13 Governo Geral"


@dataclass
class EmpresaFinanceira(AgenteInstitucional):
    setor: str = "S.12 Empresas financeiras"


@dataclass
class EmpresaNaoFinanceira(AgenteInstitucional):
    setor: str = "S.11 Empresas nao-financeiras"


@dataclass
class SetorExterno(AgenteInstitucional):
    setor: str = "Setor externo"
