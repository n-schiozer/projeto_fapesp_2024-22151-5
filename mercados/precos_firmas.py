"""Formação e agregação transitória do preço básico doméstico das firmas."""

import numpy as np
import pandas as pd

from agentes.firma import Firma


def agregar_precos_firmas(firmas: dict[str, Firma], setores: list[str]) -> pd.Series:
    """Agrega Pb doméstico pelos shares-base até existirem vendas realizadas.

    O importado não participa. Quando o mercado existir, os shares-base serão
    substituídos por pesos de vendas, sem mudar a definição do preço básico.
    """

    pb = pd.Series(0.0, index=list(setores), name="preco_basico_domestico")
    for setor in pb.index:
        firmas_setor = [firma for firma in firmas.values() if firma.setor == setor]
        if not firmas_setor:
            raise ValueError(f"Não há firmas domésticas no setor {setor}.")
        pesos = np.asarray(
            [firma.share_domestico_inicial for firma in firmas_setor], dtype=float
        )
        if not np.allclose(pesos.sum(), 1.0, rtol=1e-11, atol=1e-12):
            raise ValueError(f"Shares domésticos não somam 1 no setor {setor}.")
        precos = np.asarray(
            [firma.preco_transacao for firma in firmas_setor], dtype=float
        )
        if (precos <= 0.0).any() or not np.isfinite(precos).all():
            raise ValueError(f"Preço de transação inválido no setor {setor}.")
        pb.at[setor] = float(pesos @ precos)
    return pb
