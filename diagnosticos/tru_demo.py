from __future__ import annotations

import pandas as pd

from configuracao_projeto import DATA_DIR
from contabilidade.cei_abm.tru import (
    build_tru_model,
    load_tru_data,
    simulate_intermediate_consumption_shock,
)


def main() -> None:
    data = load_tru_data(data_dir=DATA_DIR, year=2020, level=20)
    model = build_tru_model(data)

    shock = simulate_intermediate_consumption_shock(
        model,
        product_rows=range(1, 3),
        sector_cols=range(1, 3),
        multiplier=1.10,
    )

    output_table = pd.DataFrame(
        {
            "setor": data.sector_names,
            "producao_antes": shock.output_sector_before.ravel(),
            "producao_depois": shock.output_sector_after.ravel(),
            "variacao_producao_pct": shock.output_change_percent.ravel(),
            "inflacao_setorial_pct": shock.sector_inflation_percent.ravel(),
        }
    )

    print("TRU carregada.")
    print(f"Ano: {data.year}")
    print(f"Produtos: {len(data.product_names)}")
    print(f"Setores: {len(data.sector_names)}")
    print(f"Producao total antes: {shock.output_sector_before.sum():.2f}")
    print(f"Producao total depois: {shock.output_sector_after.sum():.2f}")
    print(f"Inflacao agregada: {shock.aggregate_inflation_percent:.6f}%")
    print(f"PIB antes: {shock.gdp_before:.2f}")
    print(f"PIB depois: {shock.gdp_after:.2f}")
    print(f"Massa salarial antes: {shock.wages_before:.2f}")
    print(f"Massa salarial depois: {shock.wages_after:.2f}")
    print()
    print(output_table.to_string(index=False))


if __name__ == "__main__":
    main()


