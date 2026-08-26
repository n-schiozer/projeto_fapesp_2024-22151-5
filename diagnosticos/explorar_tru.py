# %%

from __future__ import annotations

# %%
import pandas as pd

from configuracao_projeto import DATA_DIR
from contabilidade.cei_abm.tru import (
    build_tru_model,
    load_tru_data,
    simulate_intermediate_consumption_shock,
)

data_dir = DATA_DIR

data = load_tru_data(data_dir=data_dir, year=2020, level=20)
model = build_tru_model(data)

# %%
producao = pd.DataFrame(
    data.production,
    index=data.product_names,
    columns=data.sector_names,
)

producao

# %%

marketshare = pd.DataFrame(
    model.market_share_domestic,
    index=data.product_names,
    columns=data.sector_names,
)

marketshare.sum(axis=1)

# %%
data.product_names

marketshare.loc[["Outras atividades de serviços"]]


# %%

model.output_sector.reshape(-1, 1)

# %%

model.market_share_domestic @ model.output_sector.reshape(-1, 1)


# %%
consumo_intermediario = pd.DataFrame(
    data.intermediate_consumption,
    index=data.product_names,
    columns=data.sector_names,
)

consumo_intermediario

# %%
leontief = pd.DataFrame(
    model.leontief_matrix,
    index=data.sector_names,
    columns=data.sector_names,
)

leontief

# %%
producao_setorial = pd.Series(
    model.output_sector.ravel(),
    index=data.sector_names,
    name="producao_setorial",
)

producao_setorial

# %%
marketshare = pd.Series(
    model.market_share_domestic,
    index=data.sector_names,
    name="producao_setorial",
)

marketshare


# %%
shock = simulate_intermediate_consumption_shock(
    model,
    product_rows=range(1, 3),
    sector_cols=range(1, 3),
    multiplier=1.10,
)

resultado_choque = pd.DataFrame(
    {
        "setor": data.sector_names,
        "producao_antes": shock.output_sector_before.ravel(),
        "producao_depois": shock.output_sector_after.ravel(),
        "variacao_producao_pct": shock.output_change_percent.ravel(),
        "inflacao_setorial_pct": shock.sector_inflation_percent.ravel(),
    }
)

resultado_choque
