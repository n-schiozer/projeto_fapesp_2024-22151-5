# %%
from __future__ import annotations

import pandas as pd

from configuracao_projeto import DATA_DIR
from contabilidade.cei_abm.tru_sector_sector import (
    load_tru_data,
    transform_tru_to_sector_sector,
)

# %%

data_dir = DATA_DIR

data = load_tru_data(data_dir=data_dir, year=2020, level=20)

data.final_demand_product = data.exports + data.gov_cons + data.npo_consumption + data.household_consumption + data.gross_investment + data.stocks_investment

# %%
tru_sector_sector = transform_tru_to_sector_sector(data)

excel_table = tru_sector_sector.to_excel_dataframe()

# %%

print(excel_table)


# %% 

output_table = pd.DataFrame(
        {
            "setor": data.sector_names,
            "producao": tru_sector_sector.output_sector.ravel()
        }
    )

print(output_table)


# %%
