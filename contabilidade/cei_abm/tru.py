from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TRUData:
    year: int
    product_names: list[str]
    sector_names: list[str]
    production: np.ndarray
    imports: np.ndarray
    intermediate_consumption: np.ndarray
    value_added_components: np.ndarray
    final_demand_product: np.ndarray
    final_demand_product_total: np.ndarray
    trade_margin: np.ndarray
    transport_margin: np.ndarray
    taxes: np.ndarray


@dataclass
class TRUModel:
    data: TRUData
    domestic_supply_product: np.ndarray
    total_supply_product: np.ndarray
    market_share_domestic: np.ndarray
    intermediate_consumption_sector: np.ndarray
    technical_coefficients_sector: np.ndarray
    import_intermediate_product: np.ndarray
    import_final_product: np.ndarray
    import_intermediate_sector: np.ndarray
    import_final_sector: np.ndarray
    import_coefficients: np.ndarray
    tax_coefficients: np.ndarray
    trade_margin_coefficients: np.ndarray
    transport_margin_coefficients: np.ndarray
    leontief_matrix: np.ndarray
    leontief_inverse: np.ndarray
    final_demand_sector: np.ndarray
    output_sector: np.ndarray


@dataclass
class TRUShockResult:
    output_sector_before: np.ndarray
    output_sector_after: np.ndarray
    output_change_percent: np.ndarray
    price_level_before: np.ndarray
    price_level_after: np.ndarray
    sector_inflation_percent: np.ndarray
    aggregate_inflation_percent: float
    value_added_before: np.ndarray
    value_added_after: np.ndarray
    gdp_before: float
    gdp_after: float
    wages_before: float
    wages_after: float


def _table_path(data_dir: Path, level: int, table: int, year: int) -> Path:
    xlsx_path = data_dir / f"{level}_tab{table}_{year}.xlsx"
    if xlsx_path.exists():
        return xlsx_path

    xls_path = data_dir / f"{level}_tab{table}_{year}.xls"
    if xls_path.exists():
        return xls_path

    raise FileNotFoundError(
        f"Nao encontrei {level}_tab{table}_{year}.xlsx nem {level}_tab{table}_{year}.xls em {data_dir}"
    )


def _read_excel_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")

    try:
        return pd.read_excel(path, sheet_name=sheet_name, header=None, engine="xlrd")
    except ImportError as exc:
        raise ImportError(
            "Para ler arquivos .xls antigos, instale xlrd ou converta os arquivos para .xlsx."
        ) from exc


def _numeric_block(df: pd.DataFrame, rows: slice, cols: slice) -> np.ndarray:
    return (
        df.iloc[rows, cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )


def load_tru_data(data_dir: str | Path, year: int, level: int = 20) -> TRUData:
    data_dir = Path(data_dir)
    tab1 = _table_path(data_dir, level=level, table=1, year=year)
    tab2 = _table_path(data_dir, level=level, table=2, year=year)

    oferta = _read_excel_table(tab1, "oferta")
    producao = _read_excel_table(tab1, "producao")
    importacao = _read_excel_table(tab1, "importacao")
    ci = _read_excel_table(tab2, "CI")
    va = _read_excel_table(tab2, "VA")
    demanda = _read_excel_table(tab2, "demanda")

    data_start_row = 5
    num_products = producao.shape[0] - data_start_row - 2
    num_sectors = producao.shape[1] - 3

    product_names = [
        str(value).strip()
        for value in producao.iloc[data_start_row : data_start_row + num_products, 1].fillna("")
    ]
    sector_names = [
        str(value).strip()
        for value in ci.iloc[3, 2 : 2 + num_sectors].fillna("")
    ]

    return TRUData(
        year=year,
        product_names=product_names,
        sector_names=sector_names,
        production=_numeric_block(
            producao,
            slice(data_start_row, data_start_row + num_products),
            slice(2, 2 + num_sectors),
        ),
        imports=_numeric_block(
            importacao,
            slice(data_start_row, data_start_row + num_products),
            slice(2, 3),
        ),
        intermediate_consumption=_numeric_block(
            ci,
            slice(data_start_row, data_start_row + num_products),
            slice(2, 2 + num_sectors),
        ),
        value_added_components=_numeric_block(
            va,
            slice(data_start_row, data_start_row + 14),
            slice(1, 1 + num_sectors),
        ),
        final_demand_product=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(8, 9),
        ),
        final_demand_product_total=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(9, 10),
        ),
        trade_margin=_numeric_block(
            oferta,
            slice(data_start_row, data_start_row + num_products),
            slice(3, 4),
        ),
        transport_margin=_numeric_block(
            oferta,
            slice(data_start_row, data_start_row + num_products),
            slice(4, 5),
        ),
        taxes=_numeric_block(
            oferta,
            slice(data_start_row, data_start_row + num_products),
            slice(9, 10),
        ),
    )


def build_tru_model(data: TRUData) -> TRUModel:
    production = data.production
    imports = data.imports
    ci = data.intermediate_consumption
    output = data.value_added_components[12, :].reshape(-1, 1)

    domestic_supply_product = production.sum(axis=1, keepdims=True)
    total_supply_product = domestic_supply_product + imports
    market_share_domestic = _safe_divide(production, domestic_supply_product)

    ci_sector = market_share_domestic.T @ ci
    technical_coefficients_sector = _safe_divide(ci_sector, output.T)

    ci_total_product = ci.sum(axis=1, keepdims=True)
    final_total_product = data.final_demand_product_total
    final_import_share = _safe_divide(
        final_total_product,
        final_total_product + ci_total_product,
    )
    intermediate_import_share = 1.0 - final_import_share

    import_intermediate_product = intermediate_import_share * imports
    import_final_product = final_import_share * imports
    import_intermediate_sector = market_share_domestic.T @ import_intermediate_product
    import_final_sector = market_share_domestic.T @ import_final_product

    import_coefficients = np.diagflat(_safe_divide(import_intermediate_sector, output))
    tax_coefficients = np.diagflat(_safe_divide(market_share_domestic.T @ data.taxes, output))
    trade_margin_coefficients = np.diagflat(
        _safe_divide(market_share_domestic.T @ data.trade_margin, output)
    )
    transport_margin_coefficients = np.diagflat(
        _safe_divide(market_share_domestic.T @ data.transport_margin, output)
    )

    identity = np.eye(production.shape[1])
    leontief_matrix = (
        identity
        - technical_coefficients_sector
        + import_coefficients
        + tax_coefficients
        + trade_margin_coefficients
        + transport_margin_coefficients
    )
    leontief_inverse = np.linalg.inv(leontief_matrix)

    final_demand_sector = market_share_domestic.T @ data.final_demand_product
    output_sector = leontief_inverse @ (final_demand_sector - import_final_sector)

    return TRUModel(
        data=data,
        domestic_supply_product=domestic_supply_product,
        total_supply_product=total_supply_product,
        market_share_domestic=market_share_domestic,
        intermediate_consumption_sector=ci_sector,
        technical_coefficients_sector=technical_coefficients_sector,
        import_intermediate_product=import_intermediate_product,
        import_final_product=import_final_product,
        import_intermediate_sector=import_intermediate_sector,
        import_final_sector=import_final_sector,
        import_coefficients=import_coefficients,
        tax_coefficients=tax_coefficients,
        trade_margin_coefficients=trade_margin_coefficients,
        transport_margin_coefficients=transport_margin_coefficients,
        leontief_matrix=leontief_matrix,
        leontief_inverse=leontief_inverse,
        final_demand_sector=final_demand_sector,
        output_sector=output_sector,
    )


def simulate_intermediate_consumption_shock(
    model: TRUModel,
    product_rows: range,
    sector_cols: range,
    multiplier: float,
) -> TRUShockResult:
    data = model.data
    ci_shock = data.intermediate_consumption.copy()

    product_idx = [idx - 1 for idx in product_rows]
    sector_idx = [idx - 1 for idx in sector_cols]
    ci_shock[np.ix_(product_idx, sector_idx)] *= multiplier

    output = data.value_added_components[12, :].reshape(-1, 1)
    ci_sector_shock = model.market_share_domestic.T @ ci_shock
    technical_coefficients_shock = _safe_divide(ci_sector_shock, output.T)

    identity = np.eye(data.production.shape[1])
    leontief_shock = (
        identity
        - technical_coefficients_shock
        + model.import_coefficients
        + model.tax_coefficients
        + model.trade_margin_coefficients
        + model.transport_margin_coefficients
    )
    leontief_inverse_shock = np.linalg.inv(leontief_shock)
    output_after = leontief_inverse_shock @ (
        model.final_demand_sector - model.import_final_sector
    )
    output_change_percent = (output_after / model.output_sector - 1.0) * 100.0

    value_added = data.value_added_components[0, :].reshape(1, -1)
    primary_coefficients = value_added / output.T
    price_level_before = primary_coefficients @ model.leontief_inverse
    price_level_after = primary_coefficients @ leontief_inverse_shock
    sector_inflation_percent = (price_level_after / price_level_before - 1.0) * 100.0
    inflation_weights = model.output_sector / model.output_sector.sum()
    aggregate_inflation_percent = (sector_inflation_percent @ inflation_weights).item()

    va_proportions = _safe_divide(data.value_added_components, output.T)
    value_added_after = va_proportions * output_after.T

    return TRUShockResult(
        output_sector_before=model.output_sector,
        output_sector_after=output_after,
        output_change_percent=output_change_percent,
        price_level_before=price_level_before,
        price_level_after=price_level_after,
        sector_inflation_percent=sector_inflation_percent,
        aggregate_inflation_percent=aggregate_inflation_percent,
        value_added_before=data.value_added_components,
        value_added_after=value_added_after,
        gdp_before=float(data.value_added_components[0, :].sum()),
        gdp_after=float(value_added_after[0, :].sum()),
        wages_before=float(data.value_added_components[1, :].sum()),
        wages_after=float(value_added_after[1, :].sum()),
    )

