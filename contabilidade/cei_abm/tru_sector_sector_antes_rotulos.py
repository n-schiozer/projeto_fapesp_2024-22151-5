from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


@dataclass
class TRUData:
    year: int
    product_names: list[str]
    sector_names: list[str]
    va_components_names: list[str]
    production: np.ndarray
    imports: np.ndarray
    intermediate_consumption: np.ndarray
    value_added_components: np.ndarray
    exports: np.ndarray
    gov_cons: np.ndarray
    npo_consumption: np.ndarray
    household_consumption: np.ndarray
    gross_investment: np.ndarray
    stocks_investment: np.ndarray
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
    intermediate_consumption_sector: np.ndarray
    technical_coefficients_sector: np.ndarray
    market_share_domestic: np.ndarray
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
class TRUModel_2:
    data: TRUData
    production: np.ndarray
    domestic_supply_product: np.ndarray
    total_supply_product: np.ndarray
    intermediate_consumption_sector: np.ndarray
    technical_coefficients_sector: np.ndarray
    tax_coefficients: np.ndarray
    trade_margin_coefficients: np.ndarray
    transport_margin_coefficients: np.ndarray
    leontief_matrix: np.ndarray
    leontief_inverse: np.ndarray
    final_demand_sector: np.ndarray
    output_basic_price: np.ndarray
    output_market_prices: np.ndarray


@dataclass
class TRUSectorSector:
    """TRU converted from product rows to sector rows."""

    data: TRUData
    market_share_domestic: np.ndarray
    production_sector_sector: np.ndarray
    intermediate_consumption_sector: np.ndarray
    final_demand_components_sector: np.ndarray
    exports_sector: np.ndarray
    gov_cons_sector: np.ndarray
    npo_consumption_sector: np.ndarray
    household_consumption_sector: np.ndarray
    gross_investment_sector: np.ndarray
    stocks_investment_sector: np.ndarray
    final_demand_sector: np.ndarray
    total_demand_with_margins_sector: np.ndarray
    trade_margin_sector: np.ndarray
    transport_margin_sector: np.ndarray
    taxes_sector: np.ndarray
    imports_sector: np.ndarray
    margins_and_taxes_sector: np.ndarray
    sector_demand_basic_prices: np.ndarray
    final_demand_basic_prices: np.ndarray
    value_added_components_sector: np.ndarray
    value_added_sector: np.ndarray
    wages_sector: np.ndarray
    profit_sector: np.ndarray
    technical_coefficients_sector: np.ndarray
    leontief_matrix: np.ndarray
    leontief_inverse: np.ndarray
    output_sector: np.ndarray
    accounting_gap: np.ndarray

    def to_excel_dataframe(self) -> pd.DataFrame:
        """Return the two side-by-side blocks used in the reference workbook."""

        sector_codes = [
            name.splitlines()[0].strip() or str(index + 1)
            for index, name in enumerate(self.data.sector_names)
        ]
        sector_descriptions = [
            " ".join(part.strip() for part in name.splitlines()[1:] if part.strip())
            or name
            for name in self.data.sector_names
        ]

        left_values = np.column_stack(
            [
                self.final_demand_sector,
                self.intermediate_consumption_sector,
                self.total_demand_with_margins_sector,
                self.margins_and_taxes_sector,
                self.imports_sector,
                self.sector_demand_basic_prices,
            ]
        )
        left_columns = [
            "DF",
            *(f"CI {code}" for code in sector_codes),
            "Dem.total (com margens)",
            "Margens + impostos",
            "Importacao",
            "Demanda setorial",
        ]
        left = pd.DataFrame(left_values, columns=left_columns)

        right_values = np.column_stack(
            [
                self.intermediate_consumption_sector.T,
                self.value_added_sector,
                self.wages_sector,
                self.profit_sector,
            ]
        )
        right_columns = [
            *(f"CI {code}" for code in sector_codes),
            "VA",
            "W",
            "Profit",
        ]
        right = pd.DataFrame(right_values, columns=right_columns)

        identification = pd.DataFrame(
            {
                "Setor": sector_codes,
                "Descricao": sector_descriptions,
            }
        )
        separator = pd.DataFrame({"": [""] * len(sector_codes)})
        check = pd.DataFrame({"Check": self.accounting_gap.ravel()})
        return pd.concat(
            [identification, left, separator, right, check],
            axis=1,
        )


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

def product_rows_to_sectors(
    values: np.ndarray,
    market_share_domestic: np.ndarray,
    *,
    name: str = "values",
) -> np.ndarray:
    """Aggregate product rows to sector rows using the domestic market share."""

    matrix = np.asarray(values, dtype=float)
    market_share = np.asarray(market_share_domestic, dtype=float)

    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} deve ser vetor ou matriz 2D; shape recebido: {matrix.shape}.")
    if market_share.ndim != 2:
        raise ValueError(
            "market_share_domestic deve ser uma matriz produto x setor."
        )
    if matrix.shape[0] != market_share.shape[0]:
        raise ValueError(
            f"{name} tem {matrix.shape[0]} linhas, mas a matriz de market share "
            f"tem {market_share.shape[0]} produtos."
        )
    if not np.isfinite(matrix).all() or not np.isfinite(market_share).all():
        raise ValueError(f"{name} e market_share_domestic devem conter valores finitos.")

    return market_share.T @ matrix

def transform_tru_to_sector_sector(
    data: TRUData,
    *,
    value_added_row: int = 0,
    wages_row: int = 1,
    output_row: int = 12,
    validate: bool = True,
) -> TRUSectorSector:
    
    """Convert all product-row TRU blocks to a symmetric sector-by-sector TRU.

    The transformation matrix is D = M.T, where M[p, s] is sector s's share
    in the domestic production of product p. Product-by-sector tables become
    sector-by-sector tables, while product vectors become sector vectors.
    Value-added rows are copied because they are already component-by-sector.
    """

    production = np.asarray(data.production, dtype=float)
    if production.ndim != 2:
        raise ValueError("data.production deve ser uma matriz produto x setor.")

    num_products, num_sectors = production.shape
    if len(data.product_names) != num_products:
        raise ValueError(
            "O numero de nomes de produtos nao coincide com as linhas de production."
        )
    if len(data.sector_names) != num_sectors:
        raise ValueError(
            "O numero de nomes de setores nao coincide com as colunas de production."
        )

    value_added_components = np.asarray(data.value_added_components, dtype=float)
    if value_added_components.ndim != 2:
        raise ValueError(
            "data.value_added_components deve ser uma matriz componente x setor."
        )
    if value_added_components.shape[1] != num_sectors:
        raise ValueError(
            "value_added_components ja deve estar em componente x setor."
        )
    required_va_row = max(value_added_row, wages_row, output_row)
    if required_va_row >= value_added_components.shape[0]:
        raise ValueError(
            f"value_added_components nao possui a linha {required_va_row}."
        )

    domestic_supply_product = production.sum(axis=1, keepdims=True)
    
    if np.any(domestic_supply_product < 0):
        raise ValueError("A producao domestica total por produto nao pode ser negativa.")

    zero_supply = domestic_supply_product.ravel() == 0
    if np.any(zero_supply):
        names = [
            data.product_names[index]
            for index in np.flatnonzero(zero_supply)[:5]
        ]
        warnings.warn(
            "Produtos sem producao domestica nao podem ser distribuidos por market "
            f"share e serao zerados na transformacao: {names}.",
            RuntimeWarning,
            stacklevel=2,
        )

    market_share_domestic = _safe_divide(
        production,
        domestic_supply_product,
    )

    def convert(values: np.ndarray, name: str) -> np.ndarray:
        return product_rows_to_sectors(
            values,
            market_share_domestic,
            name=name,
        )

    intermediate_consumption_sector = convert(
        data.intermediate_consumption,
        "intermediate_consumption",
    )
    if intermediate_consumption_sector.shape != (num_sectors, num_sectors):
        raise ValueError(
            "intermediate_consumption deve ter formato produto x setor."
        )

    final_demand_components_product = np.column_stack(
        [
            np.asarray(data.exports, dtype=float).reshape(-1),
            np.asarray(data.gov_cons, dtype=float).reshape(-1),
            np.asarray(data.npo_consumption, dtype=float).reshape(-1),
            np.asarray(data.household_consumption, dtype=float).reshape(-1),
            np.asarray(data.gross_investment, dtype=float).reshape(-1),
            np.asarray(data.stocks_investment, dtype=float).reshape(-1),
        ]
    )
    
    final_demand_components_sector = convert(
        final_demand_components_product,
        "final_demand_components",
    )

    exports_sector = final_demand_components_sector[:, [0]]
    gov_cons_sector = final_demand_components_sector[:, [1]]
    npo_consumption_sector = final_demand_components_sector[:, [2]]
    household_consumption_sector = final_demand_components_sector[:, [3]]
    gross_investment_sector = final_demand_components_sector[:, [4]]
    stocks_investment_sector = final_demand_components_sector[:, [5]]
        

    final_demand_sector = convert(
        data.final_demand_product,
        "final_demand_product",
    )
    total_demand_with_margins_sector = convert(
        data.final_demand_product_total,
        "final_demand_product_total",
    )
    
    production_sector_sector = convert(production, "production")
    trade_margin_sector = convert(data.trade_margin, "trade_margin")
    transport_margin_sector = convert(data.transport_margin, "transport_margin")
    taxes_sector = convert(data.taxes, "taxes")
    imports_sector = convert(data.imports, "imports")
    margins_and_taxes_sector = (
        trade_margin_sector + transport_margin_sector + taxes_sector
    )
    sector_demand_basic_prices = (
        total_demand_with_margins_sector
        - margins_and_taxes_sector
        - imports_sector
    )

    value_added_sector = value_added_components[[value_added_row], :].T.copy()
    wages_sector = value_added_components[[wages_row], :].T.copy()
    profit_sector = value_added_sector - wages_sector
    
    output_sector = convert(
        data.final_demand_product_total,
        "final_demand_components",
    )

    final_demand_basic_prices = (
        sector_demand_basic_prices
        - intermediate_consumption_sector.sum(axis=1, keepdims=True)
    )
    technical_coefficients_sector = _safe_divide(
        intermediate_consumption_sector,
        total_demand_with_margins_sector.T,
    )
    leontief_matrix = np.eye(num_sectors) - technical_coefficients_sector
    leontief_inverse = np.linalg.inv(leontief_matrix)

    accounting_gap = (
        sector_demand_basic_prices
        - intermediate_consumption_sector.T.sum(axis=1, keepdims=True)
        - value_added_sector
    )

    result = TRUSectorSector(
        data=data,
        market_share_domestic=market_share_domestic,
        production_sector_sector=production_sector_sector,
        intermediate_consumption_sector=intermediate_consumption_sector,
        final_demand_components_sector=final_demand_components_sector,
        exports_sector=exports_sector,
        gov_cons_sector=gov_cons_sector,
        npo_consumption_sector=npo_consumption_sector,
        household_consumption_sector=household_consumption_sector,
        gross_investment_sector=gross_investment_sector,
        stocks_investment_sector=stocks_investment_sector,
        final_demand_sector=final_demand_sector,
        total_demand_with_margins_sector=total_demand_with_margins_sector,
        trade_margin_sector=trade_margin_sector,
        transport_margin_sector=transport_margin_sector,
        taxes_sector=taxes_sector,
        imports_sector=imports_sector,
        margins_and_taxes_sector=margins_and_taxes_sector,
        sector_demand_basic_prices=sector_demand_basic_prices,
        final_demand_basic_prices=final_demand_basic_prices,
        value_added_components_sector=value_added_components.copy(),
        value_added_sector=value_added_sector,
        wages_sector=wages_sector,
        profit_sector=profit_sector,
        technical_coefficients_sector=technical_coefficients_sector,
        leontief_matrix=leontief_matrix,
        leontief_inverse=leontief_inverse,
        output_sector=output_sector,
        accounting_gap=accounting_gap,
    )

    if validate:
        component_gap = (
            final_demand_components_sector.sum(axis=1, keepdims=True)
            - final_demand_sector
        )
        total_use_gap = (
            final_demand_sector
            + intermediate_consumption_sector.sum(axis=1, keepdims=True)
            - total_demand_with_margins_sector
        )
        leontief_output = leontief_inverse @ final_demand_sector
        tolerance = 1e-7 * max(1.0, float(np.max(np.abs(output_sector))))

        checks = {
            "componentes da demanda final": component_gap,
            "demanda total por setor": total_use_gap,
            "identidade oferta-utilizacao": accounting_gap,
            "producao setorial da tabela VA": (
                sector_demand_basic_prices - output_sector
            ),
            "fechamento de Leontief": leontief_output - production_sector_sector,
        }
        failures = {
            name: float(np.max(np.abs(gap)))
            for name, gap in checks.items()
            if np.max(np.abs(gap)) > tolerance
        }
        if failures:
            details = ", ".join(
                f"{name}={gap:.6g}" for name, gap in failures.items()
            )
            raise ValueError(f"A TRU setor x setor nao fechou: {details}.")

    return result

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
    
    va_components_names = [
        str(value).strip()
        for value in va.iloc[5 : 5 + 14, 0].fillna("")
    ]

    return TRUData(
        year=year,
        product_names=product_names,
        sector_names=sector_names,
        va_components_names=va_components_names,
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
        exports=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(2, 3),
        ),
        gov_cons=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(3, 4),
        ),
        npo_consumption=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(4, 5),
        ),
        household_consumption=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(5, 6),
        ),
        gross_investment=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(6, 7),
        ),
        stocks_investment=_numeric_block(
            demanda,
            slice(data_start_row, data_start_row + num_products),
            slice(7, 8),
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
        )
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
    
    
    leontief_matrix_2 = (
        identity
        - technical_coefficients_sector    )
    leontief_inverse_2 = np.linalg.inv(leontief_matrix_2)

    final_demand_sector = market_share_domestic.T @ data.final_demand_product
    output_sector = leontief_inverse @ (final_demand_sector - import_final_sector)
    
    output_sector_2 = leontief_inverse_2 @ (final_demand_sector)
    
    output_sector = output_sector_2

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

def build_tru_model_2(data: TRUData) -> TRUModel_2:
    
    production = data.production
    imports = data.imports
    ci = data.intermediate_consumption
    output = data.value_added_components[12, :].reshape(-1, 1)
    final_demand_sector = data.final_demand_product.reshape(-1, 1)
       
    domestic_supply_product = production.sum(axis=0, keepdims=True)
    total_supply_product = domestic_supply_product + imports
    
    
    divisor = production.sum(axis=1) + np.asarray(data.imports).reshape(-1)

    market_share = production / divisor[:, None]
    
       
    print("Produção domésticas:", production)
       
    domestic_supply_product = production.sum(axis=1, keepdims=True)
    total_supply_product = data.final_demand_product_total # Demanda total final do produto, incluindo importações e consumo intermediário. = oferta de mercado à preços de mercado
    ci_sector = ci       

    print("Final demand shape:", final_demand_sector)
    print("Total supply shape:", total_supply_product)
    print("CI shape:", ci_sector.shape)
    print("Taxes shape:", data.taxes.shape)
    
    technical_coefficients_sector = _safe_divide(ci_sector, total_supply_product.T)
    
    tax_coefficients = np.diagflat(_safe_divide(data.taxes, total_supply_product))
    
    trade_margin_coefficients = np.diagflat(
        _safe_divide(data.trade_margin, total_supply_product)
    )
    
    transport_margin_coefficients = np.diagflat(
        _safe_divide(data.transport_margin, total_supply_product)
    )

    identity = np.eye(production.shape[1])
    
    leontief_matrix = identity - technical_coefficients_sector
    
    leontief_inverse = np.linalg.inv(leontief_matrix)

    basic_price_conversion = (
        identity
        - tax_coefficients
        - trade_margin_coefficients
        - transport_margin_coefficients
    )

    output_market_prices = leontief_inverse @ final_demand_sector
    
    print("Output market prices:", output_market_prices)

    output_basic_price = basic_price_conversion @ output_market_prices
    
    #print("Output basic prices:", output_sector)
 
    return TRUModel_2(
        data=data,
        production = production,
        domestic_supply_product=domestic_supply_product,
        total_supply_product=total_supply_product,
        intermediate_consumption_sector=ci_sector,
        technical_coefficients_sector=technical_coefficients_sector,
        tax_coefficients=tax_coefficients,
        trade_margin_coefficients=trade_margin_coefficients,
        transport_margin_coefficients=transport_margin_coefficients,
        leontief_matrix=leontief_matrix,
        leontief_inverse=leontief_inverse,
        final_demand_sector=final_demand_sector,
        output_basic_price=output_basic_price,
        output_market_prices=output_market_prices
    )
