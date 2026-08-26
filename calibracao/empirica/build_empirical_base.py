"""Ponto de entrada da Etapa 1: extração e construção da Base Empírica."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


from configuracao_projeto import DATA_ROOT, OUTPUT_DIR, PROCESSED_DIR, RAW_DIR

EMPIRICAL_DIR = DATA_ROOT

from calibracao.empirica.download_bcb import IPCA_URL, download_ipca
from calibracao.empirica.download_ibge import (
    EXPORTS_CATEGORY,
    EXPORTS_URL,
    GFCF_CATEGORY,
    GFCF_URL,
    GDP_URL,
    GOVERNMENT_CONSUMPTION_CATEGORY,
    GOVERNMENT_CONSUMPTION_URL,
    POPULATION_URL,
    download_population,
    download_quarterly_volume,
    download_real_gdp,
)
from calibracao.empirica.download_pnad import (
    UNEMPLOYMENT_URL,
    download_annual_unemployment,
)
from calibracao.empirica.moments import build_calibration_moments
from calibracao.empirica.transform import (
    annualize_quarterly_rate,
    annual_average_quarterly_index,
    annual_growth,
    compound_monthly_inflation_to_annual,
    compound_monthly_inflation_to_quarterly,
    filter_period,
    quarter_over_quarter_growth,
)


def _series_frame(
    frame: pd.DataFrame,
    series_id: str,
    unit: str,
    frequency: str,
    source: str,
    source_url: str,
    transformation: str,
    model_variable: str,
    status: str,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "series_id", series_id)
    result["unit"] = unit
    result["frequency"] = frequency
    result["source"] = source
    result["source_url"] = source_url
    result["transformation"] = transformation
    result["model_variable"] = model_variable
    result["status"] = status
    return result


def run_pipeline(
    empirical_dir: Path = EMPIRICAL_DIR,
    start: str | None = None,
    end: str | None = None,
    download: bool = True,
) -> dict[str, Path]:
    """Baixa, transforma e publica as interfaces estáveis da Base Empírica."""
    del empirical_dir
    raw = RAW_DIR
    interim = PROCESSED_DIR / "empirica"
    processed = PROCESSED_DIR / "empirica"
    outputs = OUTPUT_DIR / "empirica"
    metadata = DATA_ROOT / "metadata" / "empirica"
    for directory in (raw, interim, processed, outputs):
        directory.mkdir(parents=True, exist_ok=True)

    ipca = filter_period(
        download_ipca(
            raw / "bcb" / "bcb_sgs_433_ipca.json",
            download=download,
        ),
        start,
        end,
    )
    gdp = filter_period(
        download_real_gdp(
            raw / "ibge" / "ibge_sidra_1621_pib_real_sa.json",
            download=download,
        ),
        start,
        end,
    )
    unemployment = filter_period(
        download_annual_unemployment(
            raw / "ibge" / "ibge_sidra_4562_desemprego_anual.json",
            download=download,
        ),
        start,
        end,
    )
    population = filter_period(
        download_population(
            raw / "ibge" / "ibge_sidra_6579_populacao.json",
            download=download,
        ),
        start,
        end,
    )
    government = filter_period(
        download_quarterly_volume(
            raw / "ibge" / "ibge_sidra_1621_consumo_governo_real_sa.json",
            category=GOVERNMENT_CONSUMPTION_CATEGORY,
            download=download,
        ),
        start,
        end,
    )
    gfcf = filter_period(
        download_quarterly_volume(
            raw / "ibge" / "ibge_sidra_1621_fbcf_real_sa.json",
            category=GFCF_CATEGORY,
            download=download,
        ),
        start,
        end,
    )
    exports = filter_period(
        download_quarterly_volume(
            raw / "ibge" / "ibge_sidra_1621_exportacoes_reais_sa.json",
            category=EXPORTS_CATEGORY,
            download=download,
        ),
        start,
        end,
    )
    ipca.to_csv(interim / "ipca_mensal.csv", index=False)
    gdp.to_csv(interim / "pib_real_indice_sa.csv", index=False)
    unemployment.to_csv(interim / "taxa_desemprego_anual.csv", index=False)
    population.to_csv(interim / "populacao_anual.csv", index=False)
    government.to_csv(interim / "consumo_governo_real_indice_sa.csv", index=False)
    gfcf.to_csv(interim / "fbcf_real_indice_sa.csv", index=False)
    exports.to_csv(interim / "exportacoes_reais_indice_sa.csv", index=False)

    ipca_quarterly = compound_monthly_inflation_to_quarterly(ipca)
    ipca_annualized = annualize_quarterly_rate(ipca_quarterly)
    gdp_growth = quarter_over_quarter_growth(gdp)
    inflation_annual = compound_monthly_inflation_to_annual(ipca)
    gdp_annual = annual_average_quarterly_index(gdp)
    gdp_growth_annual = annual_growth(gdp_annual)
    population_growth = annual_growth(population)
    # A tabela 6579 não publica 2010. Não tratar a variação 2009--2011 como
    # crescimento de um único ano.
    anos = population["date"].dt.year
    anos_finais_consecutivos = set(anos.loc[anos.diff().eq(1)].tolist())
    population_growth.loc[
        ~population_growth["date"].dt.year.isin(anos_finais_consecutivos),
        "value",
    ] = float("nan")
    government_annual = annual_average_quarterly_index(government)
    gfcf_annual = annual_average_quarterly_index(gfcf)
    exports_annual = annual_average_quarterly_index(exports)
    government_growth = annual_growth(government_annual)
    gfcf_growth = annual_growth(gfcf_annual)
    exports_growth = annual_growth(exports_annual)

    targets = pd.concat(
        [
            _series_frame(ipca, "ipca_mensal", "%", "monthly", "BCB/SGS 433", IPCA_URL,
                          "Variação mensal publicada.", "inflacao", "ready"),
            _series_frame(ipca_quarterly, "inflacao_trimestral", "%", "quarterly",
                          "BCB/SGS 433", IPCA_URL,
                          "Composição de três taxas mensais: (produto(1+i)-1)*100.",
                          "inflacao", "ready"),
            _series_frame(ipca_annualized, "inflacao_anualizada_trimestral", "%", "quarterly",
                          "BCB/SGS 433", IPCA_URL,
                          "Anualização da inflação trimestral: ((1+i)^4-1)*100.",
                          "inflacao", "ready"),
            _series_frame(inflation_annual, "inflacao_anual", "%", "annual",
                          "BCB/SGS 433", IPCA_URL,
                          "Composição geométrica das doze taxas mensais.",
                          "inflacao", "ready"),
            _series_frame(gdp, "pib_real_indice_sa", "número-índice", "quarterly",
                          "IBGE/SIDRA 1621", GDP_URL,
                          "Índice de volume com ajuste sazonal, conforme publicado.",
                          "pib_real", "model_missing"),
            _series_frame(gdp_growth, "crescimento_pib_real_qoq", "%", "quarterly",
                          "IBGE/SIDRA 1621", GDP_URL,
                          "Variação percentual trimestre contra trimestre anterior.",
                          "pib_real", "model_missing"),
            _series_frame(gdp_annual, "pib_real_indice_anual", "número-índice", "annual",
                          "IBGE/SIDRA 1621", GDP_URL,
                          "Média dos quatro índices trimestrais de volume.",
                          "pib_real", "ready"),
            _series_frame(gdp_growth_annual, "crescimento_pib_real_anual", "%", "annual",
                          "IBGE/SIDRA 1621", GDP_URL,
                          "Variação percentual do índice anual contra o ano anterior.",
                          "pib_real", "ready"),
            _series_frame(unemployment, "taxa_desemprego_anual", "%", "annual",
                          "IBGE/SIDRA 4562", UNEMPLOYMENT_URL,
                          "Taxa anual publicada pela PNAD Contínua.",
                          "taxa_desemprego", "ready"),
            _series_frame(population, "populacao_anual", "pessoas", "annual",
                          "IBGE/SIDRA 6579", POPULATION_URL,
                          "População residente estimada publicada.",
                          "indice_populacao", "ready"),
            _series_frame(population_growth, "crescimento_populacao_anual", "%", "annual",
                          "IBGE/SIDRA 6579", POPULATION_URL,
                          "Variação percentual anual da população estimada.",
                          "indice_populacao", "ready"),
            _series_frame(government_growth, "crescimento_consumo_governo_real_anual", "%", "annual",
                          "IBGE/SIDRA 1621", GOVERNMENT_CONSUMPTION_URL,
                          "Variação anual do índice médio de volume trimestral.",
                          "governo_real", "diagnostic"),
            _series_frame(gfcf_growth, "crescimento_fbcf_real_anual", "%", "annual",
                          "IBGE/SIDRA 1621", GFCF_URL,
                          "Variação anual do índice médio de volume trimestral.",
                          "fbcf_fixa_real", "diagnostic"),
            _series_frame(exports_growth, "crescimento_exportacoes_reais_anual", "%", "annual",
                          "IBGE/SIDRA 1621", EXPORTS_URL,
                          "Variação anual do índice médio de volume trimestral.",
                          "exportacoes_real", "diagnostic"),
        ],
        ignore_index=True,
    ).sort_values(["series_id", "date"])
    targets["date"] = pd.to_datetime(targets["date"]).dt.strftime("%Y-%m-%d")
    targets_path = processed / "model_targets.csv"
    targets.to_csv(targets_path, index=False)

    macro_annual = (
        gdp_annual.rename(columns={"value": "pib_real_indice"})
        .merge(
            gdp_growth_annual.rename(
                columns={"value": "crescimento_pib_real_pct"}
            ),
            on="date",
            how="outer",
        )
        .merge(
            inflation_annual.rename(columns={"value": "inflacao_ipca_pct"}),
            on="date",
            how="outer",
        )
        .merge(
            unemployment.rename(columns={"value": "taxa_desemprego_pct"}),
            on="date",
            how="outer",
        )
        .merge(population.rename(columns={"value": "populacao"}), on="date", how="outer")
        .merge(population_growth.rename(columns={"value": "crescimento_populacao_pct"}), on="date", how="outer")
        .merge(government_growth.rename(columns={"value": "crescimento_consumo_governo_real_pct"}), on="date", how="outer")
        .merge(gfcf_growth.rename(columns={"value": "crescimento_fbcf_real_pct"}), on="date", how="outer")
        .merge(exports_growth.rename(columns={"value": "crescimento_exportacoes_reais_pct"}), on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )
    macro_annual.insert(0, "ano", macro_annual.pop("date").dt.year)
    macro_annual_path = processed / "macro_anual_brasil.csv"
    macro_annual.to_csv(macro_annual_path, index=False)

    parameters_path = processed / "sector_parameters.csv"
    pd.DataFrame(
        columns=[
            "sector_id", "model_sector", "parameter", "value", "unit",
            "source_series", "status", "notes",
        ]
    ).to_csv(parameters_path, index=False)

    moments = build_calibration_moments(targets)
    moments_path = outputs / "empirical_moments.csv"
    moments.to_csv(moments_path, index=False)

    manifest = pd.DataFrame(
        [
            {"artifact": "model_targets", "path": str(targets_path.relative_to(DATA_ROOT.parent)), "rows": len(targets)},
            {"artifact": "macro_anual_brasil", "path": str(macro_annual_path.relative_to(DATA_ROOT.parent)), "rows": len(macro_annual)},
            {"artifact": "sector_parameters", "path": str(parameters_path.relative_to(DATA_ROOT.parent)), "rows": 0},
            {"artifact": "empirical_moments", "path": str(moments_path.relative_to(DATA_ROOT.parent)), "rows": len(moments)},
        ]
    )
    manifest_path = outputs / "run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return {
        "model_targets": targets_path,
        "macro_anual_brasil": macro_annual_path,
        "sector_parameters": parameters_path,
        "empirical_moments": moments_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Data inicial inclusiva (AAAA-MM-DD).")
    parser.add_argument("--end", help="Data final inclusiva (AAAA-MM-DD).")
    parser.add_argument(
        "--offline", action="store_true",
        help="Não baixa dados; reutiliza os JSON já gravados em data/raw/.",
    )
    args = parser.parse_args()
    artifacts = run_pipeline(start=args.start, end=args.end, download=not args.offline)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
