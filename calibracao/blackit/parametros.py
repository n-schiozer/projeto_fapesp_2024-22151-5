"""Mapa explícito, bounds e aplicação imutável de theta."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ParametroTheta:
    nome: str
    atual: float
    lower: float
    upper: float
    precision: float
    destino: str
    equacao: str
    interpretacao: str


def especificacoes_parametros(CONFIG, CONFIG_ABM) -> tuple[ParametroTheta, ...]:
    """Retorna theta na única ordem aceita pelo adaptador."""
    return (
        ParametroTheta(
            "taxa_crescimento_populacional",
            float(CONFIG["taxa_crescimento_populacional"]),
            0.0, 0.02, 0.0001, "CONFIG",
            "N_t = N_(t-1) * (1 + g_N)",
            "crescimento exógeno da população",
        ),
        ParametroTheta(
            "taxa_crescimento_demanda_autonoma",
            float(CONFIG.get("taxa_crescimento_demanda_autonoma", 0.0)),
            -0.02, 0.06, 0.0005, "CONFIG",
            "A_real,t = A_real,0 * (1 + g_A)^t, A em {G,I,X}",
            "tendência real comum da demanda autônoma",
        ),
        ParametroTheta(
            "a1", float(CONFIG["a1"]),
            0.02, 0.60, 0.01, "CONFIG",
            "dw = a0 + a1 * ((u_base/u)^a3 - 1)",
            "intensidade da resposta salarial ao desemprego",
        ),
        ParametroTheta(
            "a3", float(CONFIG["a3"]),
            0.05, 1.00, 0.01, "CONFIG",
            "dw = a0 + a1 * ((u_base/u)^a3 - 1)",
            "curvatura da resposta salarial ao desemprego",
        ),
        ParametroTheta(
            "velocidade_ajuste_expectativa_demanda",
            float(CONFIG_ABM["velocidade_ajuste_expectativa_demanda"]),
            0.05, 1.00, 0.01, "CONFIG_ABM",
            "D_e,t = D_e,t-1 + beta_D * (D_t-1 - D_e,t-1)",
            "velocidade adaptativa da expectativa de demanda das firmas",
        ),
        ParametroTheta(
            "lambda_expectativa_precos",
            float(CONFIG_ABM["lambda_expectativa_precos"]),
            0.0, 1.50, 0.05, "CONFIG_ABM",
            "Pc_e,t = Pc_t-1 * (1 + lambda_P * inflacao_setorial_t-1)",
            "extrapolação da inflação setorial esperada",
        ),
        ParametroTheta(
            "velocidade_ajuste_estoques_firmas",
            float(CONFIG_ABM["velocidade_ajuste_estoques_firmas"]),
            0.0, 1.00, 0.01, "CONFIG_ABM",
            "DeltaE_desejado = gamma_E * (E_desejado - E)",
            "velocidade de correção dos estoques das firmas",
        ),
        ParametroTheta(
            "gamma_investimento_capacidade",
            float(CONFIG_ABM["gamma_investimento_capacidade"]),
            0.0, 1.00, 0.01, "CONFIG_ABM",
            "I_liq = gamma_K * (K_desejado - K)",
            "ajuste do investimento industrial ao hiato de capacidade",
        ),
        ParametroTheta(
            "gamma_investimento_retorno",
            float(CONFIG_ABM["gamma_investimento_retorno"]),
            0.0, 1.00, 0.01, "CONFIG_ABM",
            "I_liq = gamma_R * K * r_ajustado_anterior",
            "resposta do investimento das firmas de leilão ao retorno",
        ),
        ParametroTheta(
            "eta_preco_padrao", float(CONFIG_ABM["eta_preco_padrao"]),
            -3.0, -0.20, 0.05, "CONFIG_ABM",
            "atratividade proporcional ao preço relativo^eta_preco",
            "elasticidade-preço da alocação de demanda",
        ),
        ParametroTheta(
            "eta_qualidade_padrao", float(CONFIG_ABM["eta_qualidade_padrao"]),
            0.25, 4.00, 0.05, "CONFIG_ABM",
            "atratividade proporcional à qualidade^eta_qualidade",
            "elasticidade-qualidade da alocação de demanda",
        ),
        ParametroTheta(
            "parametro_markup",
            float(CONFIG_ABM["parametros_markup"]["parametro_markup"]),
            0.0, 0.50, 0.01, "CONFIG_ABM.parametros_markup",
            "mu_t = mu_(t-1) * (1 + gamma_mu * delta_market_share)",
            "velocidade comportamental de ajuste do markup",
        ),
        ParametroTheta(
            "rho_qualidade", float(CONFIG_ABM["rho_qualidade"]),
            0.0, 0.98, 0.01, "CONFIG_ABM",
            "z_q,t = rho_q * z_q,t-1 + epsilon_q,t",
            "persistência da qualidade idiossincrática das firmas",
        ),
        ParametroTheta(
            "sigma_qualidade", float(CONFIG_ABM["sigma_qualidade"]),
            0.005, 0.15, 0.005, "CONFIG_ABM",
            "epsilon_q,t ~ N(0, sigma_q^2)",
            "volatilidade da qualidade idiossincrática das firmas",
        ),
        ParametroTheta(
            "rho_produtividade_idiossincratica",
            float(CONFIG_ABM["rho_produtividade_idiossincratica"]),
            0.0, 0.98, 0.01, "CONFIG_ABM",
            "z_a,t = rho_a * z_a,t-1 + epsilon_a,t",
            "persistência da produtividade idiossincrática das firmas",
        ),
        ParametroTheta(
            "sigma_produtividade_idiossincratica",
            float(CONFIG_ABM["sigma_produtividade_idiossincratica"]),
            0.005, 0.15, 0.005, "CONFIG_ABM",
            "epsilon_a,t ~ N(0, sigma_a^2)",
            "volatilidade da produtividade idiossincrática das firmas",
        ),
    )


def normalizar_theta(theta, CONFIG, CONFIG_ABM) -> tuple[float, ...]:
    """Remove somente ruído numérico da grade e valida os limites de theta."""

    specs = especificacoes_parametros(CONFIG, CONFIG_ABM)
    valores = [float(valor) for valor in theta]
    if len(valores) != len(specs):
        raise ValueError(f"theta deve possuir {len(specs)} elementos.")
    normalizados = []
    for valor, spec in zip(valores, specs, strict=True):
        if not math.isfinite(valor):
            raise ValueError(f"{spec.nome} deve ser finito; recebido {valor}.")
        tolerancia = max(1e-12, abs(spec.precision) * 1e-10)
        if valor < spec.lower - tolerancia or valor > spec.upper + tolerancia:
            raise ValueError(
                f"{spec.nome}={valor} está fora de "
                f"[{spec.lower}, {spec.upper}]."
            )
        normalizados.append(min(max(valor, spec.lower), spec.upper))
    return tuple(normalizados)


def aplicar_theta(theta, CONFIG, CONFIG_ABM):
    """Aplica theta somente a cópias e neutraliza choques experimentais."""
    specs = especificacoes_parametros(CONFIG, CONFIG_ABM)
    valores = normalizar_theta(theta, CONFIG, CONFIG_ABM)
    config = deepcopy(CONFIG)
    config_abm = deepcopy(CONFIG_ABM)
    mapa = dict(zip((item.nome for item in specs), valores, strict=True))

    config["periodos"] = 15
    for nome in (
        "taxa_crescimento_populacional",
        "taxa_crescimento_demanda_autonoma",
        "a1",
        "a3",
    ):
        config[nome] = mapa[nome]
    config["multiplicador_governo"] = 1.0
    config["multiplicador_investimento"] = 1.0
    config["multiplicador_exportacoes"] = 1.0

    config_abm["eta_preco_padrao"] = mapa["eta_preco_padrao"]
    config_abm["eta_qualidade_padrao"] = mapa["eta_qualidade_padrao"]
    for nome in (
        "velocidade_ajuste_expectativa_demanda",
        "lambda_expectativa_precos",
        "velocidade_ajuste_estoques_firmas",
        "gamma_investimento_capacidade",
        "gamma_investimento_retorno",
        "rho_qualidade",
        "sigma_qualidade",
        "rho_produtividade_idiossincratica",
        "sigma_produtividade_idiossincratica",
    ):
        config_abm[nome] = mapa[nome]
    config_abm["parametros_markup"]["parametro_markup"] = mapa[
        "parametro_markup"
    ]
    config_abm["choques_climaticos"]["ativo"] = False
    return config, config_abm


def tabela_parametros(CONFIG, CONFIG_ABM) -> list[dict[str, object]]:
    return [asdict(item) for item in especificacoes_parametros(CONFIG, CONFIG_ABM)]
