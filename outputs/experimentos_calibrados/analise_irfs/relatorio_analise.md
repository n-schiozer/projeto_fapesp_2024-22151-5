# Análise dos experimentos climáticos calibrados

- Simulações por cenário: 100
- Período do choque: 5
- Loss da calibração: 2.0195996445852886
- Maior IRF média absoluta antes do choque: 0

## Síntese por cenário e variável

| Cenário | Variável | Pico médio | Período do pico | Soma pós-choque | IRF final |
|---|---|---:|---:|---:|---:|
| choque_05pct_permanente | Déficit externo / PIB | -0.5753 | 18 | -4.5631 | -0.3239 |
| choque_05pct_permanente | Déficit do governo / PIB | 0.7025 | 25 | 4.9632 | 0.7025 |
| choque_05pct_permanente | Inflação | 2.1433 | 24 | 10.7448 | 2.0400 |
| choque_05pct_permanente | PIB real | -3.5526 | 25 | -33.1826 | -3.5526 |
| choque_05pct_permanente | Desemprego | -0.6092 | 22 | -1.1913 | 0.0913 |
| choque_05pct_temporario | Déficit externo / PIB | -0.0543 | 10 | -0.5079 | -0.0167 |
| choque_05pct_temporario | Déficit do governo / PIB | 0.0740 | 8 | 0.5194 | 0.0013 |
| choque_05pct_temporario | Inflação | 0.1356 | 5 | 0.3539 | 0.0009 |
| choque_05pct_temporario | PIB real | -0.2196 | 10 | -2.7219 | -0.0668 |
| choque_05pct_temporario | Desemprego | -0.0475 | 15 | -0.0599 | -0.0215 |
| choque_10pct_permanente | Déficit externo / PIB | -1.0106 | 18 | -7.1152 | -0.3163 |
| choque_10pct_permanente | Déficit do governo / PIB | 0.9214 | 25 | 6.2733 | 0.9214 |
| choque_10pct_permanente | Inflação | 3.4892 | 24 | 15.4185 | 3.4265 |
| choque_10pct_permanente | PIB real | -5.1410 | 25 | -53.3047 | -5.1410 |
| choque_10pct_permanente | Desemprego | -1.0409 | 22 | -1.4660 | 0.2269 |
| choque_10pct_temporario | Déficit externo / PIB | -0.1086 | 10 | -1.0862 | -0.0282 |
| choque_10pct_temporario | Déficit do governo / PIB | 0.1626 | 8 | 1.1305 | -0.0159 |
| choque_10pct_temporario | Inflação | 0.2052 | 5 | 0.6428 | 0.0023 |
| choque_10pct_temporario | PIB real | -0.4684 | 10 | -5.6406 | -0.1179 |
| choque_10pct_temporario | Desemprego | -0.1167 | 14 | -0.1332 | -0.0664 |
| choque_20pct_permanente | Déficit externo / PIB | -2.9060 | 19 | -21.1957 | -0.5358 |
| choque_20pct_permanente | Déficit do governo / PIB | -1.6832 | 22 | 7.2350 | 0.7726 |
| choque_20pct_permanente | Inflação | 5.3571 | 25 | -0.7437 | 5.3571 |
| choque_20pct_permanente | PIB real | -9.0936 | 25 | -115.8925 | -9.0936 |
| choque_20pct_permanente | Desemprego | 1.7140 | 15 | 9.5033 | 1.1444 |
| choque_20pct_temporario | Déficit externo / PIB | -0.2014 | 10 | -2.2345 | -0.0358 |
| choque_20pct_temporario | Déficit do governo / PIB | 0.3322 | 8 | 2.3182 | -0.0361 |
| choque_20pct_temporario | Inflação | 0.3658 | 17 | 1.1970 | 0.0946 |
| choque_20pct_temporario | PIB real | -0.8871 | 10 | -11.3974 | -0.2499 |
| choque_20pct_temporario | Desemprego | -0.2503 | 14 | -0.2871 | -0.1510 |

## Arquivos produzidos

- `estatisticas_detalhadas`: `estatisticas_irf_detalhadas.csv`
- `horizontes`: `irfs_horizontes.csv`
- `resumo_cenarios`: `resumo_cenarios.csv`
- `comparacao_duracoes`: `comparacao_temporario_permanente.csv`
- `dose_resposta`: `dose_resposta.csv`
- `diagnostico_pre_choque`: `diagnostico_pre_choque.csv`
- `relatorio`: `relatorio_analise.md`
- `manifesto`: `manifesto_analise.json`
- `irf_pib_real`: `irf_pib_real.png`
- `irf_taxa_desemprego`: `irf_taxa_desemprego.png`
- `irf_inflacao`: `irf_inflacao.png`
- `irf_deficit_governo`: `irf_deficit_governo.png`
- `irf_deficit_externo`: `irf_deficit_externo.png`
- `dose_resposta_pib_real`: `dose_resposta_pib_real.png`
- `dose_resposta_taxa_desemprego`: `dose_resposta_taxa_desemprego.png`
- `dose_resposta_inflacao`: `dose_resposta_inflacao.png`
- `dose_resposta_deficit_governo`: `dose_resposta_deficit_governo.png`
- `dose_resposta_deficit_externo`: `dose_resposta_deficit_externo.png`
- `efeito_acumulado_pib_real`: `efeito_acumulado_pib_real.png`
- `efeito_acumulado_taxa_desemprego`: `efeito_acumulado_taxa_desemprego.png`
- `efeito_acumulado_inflacao`: `efeito_acumulado_inflacao.png`
- `efeito_acumulado_deficit_governo`: `efeito_acumulado_deficit_governo.png`
- `efeito_acumulado_deficit_externo`: `efeito_acumulado_deficit_externo.png`

As IRFs do PIB estão em percentual relativo ao benchmark. As demais estão em pontos percentuais relativos ao benchmark. O IC95 refere-se à incerteza Monte Carlo da média; p5--p95 descreve a dispersão das trajetórias.
