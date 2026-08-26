# Dados do projeto

Os caminhos são definidos em `configuracao_projeto.py`. O laboratório usa,
por padrão, `processed/tru/nivel_20/` e
`processed/cei/CEI2020_adaptado_V1.xlsx`; variáveis de ambiente servem apenas
como substituição opcional.

## Dados brutos

| Base | Instituição/fonte | Arquivos | Período | Finalidade | URL |
| --- | --- | --- | --- | --- | --- |
| Tabelas de Recursos e Usos, nível 20 | IBGE | `raw/ibge/tru/20_tab[1-4]_AAAA.xls` | 2010–2021, conforme a tabela | Fonte original da produção, oferta, CI, VA, demanda e índices usados na calibração histórica | Pendente: URL não registrada no código ou nos arquivos encontrados |
| Demografia das Empresas | IBGE | `raw/ibge/demografia_empresas/Demografia_Empresas.xlsx` | Pendente | Número de empresas, pessoal ocupado e remunerações para gerar coortes de firmas | Pendente: URL não registrada |
| IPCA mensal, SGS 433 | Banco Central do Brasil | `raw/bcb/bcb_sgs_433_ipca.json` | Desde 1980; recorte atual a partir de 2010 | Calibração e diagnóstico da inflação | https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json |
| PIB real com ajuste sazonal, SIDRA 1621 | IBGE/SIDRA | `raw/ibge/ibge_sidra_1621_pib_real_sa.json` | Desde 1996; recorte atual a partir de 2010 | Validação empírica do PIB real | https://apisidra.ibge.gov.br/values/t/1621/n1/all/v/584/p/all/c11255/90707/d/v584%202 |
| Taxa anual de desocupação, SIDRA 4562 | IBGE/PNAD Contínua | `raw/ibge/ibge_sidra_4562_desemprego_anual.json` | 2012–2025 | Validação empírica do desemprego | https://apisidra.ibge.gov.br/values/t/4562/n1/all/v/4099/p/all |

## Dados processados

| Base | Arquivos | Finalidade |
| --- | --- | --- |
| TRU convertida para XLSX | `processed/tru/nivel_20/*.xlsx` | Formato efetivamente lido pelo laboratório e pela calibração de 2010–2020 |
| CEI adaptada | `processed/cei/CEI2020_adaptado_V1.xlsx` | Condições institucionais do ano-base; fonte e URL exatas pendentes |
| Versões auxiliares da CEI | `processed/cei/CEI2020_adaptado*.xlsx` | Versões já existentes no projeto, preservadas sem sobrescrita |
| Coortes demográficas | `processed/demografia/coortes_firmas_tru_2020.csv` | Interface derivada para inicialização por coortes |
| Exemplo de calibração demográfica | `processed/demografia/Exemplo_calibracao_TRU_empresas.xlsx` | Diagnóstico e exemplo, não é dado bruto |
| Base empírica normalizada | `processed/empirica/*.csv` | Séries normalizadas, alvos e parâmetros publicados pelo pipeline empírico |
| Metadados empíricos | `metadata/empirica/*.csv` | Catálogo das séries e mapeamento para variáveis do modelo |

Arquivos gerados por diagnósticos e calibrações ficam em `../outputs/`, não
nesta árvore. Nenhum dado bruto é sobrescrito pelo pipeline.
