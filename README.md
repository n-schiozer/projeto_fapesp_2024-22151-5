# SFC–IO–ABM Brasil

Modelo macroeconômico multissetorial para o Brasil que combina três camadas:

- **SFC (stock-flow consistent):** estoques e fluxos dos setores institucionais;
- **IO/TRU:** relações intersetoriais e estrutura de oferta e demanda;
- **ABM:** firmas heterogêneas, preços, estoques, qualidade, mercados e demografia empresarial.

Este repositório é uma edição **enxuta e executável**. Ele preserva o código do modelo, os dados necessários à execução, a calibração, o laboratório simples, os dois programas Monte Carlo e todos os outputs já produzidos. Ambientes virtuais, testes, cópias históricas e código legado foram removidos.

## Comece aqui

O arquivo-base é [`rodada_benchmark.py`](rodada_benchmark.py). Ele concentra, no topo do próprio arquivo, todos os controles da rodada e os dicionários completos `CONFIG` e `CONFIG_ABM`. Edite-os para alterar hipóteses macroeconômicas, firmas, mercados, demografia ou choques; o arquivo executa então uma trajetória, imprime as tabelas macro e setorial e abre os gráficos.

```powershell
git clone <URL-DO-REPOSITORIO>
cd sfc-io-abm-brasil

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python rodada_benchmark.py
```

No macOS/Linux, ative o ambiente com `source .venv/bin/activate`. Se a política do PowerShell bloquear a ativação, execute `Set-ExecutionPolicy -Scope Process Bypass` e tente novamente.

## Três formas de usar o modelo

### 1. Benchmark calibrado — recomendação inicial

```powershell
python rodada_benchmark.py
```

Por padrão, o benchmark usa os parâmetros de `calibracao/blackit/outputs/parametros_calibrados.json`, os dados incluídos em `data/` e 25 períodos. Os gráficos aparecem em uma janela do Matplotlib; as tabelas também são impressas no terminal. Para executar literalmente os valores editados — inclusive os parâmetros que normalmente seriam substituídos pela calibração — mude `USAR_PARAMETROS_CALIBRADOS` para `False` no início de `rodada_benchmark.py`.

Para controlar semente ou horizonte diretamente:

```powershell
python experimentos/laboratorio_benchmark_calibrado.py --seed 42 --periodos 25
```

### 2. Laboratório sequencial em arquivo único

```powershell
python laboratorio_abm_regulacao_preco_medio_demografia.py
```

Este é o laboratório didático: toda a configuração, inicialização, ciclo de períodos e construção dos gráficos estão no mesmo arquivo. Altere `CONFIG` e `CONFIG_ABM` no topo do arquivo para experimentar hipóteses de demanda, firmas, mercados, regulação ou demografia. Ele é o melhor ponto de leitura para entender o fluxo completo do modelo.

### 3. Monte Carlo sem experimentos

```powershell
# Checagem rápida: 5 trajetórias de 10 períodos
python experimentos/monte_carlo.py --numero-simulacoes 5 --periodos 10

# Execução padrão: 100 trajetórias de 25 períodos
python experimentos/monte_carlo.py
```

Esse programa repete **apenas o benchmark** com sementes diferentes, sem choques climáticos ou cenários contrafactuais. Ele grava:

- `outputs/monte_carlo/historico_macro.csv` — todas as trajetórias;
- `outputs/monte_carlo/medias_macro.csv` — médias por período;
- `outputs/monte_carlo/pib_emprego_inflacao.png` — gráfico das séries médias.

Use `--processos 1` para execução sequencial ou `--processos 0` (padrão) para escolher automaticamente até quatro processos.

### 4. Monte Carlo com experimentos climáticos

```powershell
# Checagem rápida de todos os cenários
python experimentos/monte_carlo_100.py --numero-simulacoes 5

# Estudo completo: 100 trajetórias por cenário
python experimentos/monte_carlo_100.py --numero-simulacoes 100
```

`monte_carlo_100.py` compara o benchmark com choques de produtividade de 5%, 10% e 20%, temporários e permanentes, mantendo sementes comuns entre cenários. As séries, estatísticas, IRFs e gráficos ficam em `outputs/experimentos_calibrados/`.

## Estrutura

```text
.
├── rodada_benchmark.py                         # entrada mais simples
├── laboratorio_abm_regulacao_preco_medio_demografia.py
│                                                  # laboratório sequencial, um arquivo
├── experimentos/
│   ├── monte_carlo.py                            # benchmark repetido, sem choques
│   ├── monte_carlo_100.py                        # cenários/IRFs climáticos
│   └── laboratorio_benchmark_calibrado.py        # API do benchmark
├── calibracao/
│   ├── calibrar_modelo.py                         # calibração estrutural
│   ├── empirica/                                  # base e momentos empíricos
│   └── blackit/                                   # busca Black-IT e parâmetros calibrados
├── data/                                         # TRU, CEI e demografia necessários
├── outputs/                                      # resultados já gerados
└── agentes/ contabilidade/ macro/ mercados/ ...  # módulos do núcleo
```

## Módulos do modelo

| Módulo | Papel no modelo |
| --- | --- |
| `contabilidade/` | Estrutura da CEI, TRU e identidades contábeis. |
| `inicializacao/` | Carrega dados, prepara a economia inicial e cria os agentes. |
| `agentes/` | Famílias, firmas, governo, setor externo e fornecedores importados. |
| `mercados/` | Alocação da demanda, leilões, preços e atendimento setorial. |
| `financeiro/` | Fluxos financeiros e posições dos agentes. |
| `investimento/` | Formação de capital e investimento das firmas não financeiras. |
| `demografia/` | Coortes, entrada/saída e atributos das firmas. |
| `macro/` | Execução de cada período e montagem da saída inicial. |
| `simulacao/` | API de trajetórias, usada pelo benchmark e pelo Monte Carlo. |
| `resultados/` | Histórico macro, resultados setoriais e consolidação de saídas. |
| `calibracao/` | Parâmetros estruturais, dados empíricos e rotina Black-IT. |
| `experimentos/` | Benchmark, repetição Monte Carlo, choques e análise de IRFs. |
| `diagnosticos/` | Checagens de integridade e ferramentas de inspeção. |

## Lógica econômica e sequência de execução

1. `inicializacao/preparar_modelo_cei.py` lê a TRU e a CEI e constrói as condições iniciais.
2. `calibracao/calibrar_modelo.py` transforma essas condições em parâmetros coerentes do modelo.
3. Em cada período, `macro/executar_periodo.py` coordena demanda, produção, mercados, preços, estoques, investimento e fluxos financeiros.
4. `resultados/resultados_abm.py` consolida os resultados macroeconômicos e setoriais.
5. `simulacao/simular_trajetoria.py` repete o ciclo; os programas em `experimentos/` usam essa API para benchmark, Monte Carlo e contrafactuais.

As configurações principais são dois dicionários. `CONFIG` reúne as hipóteses macro (horizonte, demanda autônoma, inflação, juros, estoques e investimento). `CONFIG_ABM` reúne as hipóteses de firmas e mercados (número de firmas, leilões, regulação, heterogeneidade, demografia e choques). No laboratório de arquivo único ambos ficam explicitamente no início do código.

## Dados, calibração e resultados

Os arquivos em `data/` são parte do repositório porque uma simulação precisa da TRU, da CEI e da planilha de demografia. Se desejar usar arquivos fora do repositório, defina antes da execução:

```powershell
$env:SFC_IO_ABM_DATA_DIR = 'C:\caminho\para\tru\nivel_20'
$env:SFC_IO_ABM_ARQUIVO_CEI = 'C:\caminho\para\CEI2020_adaptado_V1.xlsx'
```

Os parâmetros utilizados nas execuções padrão já estão salvos em `calibracao/blackit/outputs/parametros_calibrados.json`. Para executar a busca de calibração novamente (processo potencialmente demorado), use:

```powershell
python calibracao/blackit/executar.py --help
```

A pasta `outputs/` é versionada deliberadamente. Ela contém os resultados de Monte Carlo, os experimentos calibrados, tabelas, IRFs, gráficos e produtos da calibração/demografia. Novas execuções simples de Monte Carlo usam `outputs/monte_carlo/`, que é ignorada pelo Git para não acumular resultados locais.

## Reprodutibilidade

- Execute a partir da raiz do repositório.
- Use as sementes indicadas nos comandos para reproduzir trajetórias.
- Os cenários em `monte_carlo_100.py` usam as mesmas sementes em todos os tratamentos, permitindo comparação direta das IRFs.
- O ambiente virtual nunca deve ser versionado; instale as dependências a partir de `requirements.txt`.
