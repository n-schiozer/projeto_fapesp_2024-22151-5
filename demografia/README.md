# Calibração demográfica das firmas

Este módulo transforma a Demografia das Empresas em coortes que inicializam
`laboratorio_abm_regulacao_preco_medio_demografia.py`. Ele não participa da
dinâmica temporal do ABM: a interface com o restante do modelo é a tabela de
coortes, consumida por `inicializar_firmas.py`.

## Sequência

1. Cada faixa preserva exatamente o número de firmas informado na Demografia.
2. Em cada setor, `alpha` da Pareto é estimado somente nas faixas com limite
   inferior maior ou igual a 30, com shares de remunerações acumulados.
3. Esse mesmo `alpha` setorial é extrapolado para sortear firmas em **todas**
   as faixas observadas. Em particular, `0 a 4` é tratada como o intervalo
   `[1, 5)`: uma empresa observada possui ao menos uma pessoa ocupada.
4. O market share individual é diretamente proporcional ao pessoal ocupado:
   `share_i = L_i / sum(L)`. Portanto, `beta_market_share = 1` e remunerações
   não entram na determinação dos shares.
5. As firmas são ordenadas por faixa crescente e, dentro da faixa, por porte
   crescente. As coortes são blocos consecutivos de **no máximo**
   `tamanho_coorte`; a última pode ser menor, uma coorte pode atravessar faixas
   adjacentes e não há embaralhamento.
6. As coortes sequenciais fecham o emprego observado no nível setorial; isso
   permite acomodar faixas cuja combinação de empresas e emprego é inconsistente
   com seus próprios limites, sem embaralhar a população sintética.
7. Apenas depois disso o emprego das coortes é multiplicado por
   `L_TRU / L_Demografia`, arredondado e fechado por residual na última coorte.
   Os shares, preços e qualidades não são recalculados nessa compatibilização.

O uso da cauda para estimar `alpha` e sua aplicação às faixas menores é uma
**extrapolação explícita da distribuição de cauda**, adotada para preservar a
população inteira de empresas observadas.

Execute os testes com:

```powershell
& 'C:\Users\Pichau\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest demografia_empresas.tests.test_calibrar_firmas_demografia -v
```
