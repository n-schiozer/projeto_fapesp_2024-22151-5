# %%
# IMPORTS
print("Step 1: Startando o processo...")

import pandas as pd
import re
import numpy as np

print("Step 1: leitura OK")



# %%
# LEITURA DOS DADOS
df = pd.read_excel("Demografia_Empresas.xlsx", sheet_name="Planilha1")
print("df existe?", "df" in globals())
print(df.head())

# %%
# LIMPEZA INICIAL
df = df[df["Código CNAE 2.0"].notna()].copy()
print(df.head())
num_setores = df["Código CNAE 2.0"].nunique()
print(f"Número de setores únicos: {num_setores}")



setores = {}

for cnae, dados_setor in df.groupby("Código CNAE 2.0"):
    
    dados_filtrados = dados_setor[dados_setor["limite_inferior"] >= 30]
    
    if not dados_filtrados.empty:
        setores[cnae] = dados_filtrados.copy()
        
        
lim_inferior = setores["A"]["limite_inferior"]


# %%
# FUNÇÃO PARA PEGAR LIMITE INFERIOR DA FAIXA
# Vê se a faixa é do tipo "30 ou mais" ou "10 a 19" e extrai o menor número correspondente
def pega_limite_inferior(faixa): 
    if pd.isna(faixa):
        return None
    
    faixa = str(faixa).strip().lower()
    
    if "ou mais" in faixa:
        num = re.findall(r"\d+", faixa)
        return int(num[0]) if num else None
    
    num = re.findall(r"\d+", faixa)
    return int(num[0]) if num else None

def market_share_previsto(beta, firmas_por_faixa):
    pesos_faixa = []
    
    for firmas in firmas_por_faixa:
        pesos = firmas ** beta
        pesos_faixa.append(pesos.sum())
    
    pesos_faixa = np.array(pesos_faixa)
    ms_prev = pesos_faixa / pesos_faixa.sum()
    
    return ms_prev

def erro_beta(beta, firmas_por_faixa, ms_obs):
    ms_prev = market_share_previsto(beta, firmas_por_faixa)
    return np.sum((ms_obs - ms_prev)**2)


empresas_sinteticas_setor = []

for cnae, df in setores.items():
    print(cnae)
    
    if cnae == "U":
        continue
    
    # Cálculo do market-share por faixa 
    
    sum_setor = df.loc[:, "Salários e outras remunerações"].sum()

    ms = df.loc[:, "Salários e outras remunerações"]/sum_setor

    df["ms"] = ms
    
    print("Dados ms:",df["ms"])

    cdf = ms[::-1].cumsum()[::-1] # Cumulative Distribution Function

    # Regressão para o modelo de Pareto pelo número de funcionários

    cdf = pd.to_numeric(cdf, errors="coerce")
    lim_inferior = pd.to_numeric(lim_inferior, errors="coerce")

    log_x = np.log(lim_inferior)
    log_y = np.log(cdf)

    coef = np.polyfit(log_x, log_y, 1)

    beta_pareto = coef[0]      # inclinação
    const_pareto = coef[1]     # intercepto

    # alpha é o negativo da inclinação
    alpha_pareto = -beta_pareto

    # Criação das empresas sintéticas por faixa
    # Determinação das faixas superiores e inferiores
    
    empresas_sinteticas_faixas = []
    
    for i in range(len(df)):
    
        n_empresas = df["Número de empresas \ne outras \norganizações"].iloc[i]
        
        print(f"Número de empresas na faixa {i} =", n_empresas)
        
        xmin_1 = df["limite_inferior"].iloc[i]
                
        if i == len(df) - 1:
            xmax_1 = 2 * df["Pessoal ocupado total"].iloc[i]/df["Número de empresas \ne outras \norganizações"].iloc[i]
            xmax_1 = np.round(xmax_1)
            
            if pd.isna(xmax_1) or xmax_1 <= xmin_1:
                xmax_1 = 2 * xmin_1
            
        else:
            xmax_1 = df["limite_inferior"].iloc[i+1]

        print(xmax_1)
        
        empresas_sinteticas = pareto_truncada(n_empresas, alpha_pareto, xmin_1, xmax_1)
        
        print("Empresas sintéticas geradas =", empresas_sinteticas)

        total_simulado = empresas_sinteticas.sum()
        total_real = df["Pessoal ocupado total"].iloc[i]

        fator = total_real / total_simulado

        empresas_sinteticas = empresas_sinteticas * fator             

        empresas_sinteticas = np.round(empresas_sinteticas).astype(int)
        
        # 3. diferença
        diff = int(total_real - empresas_sinteticas.sum())
        
        if abs(diff) > len(empresas_sinteticas):
            raise ValueError("Diferença muito grande para ajustar com o número de empresas sintéticas.")
        
        if abs(diff) != 0:   
            # 4. Ajustar a última empresa para corrigir a diferença

            idx = np.random.choice(len(empresas_sinteticas), size=abs(diff), replace=False)

            print("Index=", idx)

            empresas_sinteticas[idx] += int(diff/len(idx))

            # 3. diferença
            diff = int(total_real - empresas_sinteticas.sum())
            
            print(len(empresas_sinteticas))

        empresas_sinteticas_faixas.append(empresas_sinteticas)
           
    
    betas = np.linspace(0.1, 2.0, 200)

    erros = [erro_beta(b, empresas_sinteticas_faixas, ms) for b in betas]

    beta_otimo = betas[np.argmin(erros)]
    
    market_share_sintetico = []
    
    for i in range(len(empresas_sinteticas_faixas)):
        empresas_sinteticas_peso = empresas_sinteticas_faixas[i] ** beta_otimo
        empresas_sinteticas_peso = empresas_sinteticas_peso / empresas_sinteticas_peso.sum() * ms.iloc[i]
        market_share_sintetico.append(empresas_sinteticas_peso)
    
    print("MS;",df["ms"])
    
    # Junta as faixas para o setor e o market-share simulado para o setor
    empresas_sinteticas_faixas = np.concatenate(empresas_sinteticas_faixas)
    market_share_sintetico = np.concatenate(market_share_sintetico)
         
    empresas_sinteticas_setor.append((cnae, empresas_sinteticas_faixas,market_share_sintetico, df["ms"],alpha_pareto,beta_otimo))




len(empresas_sinteticas_setor[16][2])
 
