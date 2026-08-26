"""Nomes e posições das contas utilizadas na TRU e na CEI."""

# A CEI possui nomes de colunas repetidos (Entrada/Saída). Por isso usamos
# ``iloc``. A leitura de uma célula sempre segue este padrão:
#
#     cei.iloc[L["conta"], C["setor_e_ou_s"]]
#
# ``e`` = entrada/recebimento do setor institucional.
# ``s`` = saída/pagamento do setor institucional.
# Exemplo: ``cei.iloc[L["ir"], C["ff_s"]]`` é o IR pago pelas firmas
# financeiras; ``cei.iloc[L["ir"], C["governo_e"]]`` é o IR recebido pelo
# governo.

L = {
    "va": 1,
    "salarios": 2,
    "contribuicoes_efetivas": 3,
    "impostos_produtos": 4,
    "outros_impostos": 5,
    "juros": 6,
    "dividendos": 7,
    "ir": 8,
    "contribuicoes_sociais": 9,
    "beneficios": 10,
    "aposentadorias": 11,
    "outras_transferencias": 12,
    "consumo": 13,
    "fbcf": 14,
    "estoques": 15,
    "capacidade": 16,
}

C = {
    "familias_e": 1,
    "familias_s": 2,
    "governo_e": 3,
    "governo_s": 4,
    "ff_e": 5,
    "ff_s": 6,
    "nf_e": 7,
    "nf_s": 8,
    "externo_e": 9,
    "externo_s": 10,
}

# A tabela de valor adicionado, ao contrário da CEI, possui nomes únicos.
# Portanto, o código usa ``.loc[VA[...]]`` em vez de posições numéricas.
VA = {
    "total": "Valor adicionado bruto ( PIB )",
    "remuneracoes": "Remunerações",
    "salarios": "Salários",
    "contribuicoes_efetivas": "Contribuições sociais efetivas",
    "previdencia_oficial": "Previdência oficial /FGTS",
    "previdencia_privada": "Previdência privada",
    "contribuicoes_imputadas": "Contribuições sociais imputadas",
    "eob_mais_misto": "Excedente operacional bruto e rendimento misto bruto",
    "rendimento_misto": "Rendimento misto bruto",
    "eob": "Excedente operacional bruto (EOB)",
    "outros_impostos": "Outros impostos sobre a produção",
    "outros_subsidios": "Outros subsídios à produção",
    "producao": "Valor da produção",
    "ocupacoes": "Fator trabalho (ocupações)",
}

COLUNAS_SETORES = {
    "familias": (C["familias_e"], C["familias_s"]),
    "governo": (C["governo_e"], C["governo_s"]),
    "firmas_financeiras": (C["ff_e"], C["ff_s"]),
    "firmas_nao_financeiras": (C["nf_e"], C["nf_s"]),
    "setor_externo": (C["externo_e"], C["externo_s"]),
}

LINHAS_OBRIGATORIAS = [
    L["salarios"],
    L["contribuicoes_efetivas"],
    L["impostos_produtos"],
    L["outros_impostos"],
    L["juros"],
    L["ir"],
    L["contribuicoes_sociais"],
    L["beneficios"],
    L["aposentadorias"],
    L["outras_transferencias"],
]

# IR e dividendos não entram na base do imposto de renda das firmas.
LINHAS_BASE_IR_FIRMAS = [
    linha
    for linha in range(1, 13)
    if linha not in {L["dividendos"], L["ir"]}
]
