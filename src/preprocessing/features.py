"""
Dicionário de variáveis do modelo.

Separar a *definição* das features da sua *transformação* deixa explícito o que
entra no modelo e por quê — e permite que o teste anti-vazamento
(`build_abt.validar_ausencia_de_vazamento`) rode sobre essa lista antes de
qualquer treino.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Numéricas
# ---------------------------------------------------------------------------

NUMERICAS_HISTORICO = [
    "taxa_alf_2023",              # desempenho da rede municipal no ano-base
    "media_portugues_2023",       # proficiência média em Língua Portuguesa
    "participacao_2023",          # % de alunos que fizeram a avaliação
    "ranking_uf_2023",            # posição do município dentro da UF
    "percentil_uf_2023",          # posição relativa (0 = pior, 1 = melhor)
    "taxa_estadual_2023",         # rede estadual do mesmo município
    "taxa_privada_2023",          # rede privada do mesmo município
    "dif_municipal_estadual_2023",
    "dif_municipal_privada_2023",
]

NUMERICAS_METAS = [
    "meta_2024",
    "meta_2025",
    "meta_2030",
    "inclinacao_meta_anual",       # esforço anual exigido pela trajetória
    "gap_2023_para_meta_2024",     # distância do resultado de 2023 à meta
    "gap_2023_para_meta_2025",
]

NUMERICAS_SOCIOECONOMICAS = [
    "log_populacao",
    "area_km2",
    "densidade_hab_km2",
    "taxa_alfabetizacao_15mais",   # alfabetização adulta (Censo 2022)
    "log_pib_per_capita",
    "dif_crianca_adulto",
    "porte_relativo_na_uf",
]

NUMERICAS_CANDIDATAS = NUMERICAS_HISTORICO + NUMERICAS_METAS + NUMERICAS_SOCIOECONOMICAS

# ---------------------------------------------------------------------------
# Poda de redundância
# ---------------------------------------------------------------------------
# A análise de multicolinearidade (notebook 02, seção 3, reproduzível por
# `build_abt.analisar_redundancia`) encontrou correlações praticamente perfeitas
# entre parte dos candidatos. Foram removidas:
#
#   meta_2030 ................. constante: 80% para todos os municípios
#   inclinacao_meta_anual ..... r = -1,000 com meta_2025 (é (80 - meta_2025)/5)
#   meta_2024 ................. r =  1,000 com meta_2025
#   meta_2025 ................. r =  0,966 com taxa_alf_2023
#   taxa_privada_2023 ......... r =  0,987 com taxa_alf_2023
#   gap_2023_para_meta_2024 ... r =  0,975 com gap_2023_para_meta_2025
#
# As duas últimas remoções merecem nota analítica:
#
#   * As metas municipais do INEP são calculadas a partir da linha de base de
#     2023 — por isso carregam quase a mesma informação que a taxa daquele ano.
#     Manter as duas dividia a importância entre variáveis sinônimas e escondia
#     o desempenho histórico, que é o preditor conceitualmente central. O que
#     sobra da meta e não está na taxa é justamente a distância entre as duas,
#     preservada em `gap_2023_para_meta_2025`.
#
#   * `taxa_privada_2023` é quase idêntica à taxa municipal porque, em
#     municípios pequenos sem rede privada de 2º ano, a base do INEP repete o
#     valor da rede municipal na linha da rede privada. É um artefato de
#     preenchimento, não uma coincidência — e usá-lo como preditor seria
#     duplicar a variável histórica.
#
# A ordem de preferência abaixo resolve cada empate a favor da variável que um
# gestor público entende sem explicação.

PRIORIDADE_INTERPRETABILIDADE = [
    "taxa_alf_2023",
    "gap_2023_para_meta_2025",
    "percentil_uf_2023",
    "taxa_alfabetizacao_15mais",
    "log_pib_per_capita",
    "log_populacao",
    "participacao_2023",
    "densidade_hab_km2",
    "media_portugues_2023",
    "taxa_estadual_2023",
    "dif_municipal_estadual_2023",
    "dif_municipal_privada_2023",
    "dif_crianca_adulto",
    "ranking_uf_2023",
    "area_km2",
    "porte_relativo_na_uf",
]

REMOVIDAS_POR_REDUNDANCIA = [
    "meta_2024", "meta_2025", "meta_2030", "inclinacao_meta_anual",
    "taxa_privada_2023", "gap_2023_para_meta_2024",
]

NUMERICAS = [c for c in NUMERICAS_CANDIDATAS if c not in REMOVIDAS_POR_REDUNDANCIA]

# ---------------------------------------------------------------------------
# Categóricas
# ---------------------------------------------------------------------------

CATEGORICAS = [
    "sigla_uf",                    # 27 categorias
    "regiao",                      # 5 categorias
    "porte_municipio",             # 7 faixas populacionais
    "nivel_alfabetizacao_2023",    # classificação INEP do resultado de 2023
]

FEATURES = NUMERICAS + CATEGORICAS
FEATURES_CANDIDATAS = NUMERICAS_CANDIDATAS + CATEGORICAS

# Colunas de identificação e alvos — nunca entram como preditor.
IDENTIFICADORES = ["id_municipio", "nome_municipio", "regiao_intermediaria"]
ALVOS = ["alfabetizado", "atinge_meta_2025", "taxa_alvo"]

ALVO_PRINCIPAL = "alfabetizado"

# Descrições usadas nos relatórios e nos gráficos de importância.
DESCRICOES = {
    "taxa_alf_2023": "Taxa de alfabetização da rede municipal em 2023 (%)",
    "media_portugues_2023": "Proficiência média em Português em 2023 (pontos Saeb)",
    "participacao_2023": "Participação dos alunos na avaliação de 2023 (%)",
    "ranking_uf_2023": "Posição do município no ranking da UF em 2023",
    "percentil_uf_2023": "Percentil do município dentro da UF em 2023",
    "taxa_estadual_2023": "Taxa de alfabetização da rede estadual local em 2023 (%)",
    "taxa_privada_2023": "Taxa de alfabetização da rede privada local em 2023 (%)",
    "dif_municipal_estadual_2023": "Diferença entre rede municipal e estadual em 2023 (p.p.)",
    "dif_municipal_privada_2023": "Diferença entre rede municipal e privada em 2023 (p.p.)",
    "meta_2024": "Meta de alfabetização pactuada para 2024 (%)",
    "meta_2025": "Meta de alfabetização pactuada para 2025 (%)",
    "meta_2030": "Meta de alfabetização pactuada para 2030 (%)",
    "inclinacao_meta_anual": "Avanço anual exigido entre 2025 e 2030 (p.p./ano)",
    "gap_2023_para_meta_2024": "Distância entre o resultado de 2023 e a meta de 2024 (p.p.)",
    "gap_2023_para_meta_2025": "Distância entre o resultado de 2023 e a meta de 2025 (p.p.)",
    "log_populacao": "Log da população residente (Censo 2022)",
    "area_km2": "Área territorial do município (km²)",
    "densidade_hab_km2": "Densidade demográfica (hab/km²)",
    "taxa_alfabetizacao_15mais": "Alfabetização da população de 15+ anos (Censo 2022, %)",
    "log_pib_per_capita": "Log do PIB per capita municipal (2021)",
    "dif_crianca_adulto": "Diferença entre alfabetização infantil e adulta (p.p.)",
    "porte_relativo_na_uf": "Posição relativa de população dentro da UF (0 a 1)",
    "sigla_uf": "Unidade da Federação",
    "regiao": "Região do país",
    "porte_municipio": "Faixa populacional do município",
    "nivel_alfabetizacao_2023": "Nível de alfabetização atribuído pelo INEP em 2023",
}


def descrever(coluna: str) -> str:
    """Nome legível de uma variável, inclusive as criadas pelo one-hot encoding."""
    if coluna in DESCRICOES:
        return DESCRICOES[coluna]
    for base, texto in DESCRICOES.items():
        if coluna.startswith(f"{base}_"):
            return f"{texto} = {coluna[len(base) + 1:]}"
    return coluna
