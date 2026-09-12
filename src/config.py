"""
Configuração central do projeto.

Concentra caminhos, constantes de negócio e parâmetros de modelagem em um
único lugar, para que notebooks e scripts compartilhem exatamente as mesmas
definições (evita divergência entre a análise exploratória e o treino).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # CSVs originais (fonte da Fase 2)
EXTERNAL_DIR = DATA_DIR / "external"  # enriquecimento IBGE
PROCESSED_DIR = DATA_DIR / "processed"
LAKE_DIR = DATA_DIR / "lake"          # data lake local (bronze/silver/gold)

MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"
REPORTS_DIR = ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"

for _d in (EXTERNAL_DIR, PROCESSED_DIR, MODELS_DIR, IMAGES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Regras de negócio
# ---------------------------------------------------------------------------

# Ano cujo resultado queremos prever (alvo) e ano usado como histórico.
ANO_ALVO = 2024
ANO_BASE = 2023

# Código INEP da rede de ensino (0=total, 2=estadual, 3=municipal, 5=privada).
# A rede municipal concentra a matrícula do 2º ano do fundamental e é onde a
# política pública de alfabetização (Compromisso Nacional Criança Alfabetizada)
# efetivamente atua — por isso é a unidade de análise do projeto.
REDE_MUNICIPAL = "3"
REDE_ESTADUAL = "2"
REDE_PRIVADA = "5"
REDE_TOTAL = "0"

# Corte do alvo binário (em % de alunos alfabetizados no município).
#
# Justificativa: a camada Gold traz a série nacional do indicador — o Brasil
# fechou 2024 com 59,2% e a meta nacional pactuada para 2024 era 59,9%.
# Arredondamos para 60%: um município é rotulado ALFABETIZADO quando pelo menos
# 60% de seus alunos alcançam o nível esperado, ou seja, quando ele opera no
# patamar nacional ou acima dele. O corte também deixa as classes equilibradas
# (~57% positivos), o que evita métricas infladas por desbalanceamento.
CORTE_ALFABETIZACAO = 60.0

# Mapa código IBGE (2 primeiros dígitos) -> sigla da UF.
IBGE_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}

UF_REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte",
    "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# Faixas de porte populacional (IBGE) usadas como variável categórica.
FAIXAS_PORTE = [0, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000, float("inf")]
ROTULOS_PORTE = [
    "Ate 5 mil", "5-10 mil", "10-20 mil", "20-50 mil",
    "50-100 mil", "100-500 mil", "Acima de 500 mil",
]

# ---------------------------------------------------------------------------
# Estilo dos gráficos
# ---------------------------------------------------------------------------

FIG_DPI = 120
PALETA = {
    "primaria": "#1f4e79",
    "secundaria": "#e07b39",
    "positiva": "#2e7d32",
    "negativa": "#c62828",
    "neutra": "#6b7280",
}
