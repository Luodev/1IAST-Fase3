"""
Séries temporais: projeção do indicador até 2030 e risco de não cumprir a meta.

Responde à pergunta "como prever municípios que podem não atingir metas
futuras?".

-----------------------------------------------------------------------------
LIMITAÇÃO QUE DEFINE O MÉTODO
-----------------------------------------------------------------------------
O Indicador Criança Alfabetizada só existe desde 2023. A série disponível é:

  * Brasil ............ 3 pontos (2023, 2024, 2025)
  * UF ................ 2 pontos (2023, 2024)
  * Município ......... 2 pontos (2023, 2024)

Com 2 ou 3 observações não há sazonalidade a estimar nem autocorrelação a
modelar — ARIMA e Prophet seriam teatro estatístico. O que faz sentido aqui é
um modelo de tendência explícito e honesto:

  1. **Brasil**: suavização exponencial de Holt (tendência linear), que é o
     menor modelo de série temporal que a quantidade de pontos comporta, com
     uma projeção linear simples como referência.
  2. **Município**: extrapolação da variação observada entre 2023 e 2024, com
     dois ajustes obrigatórios —
       (a) *encolhimento* (shrinkage) da variação individual em direção à média
           da UF, porque um único delta anual tem desvio-padrão de ~17 p.p. e
           projetá-lo cru por seis anos amplifica ruído;
       (b) *amortecimento* (damping) do crescimento ao longo do horizonte, já
           que ganhos de alfabetização são mais fáceis no início e saturam
           perto do teto de 100%.

O resultado não é uma previsão pontual confiável por município — é uma
classificação de ritmo, que é o que a política pública precisa: quem está no
caminho da meta e quem não está.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANO_ALVO, ANO_BASE, METRICS_DIR
from src.data.build_abt import carregar_abt
from src.data.gold_loader import ler_gold, ler_silver

ANOS_PROJECAO = list(range(2025, 2031))

# Trajetória nacional pactuada no Compromisso Nacional Criança Alfabetizada.
METAS_NACIONAIS = {2024: 59.9, 2025: 64.0, 2026: 67.0,
                   2027: 71.0, 2028: 74.0, 2029: 77.0, 2030: 80.0}


# ---------------------------------------------------------------------------
# Série nacional
# ---------------------------------------------------------------------------

def serie_nacional() -> pd.DataFrame:
    """Série observada do Brasil (rede pública) e a trajetória de metas."""
    meta_br = ler_silver("meta_brasil")
    serie = (meta_br[["ano", "taxa_alfabetizacao"]]
             .dropna().drop_duplicates("ano").sort_values("ano")
             .rename(columns={"taxa_alfabetizacao": "taxa_observada"})
             .reset_index(drop=True))
    serie["meta"] = serie["ano"].map(METAS_NACIONAIS)
    return serie


def projetar_nacional(serie: pd.DataFrame | None = None,
                      amortecimento: float = 0.85) -> pd.DataFrame:
    """Projeta o indicador nacional até 2030 sob três cenários.

    `linear` é a reta de mínimos quadrados sobre os pontos observados; `holt` é
    a suavização exponencial com tendência do statsmodels. Com apenas três
    observações, os dois convergem para praticamente a mesma reta — o que já é
    um resultado a reportar: a série é curta demais para que um modelo de série
    temporal acrescente informação à tendência simples.

    Por isso o terceiro cenário, `amortecido`, é o que o relatório trata como
    referência. Ele repete a lógica usada nos municípios: o ganho anual encolhe
    a cada ano projetado, refletindo que os primeiros pontos percentuais de
    alfabetização são os mais fáceis e que existe um teto em 100%. Sem esse
    freio, a reta simples projeta o Brasil acima de 90% em 2030 — resultado que
    nenhum sistema educacional do mundo alcançou em seis anos.
    """
    serie = serie_nacional() if serie is None else serie
    obs = serie.dropna(subset=["taxa_observada"])
    y = obs["taxa_observada"].to_numpy(dtype=float)
    anos_obs = obs["ano"].to_numpy(dtype=int)

    horizonte = [a for a in ANOS_PROJECAO if a > anos_obs.max()]

    # Cenário 1 — tendência linear (mínimos quadrados).
    coef = np.polyfit(anos_obs, y, deg=1)
    linear = np.polyval(coef, horizonte)

    # Cenário 2 — Holt: nível + tendência, sem sazonalidade.
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        modelo = ExponentialSmoothing(y, trend="add", seasonal=None,
                                      initialization_method="estimated").fit()
        holt = np.asarray(modelo.forecast(len(horizonte)))
    except Exception:  # noqa: BLE001 - statsmodels ausente ou série curta demais
        holt = linear

    # Cenário 3 — tendência amortecida a partir do último valor observado.
    atual = float(y[-1])
    amortecido = []
    for i in range(1, len(horizonte) + 1):
        atual = min(100.0, atual + float(coef[0]) * (amortecimento ** i))
        amortecido.append(atual)

    projecao = pd.DataFrame({
        "ano": horizonte,
        "projecao_amortecida": np.round(np.clip(amortecido, 0, 100), 2),
        "projecao_holt": np.round(np.clip(holt, 0, 100), 2),
        "projecao_linear": np.round(np.clip(linear, 0, 100), 2),
    })
    projecao["meta"] = projecao["ano"].map(METAS_NACIONAIS)
    projecao["gap_amortecida_vs_meta"] = (
        projecao["projecao_amortecida"] - projecao["meta"]).round(2)
    projecao["gap_linear_vs_meta"] = (
        projecao["projecao_linear"] - projecao["meta"]).round(2)
    projecao["crescimento_anual_observado"] = round(float(coef[0]), 2)
    return projecao


# ---------------------------------------------------------------------------
# Projeção municipal
# ---------------------------------------------------------------------------

def _delta_encolhido(abt: pd.DataFrame, peso_uf: float = 0.5) -> pd.Series:
    """Variação anual de cada município, puxada em direção à média da sua UF.

    Encolher (shrinkage) é o que impede que um salto atípico de 2023 para 2024 —
    comum em municípios pequenos, onde poucas turmas mudam muito o percentual —
    vire uma projeção absurda para 2030. `peso_uf=0.5` dá metade do peso ao
    comportamento individual e metade ao da UF.
    """
    delta_individual = abt["taxa_alvo"] - abt["taxa_alf_2023"]
    delta_uf = abt.groupby("sigla_uf")["taxa_alvo"].transform("mean") - \
        abt.groupby("sigla_uf")["taxa_alf_2023"].transform("mean")
    return (1 - peso_uf) * delta_individual + peso_uf * delta_uf


def projetar_municipios(abt: pd.DataFrame | None = None, amortecimento: float = 0.85,
                        salvar: bool = True) -> pd.DataFrame:
    """Projeta a taxa de cada município de 2025 a 2030 e classifica o ritmo.

    O fator de amortecimento reduz o ganho a cada ano projetado
    (delta_t = delta * amortecimento^t). Sem ele, um município que subiu 12 p.p.
    em um ano chegaria a 2030 acima de 100% — resultado impossível que
    contaminaria o ranking de prioridades.
    """
    abt = carregar_abt() if abt is None else abt

    proj = abt[["id_municipio", "nome_municipio", "sigla_uf", "regiao",
                "taxa_alf_2023", "taxa_alvo", "meta_2025", "meta_2030"]].copy()
    proj = proj.rename(columns={"taxa_alvo": f"taxa_{ANO_ALVO}",
                                "taxa_alf_2023": f"taxa_{ANO_BASE}"})
    proj["delta_observado"] = (proj[f"taxa_{ANO_ALVO}"] - proj[f"taxa_{ANO_BASE}"]).round(2)
    proj["delta_projetado"] = _delta_encolhido(abt).round(2)

    atual = proj[f"taxa_{ANO_ALVO}"].to_numpy(dtype=float)
    delta = proj["delta_projetado"].to_numpy(dtype=float)

    for i, ano in enumerate(ANOS_PROJECAO, start=1):
        atual = np.clip(atual + delta * (amortecimento ** i), 0, 100)
        proj[f"projecao_{ano}"] = np.round(atual, 2)

    # Classificação de ritmo em relação à trajetória pactuada.
    proj["atinge_meta_2025"] = (proj["projecao_2025"] >= proj["meta_2025"]).astype(int)
    proj["atinge_meta_2030"] = (proj["projecao_2030"] >= proj["meta_2030"]).astype(int)
    proj["gap_2030"] = (proj["projecao_2030"] - proj["meta_2030"]).round(2)

    # Quanto o município precisaria avançar por ano para fechar 2030 na meta.
    proj["ritmo_necessario_ate_2030"] = (
        (proj["meta_2030"] - proj[f"taxa_{ANO_ALVO}"]) / 6).round(2)
    proj["ritmo_projetado"] = (proj["delta_projetado"]).round(2)

    proj["classificacao_ritmo"] = np.select(
        [
            proj["delta_observado"] < 0,
            proj["ritmo_projetado"] >= proj["ritmo_necessario_ate_2030"],
            proj["ritmo_projetado"] >= proj["ritmo_necessario_ate_2030"] * 0.6,
        ],
        ["Em retrocesso", "No ritmo da meta", "Ritmo insuficiente"],
        default="Ritmo critico",
    )

    if salvar:
        proj.to_csv(METRICS_DIR / "projecao_municipios_2030.csv", index=False)

    return proj


def resumo_por_uf(projecao: pd.DataFrame | None = None) -> pd.DataFrame:
    """Consolida a projeção municipal por UF — visão para o gestor estadual."""
    projecao = projetar_municipios(salvar=False) if projecao is None else projecao
    resumo = (
        projecao.groupby(["regiao", "sigla_uf"])
        .agg(municipios=("id_municipio", "size"),
             taxa_2024=(f"taxa_{ANO_ALVO}", "mean"),
             projecao_2030=("projecao_2030", "mean"),
             meta_2030=("meta_2030", "mean"),
             perc_atinge_2030=("atinge_meta_2030", "mean"),
             perc_em_retrocesso=("classificacao_ritmo",
                                 lambda s: (s == "Em retrocesso").mean()))
        .round(3).reset_index()
        .sort_values("perc_atinge_2030")
    )
    return resumo


def evolucao_uf() -> pd.DataFrame:
    """Série observada por UF (Gold: evolucao_temporal), base dos gráficos."""
    return ler_gold("evolucao_temporal").sort_values(["sigla_uf", "ano"])


if __name__ == "__main__":
    print("--- Série nacional ---")
    print(serie_nacional().to_string(index=False))
    print("\n--- Projeção nacional ---")
    print(projetar_nacional().to_string(index=False))
    proj = projetar_municipios()
    print("\n--- Classificação de ritmo ---")
    print(proj["classificacao_ritmo"].value_counts().to_string())
    print("\n--- UFs mais distantes da meta 2030 ---")
    print(resumo_por_uf(proj).head(10).to_string(index=False))
