"""
Construção da ABT (Analytical Base Table).

A ABT é a tabela que entra na modelagem: uma linha por município, o alvo de
2024 e apenas variáveis conhecidas *antes* do resultado de 2024.

-----------------------------------------------------------------------------
NOTA METODOLÓGICA — sobre o grão da predição
-----------------------------------------------------------------------------
O enunciado pede um modelo que preveja "se um aluno será considerado
alfabetizado ou não alfabetizado". A camada Gold da Fase 2, porém, é agregada:
o menor grão disponível é município x rede x ano — o INEP não publica microdado
por aluno para o Indicador Criança Alfabetizada.

Adotamos então o grão mais fino possível e mantemos a leitura no aluno:

  * cada linha representa o *aluno típico* de um município da rede municipal;
  * o alvo é 1 quando o município alfabetiza pelo menos CORTE_ALFABETIZACAO%
    dos seus alunos, isto é, quando um aluno sorteado ali tem alta chance de
    terminar o 2º ano alfabetizado;
  * a probabilidade prevista pelo modelo é calibrada e validada contra a taxa
    real do município (ver `evaluation/metrics.calibracao`), o que permite lê-la
    como "chance de o aluno daquele município estar alfabetizado".

A limitação está declarada no README e nas limitações do relatório técnico.

-----------------------------------------------------------------------------
REGRAS ANTI-VAZAMENTO (data leakage)
-----------------------------------------------------------------------------
O alvo deriva de `taxa_alfabetizacao` de 2024. Portanto NENHUMA variável medida
na mesma avaliação de 2024 pode entrar como preditor:

  BLOQUEADO  taxa_alfabetizacao (2024)     -> é o próprio alvo
  BLOQUEADO  media_portugues (2024)        -> mesma prova que gerou o alvo
  BLOQUEADO  proporcao_aluno_nivel_* (2024)-> decomposição do alvo
  BLOQUEADO  nivel_alfabetizacao (2024)    -> classificação do resultado de 2024
  BLOQUEADO  gap_meta / status_meta (2024) -> calculados a partir do alvo

  PERMITIDO  qualquer indicador de 2023 (histórico, anterior ao alvo)
  PERMITIDO  metas 2024-2030 — publicadas pelo INEP a partir da linha de base
             de 2023, ou seja, informação disponível antes da prova de 2024
  PERMITIDO  contexto territorial e socioeconômico do Censo 2022 (IBGE)

Além disso, todo ajuste estatístico (imputação, escala, encoding) acontece
dentro do Pipeline do scikit-learn e é aprendido somente no conjunto de treino
— nunca sobre a base completa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ANO_ALVO, ANO_BASE, CORTE_ALFABETIZACAO, FAIXAS_PORTE, PROCESSED_DIR,
    REDE_ESTADUAL, REDE_MUNICIPAL, REDE_PRIVADA, ROTULOS_PORTE,
)
from src.data.gold_loader import ler_gold, ler_silver
from src.data.ibge import carregar_contexto_municipal

ARQUIVO_ABT = PROCESSED_DIR / "abt_alfabetizacao.parquet"

# Colunas que jamais podem virar preditor (usadas no teste automático de
# vazamento em `validar_ausencia_de_vazamento`).
COLUNAS_PROIBIDAS = [
    "taxa_alfabetizacao", "taxa_alfabetizacao_2024", "media_portugues_2024",
    "gap_meta_2025", "status_meta_2025", "nivel_alfabetizacao_2024",
    "alfabetizado", "atinge_meta_2025", "taxa_alvo",
]


# ---------------------------------------------------------------------------
# Blocos de variáveis
# ---------------------------------------------------------------------------

def _historico_rede_municipal(gold_mun: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    """Desempenho da rede municipal no ano-base (2023) — o preditor mais forte
    esperado, já que desempenho educacional é fortemente autocorrelacionado."""
    base = gold_mun[gold_mun["ano"] == ANO_BASE][[
        "id_municipio", "taxa_alfabetizacao", "media_portugues",
        "nivel_alfabetizacao", "percentual_participacao",
    ]].rename(columns={
        "taxa_alfabetizacao": "taxa_alf_2023",
        "media_portugues": "media_portugues_2023",
        "nivel_alfabetizacao": "nivel_alfabetizacao_2023",
        "percentual_participacao": "participacao_2023",
    })

    rank = ranking[ranking["ano"] == ANO_BASE][[
        "id_municipio", "ranking_uf", "municipios_na_uf", "percentil_uf",
    ]].rename(columns={
        "ranking_uf": "ranking_uf_2023",
        "municipios_na_uf": "municipios_na_uf_2023",
        "percentil_uf": "percentil_uf_2023",
    })

    return base.merge(rank, on="id_municipio", how="left")


def _historico_outras_redes(indicador: pd.DataFrame) -> pd.DataFrame:
    """Desempenho das redes estadual e privada do mesmo município em 2023.

    Funciona como controle de contexto: se a rede privada da cidade vai bem e a
    municipal vai mal, o problema é de gestão da rede, não do território.
    """
    base = indicador[(indicador["ano"] == ANO_BASE)
                     & (indicador["rede"].isin([REDE_ESTADUAL, REDE_PRIVADA]))]

    larga = base.pivot_table(index="id_municipio", columns="rede",
                             values="taxa_alfabetizacao", aggfunc="mean")
    larga = larga.rename(columns={REDE_ESTADUAL: "taxa_estadual_2023",
                                  REDE_PRIVADA: "taxa_privada_2023"}).reset_index()

    for col in ("taxa_estadual_2023", "taxa_privada_2023"):
        if col not in larga.columns:
            larga[col] = np.nan

    return larga[["id_municipio", "taxa_estadual_2023", "taxa_privada_2023"]]


def _metas(gold_mun: pd.DataFrame) -> pd.DataFrame:
    """Trajetória de metas pactuada pelo INEP (constante por município)."""
    metas = gold_mun[gold_mun["ano"] == ANO_ALVO][[
        "id_municipio", "meta_2024", "meta_2025", "meta_2030",
    ]].copy()
    # Inclinação anual exigida pela trajetória: o quanto o município precisa
    # avançar por ano entre 2025 e 2030 para cumprir o compromisso.
    metas["inclinacao_meta_anual"] = ((metas["meta_2030"] - metas["meta_2025"]) / 5).round(3)
    return metas


def _contexto_socioeconomico() -> pd.DataFrame:
    """Contexto territorial e socioeconômico do IBGE (Censo 2022 e PIB 2021)."""
    ctx = carregar_contexto_municipal()

    ctx["log_populacao"] = np.log1p(ctx["populacao_2022"])
    ctx["log_pib_per_capita"] = np.log1p(ctx["pib_per_capita"])
    ctx["porte_municipio"] = pd.cut(
        ctx["populacao_2022"], bins=FAIXAS_PORTE, labels=ROTULOS_PORTE, right=False,
    ).astype("object")

    return ctx[[
        "id_municipio", "nome_municipio", "sigla_uf", "regiao",
        "regiao_intermediaria", "populacao_2022", "log_populacao", "area_km2",
        "densidade_hab_km2", "taxa_alfabetizacao_15mais", "pib_per_capita",
        "log_pib_per_capita", "porte_municipio",
    ]]


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------

def construir_abt(salvar: bool = True) -> pd.DataFrame:
    """Monta a ABT completa a partir da camada Gold + enriquecimento externo."""
    gold_mun = ler_gold("alfabetizacao_por_municipio")
    ranking = ler_gold("ranking_municipios")
    indicador = ler_silver("indicador_municipio")

    # ---- Alvo: resultado de 2024 da rede municipal -------------------------
    alvo = gold_mun[gold_mun["ano"] == ANO_ALVO][[
        "id_municipio", "taxa_alfabetizacao", "meta_2025",
    ]].rename(columns={"taxa_alfabetizacao": "taxa_alvo"})
    alvo = alvo.dropna(subset=["taxa_alvo"]).drop_duplicates("id_municipio")

    alvo["alfabetizado"] = (alvo["taxa_alvo"] >= CORTE_ALFABETIZACAO).astype(int)
    # Alvo secundário, usado na aplicação estratégica (não entra no treino).
    alvo["atinge_meta_2025"] = (alvo["taxa_alvo"] >= alvo["meta_2025"]).astype(int)
    alvo = alvo.drop(columns=["meta_2025"])

    # ---- Preditores --------------------------------------------------------
    abt = (
        alvo
        .merge(_historico_rede_municipal(gold_mun, ranking), on="id_municipio", how="inner")
        .merge(_historico_outras_redes(indicador), on="id_municipio", how="left")
        .merge(_metas(gold_mun), on="id_municipio", how="left")
        .merge(_contexto_socioeconomico(), on="id_municipio", how="left")
    )

    # ---- Engenharia de atributos ------------------------------------------
    # Distância do resultado de 2023 para a meta de 2024: mede o tamanho do
    # esforço que o município tinha pela frente no início do ciclo.
    abt["gap_2023_para_meta_2024"] = (abt["taxa_alf_2023"] - abt["meta_2024"]).round(2)
    abt["gap_2023_para_meta_2025"] = (abt["taxa_alf_2023"] - abt["meta_2025"]).round(2)

    # Diferença entre a rede municipal e as demais redes da mesma cidade.
    abt["dif_municipal_estadual_2023"] = (abt["taxa_alf_2023"] - abt["taxa_estadual_2023"]).round(2)
    abt["dif_municipal_privada_2023"] = (abt["taxa_alf_2023"] - abt["taxa_privada_2023"]).round(2)

    # Defasagem entre a alfabetização das crianças e a dos adultos do município:
    # isola o efeito da escola do efeito do capital educacional das famílias.
    abt["dif_crianca_adulto"] = (abt["taxa_alf_2023"] - abt["taxa_alfabetizacao_15mais"]).round(2)

    # Escala de atendimento: quantos alunos, aproximadamente, cada ponto
    # percentual representa. Municípios grandes têm inércia maior.
    abt["porte_relativo_na_uf"] = (
        abt.groupby("sigla_uf")["populacao_2022"].rank(pct=True).round(4)
    )

    abt["nivel_alfabetizacao_2023"] = abt["nivel_alfabetizacao_2023"].astype("object")

    if salvar:
        ARQUIVO_ABT.parent.mkdir(parents=True, exist_ok=True)
        abt.to_parquet(ARQUIVO_ABT, index=False)

    return abt


def carregar_abt(forcar: bool = False) -> pd.DataFrame:
    """Carrega a ABT do disco, construindo-a se ainda não existir."""
    if ARQUIVO_ABT.exists() and not forcar:
        return pd.read_parquet(ARQUIVO_ABT)
    return construir_abt()


# ---------------------------------------------------------------------------
# Verificação automática de vazamento
# ---------------------------------------------------------------------------

def validar_ausencia_de_vazamento(colunas_preditoras: list[str],
                                  abt: pd.DataFrame | None = None,
                                  limite_correlacao: float = 0.95) -> dict:
    """Checa duas condições e levanta erro se alguma falhar.

    1. Nenhuma coluna proibida (alvo ou medida de 2024) está entre os preditores.
    2. Nenhum preditor numérico tem correlação quase perfeita com a taxa de
       2024 — correlação acima de `limite_correlacao` indica que a variável é,
       na prática, o próprio alvo disfarçado.

    Rodar esta função no início do treino transforma a regra anti-vazamento em
    teste executável, em vez de uma promessa no README.
    """
    abt = carregar_abt() if abt is None else abt

    proibidas = sorted(set(colunas_preditoras) & set(COLUNAS_PROIBIDAS))
    if proibidas:
        raise ValueError(f"Vazamento: preditores proibidos presentes -> {proibidas}")

    numericas = [c for c in colunas_preditoras
                 if c in abt.columns and pd.api.types.is_numeric_dtype(abt[c])]
    # Colunas constantes têm desvio zero e geram divisão por zero no cálculo da
    # correlação; elas não podem vazar nada, então saem da checagem.
    numericas = [c for c in numericas if abt[c].nunique(dropna=True) > 1]
    correlacoes = abt[numericas].corrwith(abt["taxa_alvo"]).abs().dropna().sort_values(
        ascending=False)

    suspeitas = correlacoes[correlacoes > limite_correlacao]
    if len(suspeitas):
        raise ValueError(
            f"Vazamento: correlação > {limite_correlacao} com o alvo -> "
            f"{suspeitas.round(3).to_dict()}"
        )

    return {
        "preditores_avaliados": len(colunas_preditoras),
        "colunas_proibidas_encontradas": 0,
        "maior_correlacao_com_alvo": round(float(correlacoes.iloc[0]), 3),
        "variavel_mais_correlacionada": correlacoes.index[0],
    }


def analisar_redundancia(colunas: list[str], abt: pd.DataFrame | None = None,
                         limite: float = 0.95) -> pd.DataFrame:
    """Lista os pares de preditores com correlação absoluta acima do limite.

    Multicolinearidade não derruba o poder preditivo de um modelo, mas arruína a
    interpretação — que é metade do que este projeto entrega. Quando duas
    variáveis carregam a mesma informação, a importância se divide entre elas de
    forma arbitrária e nenhuma das duas aparece como relevante, mesmo que o
    conceito que elas medem seja decisivo.
    """
    import itertools

    abt = carregar_abt() if abt is None else abt
    numericas = [c for c in colunas
                 if c in abt.columns and pd.api.types.is_numeric_dtype(abt[c])
                 and abt[c].nunique(dropna=True) > 1]
    matriz = abt[numericas].corr().abs()

    pares = [
        {"variavel_a": a, "variavel_b": b, "correlacao": round(float(matriz.loc[a, b]), 4)}
        for a, b in itertools.combinations(numericas, 2)
        if matriz.loc[a, b] >= limite
    ]
    constantes = [c for c in colunas
                  if c in abt.columns and abt[c].nunique(dropna=True) <= 1]

    resultado = pd.DataFrame(pares).sort_values("correlacao", ascending=False) \
        if pares else pd.DataFrame(columns=["variavel_a", "variavel_b", "correlacao"])
    resultado.attrs["constantes"] = constantes
    return resultado.reset_index(drop=True)


def podar_redundantes(colunas: list[str], prioridade: list[str],
                      abt: pd.DataFrame | None = None,
                      limite: float = 0.95) -> tuple[list[str], list[dict]]:
    """Remove, de cada par redundante, a variável de menor prioridade.

    `prioridade` é a ordem de preferência declarada por interpretabilidade: entre
    duas variáveis que dizem a mesma coisa, fica a que um gestor entende sem
    explicação. Variáveis constantes saem sempre — não distinguem nada.

    Devolve a lista podada e o registro do que foi removido e por quê, para que a
    decisão apareça no relatório em vez de ficar escondida no código.
    """
    abt = carregar_abt() if abt is None else abt
    ordem = {nome: i for i, nome in enumerate(prioridade)}

    mantidas: list[str] = []
    removidas: list[dict] = []

    for coluna in sorted(colunas, key=lambda c: ordem.get(c, len(prioridade))):
        if coluna not in abt.columns:
            continue
        if pd.api.types.is_numeric_dtype(abt[coluna]) and abt[coluna].nunique(dropna=True) <= 1:
            removidas.append({"variavel": coluna, "motivo": "constante", "par": None,
                              "correlacao": None})
            continue
        if not pd.api.types.is_numeric_dtype(abt[coluna]):
            mantidas.append(coluna)
            continue

        conflito = None
        for mantida in mantidas:
            if not pd.api.types.is_numeric_dtype(abt[mantida]):
                continue
            correlacao = abs(float(abt[[coluna, mantida]].corr().iloc[0, 1]))
            if correlacao >= limite:
                conflito = (mantida, round(correlacao, 4))
                break

        if conflito:
            removidas.append({"variavel": coluna, "motivo": "redundante",
                              "par": conflito[0], "correlacao": conflito[1]})
        else:
            mantidas.append(coluna)

    return mantidas, removidas


if __name__ == "__main__":
    df = construir_abt()
    print("ABT:", df.shape)
    print(df.dtypes)
    print("\nTaxa de positivos (alfabetizado):", round(df["alfabetizado"].mean(), 4))
    print("Taxa de positivos (atinge_meta_2025):", round(df["atinge_meta_2025"].mean(), 4))
    print("\nNulos:\n", df.isna().sum()[lambda s: s > 0])
