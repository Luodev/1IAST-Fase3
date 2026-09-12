"""
Enriquecimento da base analítica com dados externos do IBGE.

A camada Gold da Fase 2 é rica em indicadores educacionais, mas não explica o
*contexto* em que a escola opera. O PDF do Tech Challenge autoriza (e incentiva)
o uso de fontes externas; aqui usamos três APIs públicas do IBGE:

  1. Localidades       — região, UF, meso/microrregião de cada município
  2. Censo 2022 (4714) — população residente, área territorial, densidade
  3. Censo 2022 (9543) — taxa de alfabetização da população de 15 anos ou mais
  4. PIB Municipal (5938) — PIB a preços correntes, base do PIB per capita

Todas as respostas são salvas em `data/external/` como CSV. Se o arquivo já
existir, a rede não é acessada — o projeto continua reproduzível offline e o
resultado do treino não muda entre execuções.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.config import EXTERNAL_DIR

BASE_LOCALIDADES = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
BASE_AGREGADOS = "https://servicodados.ibge.gov.br/api/v3/agregados"
TIMEOUT = 180


def _get(url: str, tentativas: int = 3) -> list | dict:
    """GET com retentativa simples — as APIs do IBGE têm picos de latência."""
    erro = None
    for i in range(tentativas):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - queremos registrar e repetir
            erro = exc
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"Falha ao consultar {url}: {erro}")


def _serie_agregado(agregado: int, variavel: int, periodo: str | int) -> pd.DataFrame:
    """Baixa uma variável de um agregado SIDRA para todos os municípios (N6).

    Retorna DataFrame com `id_municipio` (str de 7 dígitos) e `valor` (float).
    Valores não disponíveis no SIDRA vêm como '-', '..' ou 'X' e viram NaN.
    """
    url = (f"{BASE_AGREGADOS}/{agregado}/periodos/{periodo}"
           f"/variaveis/{variavel}?localidades=N6[all]")
    dados = _get(url)

    linhas = []
    for var in dados:
        for res in var["resultados"]:
            for serie in res["series"]:
                cod = serie["localidade"]["id"]
                for _, valor in serie["serie"].items():
                    linhas.append({"id_municipio": str(cod), "valor": valor})

    df = pd.DataFrame(linhas)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.drop_duplicates("id_municipio")


# ---------------------------------------------------------------------------
# Blocos de coleta
# ---------------------------------------------------------------------------

def baixar_localidades(destino: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Hierarquia territorial de todos os 5.570 municípios brasileiros."""
    arquivo = destino / "ibge_localidades.csv"
    if arquivo.exists():
        return pd.read_csv(arquivo, dtype={"id_municipio": str})

    dados = _get(BASE_LOCALIDADES)

    def caminho(no: dict | None, *chaves: str):
        """Navega a árvore aninhada devolvendo None se algum nível faltar.

        Municípios criados após a última revisão da malha (ex.: Boa Esperança
        do Norte/MT) vêm sem o bloco `microrregiao`; nesses casos a hierarquia
        só existe pelo caminho `regiao-imediata`.
        """
        for chave in chaves:
            if not isinstance(no, dict):
                return None
            no = no.get(chave)
        return no

    linhas = []
    for m in dados:
        uf = (caminho(m, "microrregiao", "mesorregiao", "UF")
              or caminho(m, "regiao-imediata", "regiao-intermediaria", "UF")
              or {})
        linhas.append({
            "id_municipio": str(m["id"]),
            "nome_municipio": m["nome"],
            "sigla_uf": uf.get("sigla"),
            "nome_uf": uf.get("nome"),
            "regiao": (uf.get("regiao") or {}).get("nome"),
            "mesorregiao": caminho(m, "microrregiao", "mesorregiao", "nome"),
            "microrregiao": caminho(m, "microrregiao", "nome"),
            "regiao_intermediaria": caminho(m, "regiao-imediata", "regiao-intermediaria", "nome"),
        })

    df = pd.DataFrame(linhas)
    df.to_csv(arquivo, index=False)
    return df


def baixar_demografia(destino: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """População, área e densidade demográfica (Censo 2022, tabela 4714)."""
    arquivo = destino / "ibge_demografia_censo2022.csv"
    if arquivo.exists():
        return pd.read_csv(arquivo, dtype={"id_municipio": str})

    pop = _serie_agregado(4714, 93, 2022).rename(columns={"valor": "populacao_2022"})
    area = _serie_agregado(4714, 6318, 2022).rename(columns={"valor": "area_km2"})
    dens = _serie_agregado(4714, 614, 2022).rename(columns={"valor": "densidade_hab_km2"})

    df = pop.merge(area, on="id_municipio", how="outer").merge(dens, on="id_municipio", how="outer")
    df.to_csv(arquivo, index=False)
    return df


def baixar_alfabetizacao_adulta(destino: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Taxa de alfabetização das pessoas de 15+ anos (Censo 2022, tabela 9543).

    Proxy do capital educacional das famílias: em municípios onde a geração
    adulta tem baixa alfabetização, a criança recebe menos apoio de letramento
    em casa. É uma variável socioeconômica, não um vazamento — mede os adultos
    de 2022, não as crianças avaliadas em 2024.
    """
    arquivo = destino / "ibge_alfabetizacao_adulta_censo2022.csv"
    if arquivo.exists():
        return pd.read_csv(arquivo, dtype={"id_municipio": str})

    df = _serie_agregado(9543, 2513, 2022).rename(columns={"valor": "taxa_alfabetizacao_15mais"})
    df.to_csv(arquivo, index=False)
    return df


def baixar_pib(destino: Path = EXTERNAL_DIR, ano: int = 2021) -> pd.DataFrame:
    """PIB municipal a preços correntes, em mil reais (tabela 5938)."""
    arquivo = destino / f"ibge_pib_{ano}.csv"
    if arquivo.exists():
        return pd.read_csv(arquivo, dtype={"id_municipio": str})

    df = _serie_agregado(5938, 37, ano).rename(columns={"valor": "pib_mil_reais"})
    df["ano_pib"] = ano
    df.to_csv(arquivo, index=False)
    return df


# ---------------------------------------------------------------------------
# Consolidação
# ---------------------------------------------------------------------------

def carregar_contexto_municipal(destino: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Tabela única de contexto territorial e socioeconômico por município."""
    arquivo = destino / "contexto_municipal.csv"
    if arquivo.exists():
        return pd.read_csv(arquivo, dtype={"id_municipio": str})

    df = (
        baixar_localidades(destino)
        .merge(baixar_demografia(destino), on="id_municipio", how="left")
        .merge(baixar_alfabetizacao_adulta(destino), on="id_municipio", how="left")
        .merge(baixar_pib(destino), on="id_municipio", how="left")
    )

    # PIB per capita em reais correntes (PIB vem em milhares).
    df["pib_per_capita"] = (df["pib_mil_reais"] * 1_000 / df["populacao_2022"]).round(2)

    df.to_csv(arquivo, index=False)
    return df


if __name__ == "__main__":
    ctx = carregar_contexto_municipal()
    print(ctx.shape)
    print(ctx.head())
    print(ctx.isna().sum())
