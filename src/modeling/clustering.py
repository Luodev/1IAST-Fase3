"""
Segmentação não supervisionada de municípios.

Responde à pergunta de negócio "quais regiões possuem padrões semelhantes?".
Enquanto o modelo supervisionado diz *quem* está em risco, a clusterização diz
*que tipo* de risco é — e tipos diferentes pedem políticas diferentes: um
município pobre e isolado com rede frágil não precisa da mesma intervenção que
um município rico cuja rede municipal ficou para trás da rede privada.

Técnicas:
  * PCA para reduzir a redundância entre as variáveis antes de agrupar e para
    permitir a visualização em duas dimensões;
  * KMeans com escolha de k por cotovelo (inércia) e silhueta;
  * DBSCAN como detector de outliers — municípios que não cabem em nenhum
    perfil e merecem análise caso a caso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline

from src.config import METRICS_DIR, RANDOM_STATE
from src.data.build_abt import carregar_abt
from src.preprocessing.pipeline import construir_preprocessador

# Variáveis do perfil municipal. Diferente do modelo supervisionado, aqui o
# resultado de 2024 PODE entrar: não há alvo a prever, queremos descrever o
# retrato completo do município para agrupar realidades parecidas.
NUMERICAS_PERFIL = [
    "taxa_alvo",                  # taxa de alfabetização em 2024
    "taxa_alf_2023",
    "media_portugues_2023",
    "gap_2023_para_meta_2025",
    "participacao_2023",
    "log_populacao",
    "densidade_hab_km2",
    "log_pib_per_capita",
    "taxa_alfabetizacao_15mais",
    "dif_crianca_adulto",
]
CATEGORICAS_PERFIL = ["regiao"]


def _winsorizar(abt: pd.DataFrame, colunas: list[str],
                inferior: float = 0.01, superior: float = 0.99) -> pd.DataFrame:
    """Trunca cauda a cauda os valores extremos das variáveis do perfil.

    Sem isso, o KMeans gastava clusters inteiros em punhados de municípios
    atípicos — cidades com PIB per capita inflado por royalties, capitais com
    densidade de milhares de habitantes por km². A partir de k = 4 o menor grupo
    ficava com menos de 1% dos municípios, o que não descreve nenhum padrão
    regional e não sustenta política pública.

    Truncar não apaga esses casos da análise: eles continuam no conjunto, apenas
    deixam de esticar a escala, e seguem identificados individualmente pela
    detecção de outliers do DBSCAN mais adiante.
    """
    dados = abt.copy()
    for coluna in colunas:
        if coluna in dados.columns and pd.api.types.is_numeric_dtype(dados[coluna]):
            piso, teto = dados[coluna].quantile([inferior, superior])
            dados[coluna] = dados[coluna].clip(piso, teto)
    return dados


def _matriz_perfil(abt: pd.DataFrame):
    """Aplica o mesmo pré-processamento do modelo supervisionado ao perfil.

    Padronizar é obrigatório: KMeans usa distância euclidiana, e sem escala
    comum a população (na casa dos milhões) dominaria a taxa de alfabetização
    (0 a 100).
    """
    pre = construir_preprocessador(NUMERICAS_PERFIL, CATEGORICAS_PERFIL,
                                   indicar_faltantes=False)
    matriz = pre.fit_transform(_winsorizar(abt, NUMERICAS_PERFIL))
    if hasattr(matriz, "toarray"):
        matriz = matriz.toarray()
    return np.asarray(matriz), pre


def escolher_k(matriz: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    """Avalia k candidatos com três critérios complementares.

    * inércia (cotovelo) — soma das distâncias internas, sempre cai com k;
    * silhueta — quanto cada ponto está mais perto do próprio grupo do que do
      vizinho (maior é melhor);
    * Davies-Bouldin — razão entre dispersão interna e separação (menor é melhor).

    Usar mais de um critério evita escolher k por impressão visual do cotovelo.
    """
    linhas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        rotulos = km.fit_predict(matriz)
        _, tamanhos = np.unique(rotulos, return_counts=True)
        linhas.append({
            "k": k,
            "inercia": round(float(km.inertia_), 2),
            "silhueta": round(float(silhouette_score(matriz, rotulos)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(matriz, rotulos)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(matriz, rotulos)), 1),
            "menor_cluster_perc": round(float(tamanhos.min() / len(rotulos)), 4),
        })
    return pd.DataFrame(linhas)


def _joelho(k: np.ndarray, inercia: np.ndarray) -> int:
    """Ponto de cotovelo: o k mais distante da reta que liga os extremos.

    É a formalização do "olhar o gráfico e achar a dobra" — em vez de escolher a
    olho, medimos a distância perpendicular de cada ponto à reta que vai do
    primeiro ao último k e ficamos com a maior.
    """
    x = (k - k.min()) / (k.max() - k.min())
    y = (inercia - inercia.min()) / (inercia.max() - inercia.min())
    # Reta de (0,1) a (1,0) no espaço normalizado; distância = |x + y - 1| / raiz(2).
    return int(k[np.argmax(np.abs(x + y - 1))])


def _melhor_k(diagnostico: pd.DataFrame, tamanho_minimo: float = 0.02) -> int:
    """Escolhe k combinando cotovelo, silhueta e uma restrição operacional.

    A silhueta neste conjunto fica entre 0,20 e 0,25 para todo k testado — quase
    plana. Isso não é defeito do método: dados socioeconômicos municipais formam
    um gradiente contínuo, não ilhas separadas. Quando a silhueta não discrimina,
    escolhê-la como critério equivale a escolher ruído, e ela sempre puxa para
    k = 2, que é grosseiro demais para orientar política pública.

    Por isso o critério principal é o **cotovelo da inércia**, que responde onde
    parar de ganhar coesão ao dividir mais. Sobre ele aplicamos uma restrição de
    negócio: nenhum perfil pode ter menos de `tamanho_minimo` dos municípios —
    um grupo de 50 cidades não sustenta uma linha de programa, e para esses casos
    existe a detecção de outliers do DBSCAN.
    """
    validos = diagnostico[diagnostico["menor_cluster_perc"] >= tamanho_minimo]
    if validos.empty:
        validos = diagnostico

    joelho = _joelho(validos["k"].to_numpy(), validos["inercia"].to_numpy())
    return joelho if joelho in set(validos["k"]) else int(
        validos.loc[validos["silhueta"].idxmax(), "k"])


def segmentar(k: int | None = None, abt: pd.DataFrame | None = None,
              salvar: bool = True) -> dict:
    """Executa a segmentação completa e devolve perfis interpretáveis."""
    abt = carregar_abt() if abt is None else abt
    matriz, _ = _matriz_perfil(abt)

    diagnostico = escolher_k(matriz)
    if k is None:
        k = _melhor_k(diagnostico)

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    rotulos = kmeans.fit_predict(matriz)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    componentes = pca.fit_transform(matriz)

    resultado = abt[[
        "id_municipio", "nome_municipio", "sigla_uf", "regiao", "taxa_alvo",
        "taxa_alf_2023", "log_pib_per_capita", "pib_per_capita", "populacao_2022",
        "taxa_alfabetizacao_15mais", "gap_2023_para_meta_2025", "alfabetizado",
    ]].copy()
    resultado["cluster"] = rotulos
    resultado["pca_1"] = componentes[:, 0].round(4)
    resultado["pca_2"] = componentes[:, 1].round(4)

    perfis = (
        resultado.groupby("cluster")
        .agg(municipios=("id_municipio", "size"),
             taxa_alfabetizacao_2024=("taxa_alvo", "mean"),
             taxa_alfabetizacao_2023=("taxa_alf_2023", "mean"),
             gap_para_meta_2025=("gap_2023_para_meta_2025", "mean"),
             pib_per_capita=("pib_per_capita", "median"),
             populacao_mediana=("populacao_2022", "median"),
             alfabetizacao_adulta=("taxa_alfabetizacao_15mais", "mean"),
             perc_alfabetizados=("alfabetizado", "mean"))
        .round(2).reset_index()
    )

    # Região predominante e nome legível de cada cluster.
    perfis["regiao_predominante"] = [
        resultado[resultado["cluster"] == c]["regiao"].mode().iloc[0]
        for c in perfis["cluster"]
    ]
    perfis["rotulo"] = _rotular_perfis(perfis)
    resultado = resultado.merge(perfis[["cluster", "rotulo"]], on="cluster", how="left")

    # DBSCAN sobre as componentes principais: identifica municípios atípicos.
    dbscan = DBSCAN(eps=0.8, min_samples=15).fit(componentes)
    resultado["outlier_dbscan"] = (dbscan.labels_ == -1).astype(int)

    saida = {
        "k_escolhido": k,
        "diagnostico_k": diagnostico,
        "municipios": resultado,
        "perfis": perfis,
        "variancia_explicada_pca": pca.explained_variance_ratio_.round(4).tolist(),
        "silhueta_final": round(float(silhouette_score(matriz, rotulos)), 4),
        "n_outliers": int(resultado["outlier_dbscan"].sum()),
    }

    if salvar:
        resultado.to_csv(METRICS_DIR / "clusters_municipios.csv", index=False)
        perfis.to_csv(METRICS_DIR / "clusters_perfis.csv", index=False)
        diagnostico.to_csv(METRICS_DIR / "clusters_diagnostico_k.csv", index=False)

    return saida


def _rotular_perfis(perfis: pd.DataFrame) -> list[str]:
    """Traduz o centro de cada cluster em um nome que um gestor entende.

    O nome combina o patamar de desempenho com o traço de contexto que mais
    distingue o grupo. Se dois clusters acabam com o mesmo nome, a região
    predominante entra como desempate — o rótulo precisa identificar o grupo
    sozinho, sem consulta à tabela.
    """
    def desempenho(taxa: float) -> str:
        if taxa >= 75:
            return "Alto desempenho"
        if taxa >= 62:
            return "Desempenho intermediario"
        if taxa >= 48:
            return "Risco moderado"
        return "Risco critico"

    def contexto(linha: pd.Series) -> str:
        if linha["populacao_mediana"] >= 100_000:
            return "centros urbanos"
        if linha["pib_per_capita"] >= 40_000:
            return "economia forte"
        if linha["pib_per_capita"] <= 20_000:
            return "baixa renda"
        if linha["populacao_mediana"] <= 6_000:
            return "municipios muito pequenos"
        return "interior"

    rotulos = [f"{desempenho(l['taxa_alfabetizacao_2024'])} - {contexto(l)}"
               for _, l in perfis.iterrows()]

    vistos: dict[str, int] = {}
    finais = []
    for rotulo, (_, linha) in zip(rotulos, perfis.iterrows()):
        if rotulos.count(rotulo) > 1:
            rotulo = f"{rotulo} ({linha['regiao_predominante']})"
        vistos[rotulo] = vistos.get(rotulo, 0) + 1
        if vistos[rotulo] > 1:
            rotulo = f"{rotulo} {vistos[rotulo]}"
        finais.append(rotulo)
    return finais


if __name__ == "__main__":
    saida = segmentar()
    print("k escolhido:", saida["k_escolhido"], "| silhueta:", saida["silhueta_final"])
    print(saida["perfis"].to_string(index=False))
