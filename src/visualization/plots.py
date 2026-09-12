"""
Gráficos do projeto.

Cada função gera uma figura, salva em `images/` e devolve o caminho. São usadas
tanto pelos notebooks quanto pelo `run_pipeline.py`, de modo que o relatório e a
apresentação nunca ficam com figuras defasadas em relação ao último treino.

Convenções: título explicando o que o gráfico mostra, eixos rotulados em
português e paleta única definida em `src/config.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sem janela — permite rodar em script e em CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, precision_recall_curve, roc_curve,
)

from src.config import FIG_DPI, IMAGES_DIR, PALETA

plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _salvar(fig, nome: str) -> Path:
    caminho = IMAGES_DIR / f"{nome}.png"
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Análise exploratória
# ---------------------------------------------------------------------------

def distribuicao_alvo(abt: pd.DataFrame, corte: float) -> Path:
    """Histograma da taxa de alfabetização com o corte do alvo destacado."""
    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))

    eixos[0].hist(abt["taxa_alvo"].dropna(), bins=40, color=PALETA["primaria"],
                  edgecolor="white")
    eixos[0].axvline(corte, color=PALETA["negativa"], linestyle="--", linewidth=2,
                     label=f"Corte do alvo ({corte:.0f}%)")
    eixos[0].set_title("Taxa de alfabetização dos municípios em 2024")
    eixos[0].set_xlabel("% de alunos alfabetizados")
    eixos[0].set_ylabel("Municípios")
    eixos[0].legend()

    contagem = abt["alfabetizado"].value_counts().sort_index()
    barras = eixos[1].bar(["Não alfabetizado\n(abaixo do patamar)", "Alfabetizado"],
                          contagem.to_numpy(),
                          color=[PALETA["negativa"], PALETA["positiva"]])
    for barra, valor in zip(barras, contagem.to_numpy()):
        eixos[1].text(barra.get_x() + barra.get_width() / 2, valor,
                      f"{valor}\n({valor / contagem.sum():.1%})",
                      ha="center", va="bottom")
    eixos[1].set_title("Balanceamento do alvo")
    eixos[1].set_ylabel("Municípios")
    eixos[1].set_ylim(0, contagem.max() * 1.2)

    fig.suptitle("Distribuição do alvo — municípios da rede municipal", fontsize=12)
    return _salvar(fig, "01_distribuicao_alvo")


def taxa_por_regiao(abt: pd.DataFrame) -> Path:
    """Boxplot da taxa por região — a desigualdade territorial em um gráfico."""
    ordem = (abt.groupby("regiao")["taxa_alvo"].median().sort_values().index.tolist())
    dados = [abt.loc[abt["regiao"] == r, "taxa_alvo"].dropna() for r in ordem]

    fig, eixo = plt.subplots(figsize=(9, 4.5))
    # `tick_labels` substituiu `labels` no matplotlib 3.9; tratamos os dois para
    # o projeto rodar em ambientes com versões diferentes.
    try:
        caixa = eixo.boxplot(dados, tick_labels=ordem, patch_artist=True, showfliers=False)
    except TypeError:
        caixa = eixo.boxplot(dados, labels=ordem, patch_artist=True, showfliers=False)
    for corpo in caixa["boxes"]:
        corpo.set_facecolor(PALETA["primaria"])
        corpo.set_alpha(0.65)
    for mediana in caixa["medians"]:
        mediana.set_color(PALETA["secundaria"])
        mediana.set_linewidth(2)

    eixo.axhline(abt["taxa_alvo"].mean(), color=PALETA["neutra"], linestyle=":",
                 label=f"Média nacional ({abt['taxa_alvo'].mean():.1f}%)")
    eixo.set_title("Taxa de alfabetização por região (2024)")
    eixo.set_ylabel("% de alunos alfabetizados")
    eixo.legend()
    return _salvar(fig, "02_taxa_por_regiao")


def correlacoes(abt: pd.DataFrame, colunas: list[str]) -> Path:
    """Matriz de correlação entre os preditores numéricos e o alvo contínuo."""
    dados = abt[colunas + ["taxa_alvo"]].corr()

    fig, eixo = plt.subplots(figsize=(11, 9))
    imagem = eixo.imshow(dados, cmap="RdBu_r", vmin=-1, vmax=1)
    eixo.set_xticks(range(len(dados)), dados.columns, rotation=90, fontsize=8)
    eixo.set_yticks(range(len(dados)), dados.columns, fontsize=8)
    eixo.grid(False)

    for i in range(len(dados)):
        for j in range(len(dados)):
            valor = dados.iloc[i, j]
            if abs(valor) >= 0.3:
                eixo.text(j, i, f"{valor:.2f}", ha="center", va="center",
                          fontsize=7, color="white" if abs(valor) > 0.6 else "black")

    fig.colorbar(imagem, ax=eixo, shrink=0.7, label="Correlação de Pearson")
    eixo.set_title("Correlação entre preditores e taxa de alfabetização de 2024")
    return _salvar(fig, "03_matriz_correlacao")


def dispersao_historico(abt: pd.DataFrame, corte: float) -> Path:
    """Taxa de 2023 x taxa de 2024 — mostra persistência e reversão à média."""
    fig, eixo = plt.subplots(figsize=(7, 6))
    eixo.scatter(abt["taxa_alf_2023"], abt["taxa_alvo"], s=6, alpha=0.25,
                 color=PALETA["primaria"])
    eixo.plot([0, 100], [0, 100], color=PALETA["neutra"], linestyle="--",
              label="Estabilidade (2024 = 2023)")
    eixo.axhline(corte, color=PALETA["negativa"], linestyle=":",
                 label=f"Corte do alvo ({corte:.0f}%)")

    correlacao = abt["taxa_alf_2023"].corr(abt["taxa_alvo"])
    eixo.set_title(f"Persistência do desempenho entre 2023 e 2024 (r = {correlacao:.2f})")
    eixo.set_xlabel("Taxa de alfabetização em 2023 (%)")
    eixo.set_ylabel("Taxa de alfabetização em 2024 (%)")
    eixo.legend()
    return _salvar(fig, "04_dispersao_2023_2024")


def socioeconomico_vs_alfabetizacao(abt: pd.DataFrame) -> Path:
    """Relação do alvo com PIB per capita e alfabetização adulta."""
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))

    eixos[0].scatter(abt["pib_per_capita"], abt["taxa_alvo"], s=6, alpha=0.2,
                     color=PALETA["primaria"])
    eixos[0].set_xscale("log")
    eixos[0].set_xlabel("PIB per capita (R$, escala log)")
    eixos[0].set_ylabel("Taxa de alfabetização em 2024 (%)")
    eixos[0].set_title("Renda do município e alfabetização")

    eixos[1].scatter(abt["taxa_alfabetizacao_15mais"], abt["taxa_alvo"], s=6,
                     alpha=0.2, color=PALETA["secundaria"])
    eixos[1].set_xlabel("Alfabetização da população de 15+ anos (%)")
    eixos[1].set_ylabel("Taxa de alfabetização em 2024 (%)")
    correlacao = abt["taxa_alfabetizacao_15mais"].corr(abt["taxa_alvo"])
    eixos[1].set_title(f"Alfabetização adulta e infantil (r = {correlacao:.2f})")

    fig.suptitle("Contexto socioeconômico e desempenho educacional", fontsize=12)
    return _salvar(fig, "05_socioeconomico")


# ---------------------------------------------------------------------------
# Avaliação do modelo
# ---------------------------------------------------------------------------

def comparacao_modelos(ranking: pd.DataFrame) -> Path:
    """ROC AUC de validação x treino — leitura direta de overfitting."""
    fig, eixo = plt.subplots(figsize=(9, 4.5))
    posicoes = np.arange(len(ranking))
    largura = 0.38

    eixo.barh(posicoes + largura / 2, ranking["roc_auc_treino"], largura,
              label="Treino", color=PALETA["neutra"], alpha=0.7)
    eixo.barh(posicoes - largura / 2, ranking["roc_auc_val"], largura,
              xerr=ranking["roc_auc_desvio"], label="Validação cruzada",
              color=PALETA["primaria"], capsize=3)

    for i, valor in enumerate(ranking["roc_auc_val"]):
        eixo.text(valor + 0.008, i - largura / 2, f"{valor:.3f}", va="center", fontsize=9)

    eixo.set_yticks(posicoes, ranking["modelo"])
    eixo.set_xlim(0.5, 1.0)
    eixo.set_xlabel("ROC AUC")
    eixo.set_title("Comparação dos algoritmos candidatos")
    eixo.legend(loc="lower right")
    eixo.invert_yaxis()
    return _salvar(fig, "06_comparacao_modelos")


def curvas_de_desempenho(y_true, y_prob) -> Path:
    """ROC, precisão-revocação e matriz de confusão em um painel."""
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.3))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    eixos[0].plot(fpr, tpr, color=PALETA["primaria"], linewidth=2,
                  label=f"Modelo (AUC = {auc:.3f})")
    eixos[0].plot([0, 1], [0, 1], "--", color=PALETA["neutra"], label="Aleatório")
    eixos[0].set_xlabel("Taxa de falso positivo")
    eixos[0].set_ylabel("Taxa de verdadeiro positivo")
    eixos[0].set_title("Curva ROC")
    eixos[0].legend(loc="lower right")

    precisao, revocacao, _ = precision_recall_curve(y_true, y_prob)
    eixos[1].plot(revocacao, precisao, color=PALETA["secundaria"], linewidth=2)
    eixos[1].axhline(np.mean(y_true), linestyle="--", color=PALETA["neutra"],
                     label=f"Base ({np.mean(y_true):.2f})")
    eixos[1].set_xlabel("Revocação")
    eixos[1].set_ylabel("Precisão")
    eixos[1].set_title("Curva Precisão-Revocação")
    eixos[1].legend()

    ConfusionMatrixDisplay.from_predictions(
        y_true, (np.asarray(y_prob) >= 0.5).astype(int), ax=eixos[2],
        display_labels=["Risco", "Alfabetizado"], colorbar=False, cmap="Blues")
    eixos[2].set_title("Matriz de confusão (limiar 0,50)")
    eixos[2].grid(False)

    fig.suptitle("Desempenho do modelo no conjunto de teste", fontsize=12)
    return _salvar(fig, "07_curvas_desempenho")


def curva_de_calibracao(tabela: pd.DataFrame) -> Path:
    """Probabilidade prevista x frequência observada."""
    fig, eixo = plt.subplots(figsize=(6, 5.5))
    eixo.plot([0, 1], [0, 1], "--", color=PALETA["neutra"], label="Calibração perfeita")
    eixo.plot(tabela["prob_media_prevista"], tabela["frequencia_real"],
              marker="o", color=PALETA["primaria"], linewidth=2, label="Modelo calibrado")

    for _, linha in tabela.iterrows():
        eixo.annotate(f"n={int(linha['n'])}",
                      (linha["prob_media_prevista"], linha["frequencia_real"]),
                      textcoords="offset points", xytext=(6, -10), fontsize=7,
                      color=PALETA["neutra"])

    eixo.set_xlabel("Probabilidade média prevista")
    eixo.set_ylabel("Frequência real de municípios alfabetizados")
    eixo.set_title("Calibração — a probabilidade pode ser lida como risco")
    eixo.legend()
    return _salvar(fig, "08_calibracao")


def curva_de_aprendizado(curva: pd.DataFrame) -> Path:
    """Desempenho em função do volume de treino."""
    fig, eixo = plt.subplots(figsize=(7, 4.5))
    eixo.plot(curva["n_treino"], curva["roc_auc_treino"], marker="o",
              color=PALETA["secundaria"], label="Treino")
    eixo.fill_between(curva["n_treino"],
                      curva["roc_auc_treino"] - curva["roc_auc_treino_desvio"],
                      curva["roc_auc_treino"] + curva["roc_auc_treino_desvio"],
                      alpha=0.15, color=PALETA["secundaria"])
    eixo.plot(curva["n_treino"], curva["roc_auc_validacao"], marker="s",
              color=PALETA["primaria"], label="Validação cruzada")
    eixo.fill_between(curva["n_treino"],
                      curva["roc_auc_validacao"] - curva["roc_auc_validacao_desvio"],
                      curva["roc_auc_validacao"] + curva["roc_auc_validacao_desvio"],
                      alpha=0.15, color=PALETA["primaria"])

    eixo.set_xlabel("Municípios usados no treino")
    eixo.set_ylabel("ROC AUC")
    eixo.set_title("Curva de aprendizado — viés x variância")
    eixo.legend()
    return _salvar(fig, "09_curva_aprendizado")


def importancia_variaveis(importancia: pd.DataFrame, top: int = 15,
                          coluna: str = "queda_roc_auc",
                          titulo: str = "Importância por permutação",
                          nome: str = "10_importancia_permutacao") -> Path:
    """Barras horizontais das variáveis mais influentes."""
    dados = importancia.head(top).iloc[::-1]

    fig, eixo = plt.subplots(figsize=(9, 0.42 * len(dados) + 1.8))
    eixo.barh(dados["descricao"].str.slice(0, 58), dados[coluna],
              color=PALETA["primaria"], alpha=0.85)
    if "desvio" in dados.columns:
        eixo.errorbar(dados[coluna], range(len(dados)), xerr=dados["desvio"],
                      fmt="none", ecolor=PALETA["neutra"], capsize=3)
    eixo.set_xlabel(coluna.replace("_", " "))
    eixo.set_title(titulo)
    return _salvar(fig, nome)


def limiar_e_custo(varredura: pd.DataFrame, limiar_escolhido: float) -> Path:
    """Custo de política pública em função do limiar de decisão."""
    fig, eixo = plt.subplots(figsize=(8, 4.5))
    eixo.plot(varredura["limiar"], varredura["custo_total"],
              color=PALETA["primaria"], linewidth=2, label="Custo total ponderado")
    eixo.axvline(limiar_escolhido, color=PALETA["negativa"], linestyle="--",
                 label=f"Limiar escolhido ({limiar_escolhido:.2f})")
    eixo.axvline(0.5, color=PALETA["neutra"], linestyle=":", label="Padrão (0,50)")

    eixo_2 = eixo.twinx()
    eixo_2.plot(varredura["limiar"], varredura["f1"], color=PALETA["secundaria"],
                alpha=0.8, label="F1")
    eixo_2.set_ylabel("F1", color=PALETA["secundaria"])
    eixo_2.grid(False)

    eixo.set_xlabel("Limiar de decisão")
    eixo.set_ylabel("Custo ponderado (3x deixar município em risco sem apoio)")
    eixo.set_title("Escolha do limiar sob a ótica da política pública")
    eixo.legend(loc="upper center")
    return _salvar(fig, "11_limiar_custo")


# ---------------------------------------------------------------------------
# Clusterização
# ---------------------------------------------------------------------------

def diagnostico_k(diagnostico: pd.DataFrame, k_escolhido: int) -> Path:
    """Cotovelo, silhueta e Davies-Bouldin lado a lado."""
    fig, eixos = plt.subplots(1, 3, figsize=(14, 4))
    metricas = [("inercia", "Inércia (cotovelo)", PALETA["primaria"]),
                ("silhueta", "Silhueta (maior = melhor)", PALETA["positiva"]),
                ("davies_bouldin", "Davies-Bouldin (menor = melhor)", PALETA["secundaria"])]

    for eixo, (coluna, titulo, cor) in zip(eixos, metricas):
        eixo.plot(diagnostico["k"], diagnostico[coluna], marker="o", color=cor)
        eixo.axvline(k_escolhido, color=PALETA["negativa"], linestyle="--",
                     label=f"k = {k_escolhido}")
        eixo.set_xlabel("Número de clusters (k)")
        eixo.set_title(titulo)
        eixo.legend()

    fig.suptitle("Escolha do número de perfis municipais", fontsize=12)
    return _salvar(fig, "12_diagnostico_clusters")


def mapa_clusters(municipios: pd.DataFrame, perfis: pd.DataFrame) -> Path:
    """Municípios projetados nas duas primeiras componentes principais."""
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5.5))
    cores = plt.cm.tab10(np.linspace(0, 1, len(perfis)))

    for i, (cluster, bloco) in enumerate(municipios.groupby("cluster")):
        rotulo = perfis.loc[perfis["cluster"] == cluster, "rotulo"].iloc[0]
        eixos[0].scatter(bloco["pca_1"], bloco["pca_2"], s=7, alpha=0.45,
                         color=cores[i], label=f"{cluster}: {rotulo}")
    eixos[0].set_xlabel("Componente principal 1")
    eixos[0].set_ylabel("Componente principal 2")
    eixos[0].set_title("Perfis municipais no espaço das componentes principais")
    eixos[0].legend(fontsize=7, loc="upper left", markerscale=2, framealpha=0.9)

    ordenado = perfis.sort_values("taxa_alfabetizacao_2024").reset_index(drop=True)
    eixos[1].barh(range(len(ordenado)), ordenado["taxa_alfabetizacao_2024"],
                  color=[cores[int(c)] for c in ordenado["cluster"]])
    # O nome do perfil vai dentro da barra: fora dela, rótulos longos invadiriam
    # o gráfico vizinho.
    for i, linha in ordenado.iterrows():
        eixos[1].text(1.5, i, linha["rotulo"], va="center", fontsize=9,
                      color="white", fontweight="bold")
        eixos[1].text(linha["taxa_alfabetizacao_2024"] + 1.5, i,
                      f"{int(linha['municipios'])} municípios", va="center", fontsize=8,
                      color=PALETA["neutra"])
    eixos[1].set_yticks(range(len(ordenado)), [""] * len(ordenado))
    eixos[1].set_xlabel("Taxa média de alfabetização em 2024 (%)")
    eixos[1].set_title("Desempenho médio por perfil")
    eixos[1].set_xlim(0, 100)

    return _salvar(fig, "13_clusters")


# ---------------------------------------------------------------------------
# Séries temporais
# ---------------------------------------------------------------------------

def projecao_nacional(serie: pd.DataFrame, projecao: pd.DataFrame) -> Path:
    """Série observada, projeção e trajetória de metas do Brasil."""
    fig, eixo = plt.subplots(figsize=(9, 5))

    observado = serie.dropna(subset=["taxa_observada"])
    eixo.plot(observado["ano"], observado["taxa_observada"], marker="o",
              linewidth=2.5, color=PALETA["primaria"], label="Observado")

    ultimo = observado["taxa_observada"].iloc[-1]
    ponte_ano = [observado["ano"].iloc[-1]] + projecao["ano"].tolist()

    eixo.plot(ponte_ano, [ultimo] + projecao["projecao_amortecida"].tolist(),
              marker="s", linestyle="--", color=PALETA["secundaria"],
              label="Projeção amortecida (cenário de referência)")
    eixo.plot(ponte_ano, [ultimo] + projecao["projecao_linear"].tolist(),
              marker="^", linestyle=":", color=PALETA["neutra"],
              label="Projeção por tendência linear")

    metas = pd.concat([serie[["ano", "meta"]], projecao[["ano", "meta"]]]).dropna()
    metas = metas.drop_duplicates("ano").sort_values("ano")
    eixo.plot(metas["ano"], metas["meta"], marker="d", color=PALETA["positiva"],
              linewidth=2, label="Meta pactuada")

    eixo.set_xlabel("Ano")
    eixo.set_ylabel("% de alunos alfabetizados (rede pública)")
    eixo.set_title("Brasil: indicador observado, projeção e trajetória de metas")
    eixo.legend()
    return _salvar(fig, "14_projecao_nacional")


def ritmo_municipios(projecao: pd.DataFrame) -> Path:
    """Distribuição da classificação de ritmo e das UFs mais distantes da meta."""
    fig, eixos = plt.subplots(1, 2, figsize=(14, 4.8))

    ordem = ["No ritmo da meta", "Ritmo insuficiente", "Ritmo critico", "Em retrocesso"]
    cores = [PALETA["positiva"], PALETA["secundaria"], "#b45309", PALETA["negativa"]]
    contagem = projecao["classificacao_ritmo"].value_counts().reindex(ordem).fillna(0)

    barras = eixos[0].bar(range(len(ordem)), contagem.to_numpy(), color=cores)
    eixos[0].set_xticks(range(len(ordem)),
                        [o.replace(" ", "\n", 1) for o in ordem], fontsize=9)
    for barra, valor in zip(barras, contagem.to_numpy()):
        eixos[0].text(barra.get_x() + barra.get_width() / 2, valor,
                      f"{int(valor)}\n({valor / contagem.sum():.0%})",
                      ha="center", va="bottom", fontsize=9)
    eixos[0].set_ylabel("Municípios")
    eixos[0].set_ylim(0, contagem.max() * 1.25)
    eixos[0].set_title("Ritmo projetado em relação à meta de 2030")

    por_uf = (projecao.groupby("sigla_uf")["atinge_meta_2030"].mean()
              .sort_values().head(15) * 100)
    eixos[1].barh(por_uf.index, por_uf.to_numpy(), color=PALETA["negativa"], alpha=0.85)
    eixos[1].set_xlabel("% de municípios que atingem a meta de 2030 na projeção")
    eixos[1].set_title("15 UFs mais distantes da meta de 2030")

    fig.suptitle("Projeção do indicador por município até 2030", fontsize=12)
    return _salvar(fig, "15_ritmo_municipios")


# ---------------------------------------------------------------------------
# Aprendizado por reforço
# ---------------------------------------------------------------------------

def convergencia_rl(historico: pd.DataFrame) -> Path:
    """Arrependimento acumulado e taxa de acerto dos agentes."""
    fig, eixos = plt.subplots(1, 2, figsize=(13, 4.5))
    cores = {"Epsilon-Greedy": PALETA["secundaria"], "UCB": PALETA["primaria"],
             "Thompson Sampling": PALETA["positiva"]}

    for agente, bloco in historico.groupby("agente"):
        cor = cores.get(agente, PALETA["neutra"])
        eixos[0].plot(bloco["rodada"], bloco["regret_acumulado"], label=agente, color=cor)
        eixos[1].plot(bloco["rodada"], bloco["taxa_acerto"], label=agente, color=cor)

    eixos[0].set_xlabel("Rodadas de aplicação")
    eixos[0].set_ylabel("Arrependimento acumulado (p.p. não ganhos)")
    eixos[0].set_title("Custo do aprendizado por agente")
    eixos[0].legend()

    eixos[1].axhline(0.25, linestyle="--", color=PALETA["neutra"],
                     label="Escolha aleatória (1 em 4)")
    eixos[1].set_xlabel("Rodadas de aplicação")
    eixos[1].set_ylabel("Proporção de escolhas ótimas")
    eixos[1].set_title("Convergência para a intervenção correta")
    eixos[1].legend()

    fig.suptitle("Aprendizado por reforço na escolha da intervenção", fontsize=12)
    return _salvar(fig, "16_convergencia_rl")


def politica_rl(politica: pd.DataFrame) -> Path:
    """Intervenção recomendada e retorno por unidade de custo em cada perfil."""
    fig, eixo = plt.subplots(figsize=(10, 0.6 * len(politica) + 2))
    ordenado = politica.sort_values("ganho_por_unidade_de_custo")

    barras = eixo.barh(ordenado["rotulo"].str.slice(0, 42),
                       ordenado["ganho_por_unidade_de_custo"],
                       color=PALETA["primaria"], alpha=0.85)
    for barra, (_, linha) in zip(barras, ordenado.iterrows()):
        eixo.text(barra.get_width() + 0.06, barra.get_y() + barra.get_height() / 2,
                  linha["intervencao_recomendada"], va="center", fontsize=9,
                  color=PALETA["neutra"])

    eixo.set_xlabel("Ganho esperado em p.p. por unidade de orçamento")
    eixo.set_title("Política aprendida: qual intervenção enviar a cada perfil")
    eixo.set_xlim(0, ordenado["ganho_por_unidade_de_custo"].max() * 2.6)
    return _salvar(fig, "17_politica_rl")


def ranking_risco(ranking: pd.DataFrame, top: int = 20) -> Path:
    """Municípios com maior risco educacional previsto."""
    dados = ranking.head(top).iloc[::-1]
    rotulos = dados["nome_municipio"] + " (" + dados["sigla_uf"] + ")"

    fig, eixo = plt.subplots(figsize=(9, 0.35 * top + 1.8))
    eixo.barh(rotulos, dados["risco"], color=PALETA["negativa"], alpha=0.85)
    for i, (_, linha) in enumerate(dados.iterrows()):
        eixo.text(linha["risco"] + 0.008, i, f"taxa 2024: {linha['taxa_alvo']:.1f}%",
                  va="center", fontsize=8, color=PALETA["neutra"])

    eixo.set_xlabel("Risco previsto (1 - probabilidade de alcançar o patamar)")
    eixo.set_title(f"{top} municípios com maior risco educacional previsto")
    eixo.set_xlim(0, 1.25)
    return _salvar(fig, "18_ranking_risco")
