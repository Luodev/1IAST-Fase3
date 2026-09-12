"""
Métricas de avaliação e escolha de limiar.

Reúne (a) o painel de métricas usado para comparar modelos, (b) a análise de
calibração — que é o que autoriza ler a probabilidade prevista como "chance de
o aluno estar alfabetizado" — e (c) a otimização do limiar de decisão sob uma
lógica de política pública, não sob o 0,5 padrão.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def painel_de_metricas(y_true, y_prob, limiar: float = 0.5) -> dict:
    """Métricas de classificação em um dicionário achatado.

    Além das usuais, inclui:
      * **ROC AUC** — separação entre municípios que alfabetizam e os que não,
        independente do limiar escolhido;
      * **PR AUC** — mais informativa que a ROC quando o interesse está na
        classe minoritária (aqui, os municípios em risco);
      * **Brier score** — erro quadrático da *probabilidade*. Quanto menor,
        mais a probabilidade prevista corresponde à frequência real.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= limiar).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "limiar": round(float(limiar), 4),
        "acuracia": round(accuracy_score(y_true, y_pred), 4),
        "precisao": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        # Recall da classe 0 = capacidade de encontrar município em risco.
        "recall_risco": round(tn / (tn + fp) if (tn + fp) else 0.0, 4),
        "vp": int(tp), "vn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def calibracao(y_true, y_prob, n_faixas: int = 10) -> pd.DataFrame:
    """Tabela de calibração: probabilidade média prevista x frequência real.

    Um modelo calibrado coloca os pontos sobre a diagonal — em municípios com
    probabilidade prevista de 0,70, cerca de 70% de fato alfabetizam no patamar
    nacional. É essa propriedade que permite usar a saída como estimativa de
    risco, e não apenas como um ranking.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["faixa"] = pd.qcut(df["p"], q=n_faixas, duplicates="drop")

    tabela = df.groupby("faixa", observed=True).agg(
        n=("y", "size"),
        prob_media_prevista=("p", "mean"),
        frequencia_real=("y", "mean"),
    ).reset_index()
    tabela["erro_absoluto"] = (tabela["prob_media_prevista"] - tabela["frequencia_real"]).abs()
    return tabela.round(4)


def otimizar_limiar(y_true, y_prob, custo_fn: float = 3.0,
                    custo_fp: float = 1.0) -> pd.DataFrame:
    """Varre limiares e calcula o custo esperado de cada um.

    A assimetria de custos vem da política pública: classificar como "vai bem"
    um município que na verdade está em risco (falso positivo da classe
    alfabetizado = falso negativo do risco) significa deixar de mandar apoio a
    quem precisa — muito mais caro do que mandar apoio a mais um município que
    já ia bem. Por isso o erro que deixa criança sem política pesa 3x.

    Retorna a varredura completa; a linha de menor `custo_total` é a escolha.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    linhas = []
    for limiar in np.round(np.arange(0.05, 0.96, 0.01), 2):
        y_pred = (y_prob >= limiar).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        # fp = município em risco classificado como tranquilo -> perde política
        linhas.append({
            "limiar": limiar,
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "precisao": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "risco_nao_detectado": int(fp),
            "apoio_desnecessario": int(fn),
            "custo_total": round(custo_fn * fp + custo_fp * fn, 2),
        })

    return pd.DataFrame(linhas)


def limiar_otimo(y_true, y_prob, custo_fn: float = 3.0, custo_fp: float = 1.0) -> float:
    """Limiar de menor custo esperado segundo `otimizar_limiar`."""
    varredura = otimizar_limiar(y_true, y_prob, custo_fn, custo_fp)
    return float(varredura.loc[varredura["custo_total"].idxmin(), "limiar"])


def comparar_com_taxa_real(y_prob, taxa_real, corte: float) -> dict:
    """Relaciona a probabilidade prevista com a taxa observada do município.

    Ponte entre o grão municipal e a leitura no aluno: se a probabilidade
    prevista se correlaciona fortemente com a taxa real de alfabetização, o
    escore funciona como estimativa da proporção de alunos alfabetizados.
    """
    y_prob = np.asarray(y_prob)
    taxa_real = np.asarray(taxa_real)
    return {
        "correlacao_pearson": round(float(np.corrcoef(y_prob, taxa_real)[0, 1]), 4),
        "correlacao_spearman": round(
            float(pd.Series(y_prob).corr(pd.Series(taxa_real), method="spearman")), 4),
        "corte_usado": corte,
    }
