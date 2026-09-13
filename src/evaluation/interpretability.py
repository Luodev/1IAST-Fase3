"""
Interpretabilidade do modelo supervisionado.

Responde a "quais fatores mais impactam a alfabetização?" e "quais variáveis
possuem maior influência nos modelos?".

Três leituras complementares, porque cada uma responde a uma pergunta diferente:

  * **Importância por permutação** — o quanto a métrica cai quando embaralhamos
    uma variável. Mede impacto *no desempenho do modelo*, é calculada no conjunto
    de teste e não sofre o viés da importância por impureza de árvores, que
    infla variáveis de alta cardinalidade.
  * **SHAP** — a contribuição de cada variável para *cada predição*, com sinal.
    É o que permite dizer "neste município, a baixa alfabetização adulta puxou a
    probabilidade para baixo em 8 pontos".
  * **Coeficientes / importância nativa** — barata, útil como conferência.

O SHAP é opcional: se a biblioteca não estiver instalada, o módulo segue
funcionando com as outras duas leituras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import METRICS_DIR, RANDOM_STATE
from src.preprocessing.features import descrever
from src.preprocessing.pipeline import nomes_das_features, transformar


def importancia_por_permutacao(pipeline, X, y, n_repeticoes: int = 15,
                               salvar: bool = True) -> pd.DataFrame:
    """Queda no ROC AUC ao embaralhar cada variável do conjunto de teste."""
    resultado = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=n_repeticoes,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    df = pd.DataFrame({
        "variavel": X.columns,
        "queda_roc_auc": resultado.importances_mean.round(5),
        "desvio": resultado.importances_std.round(5),
    })
    df["descricao"] = df["variavel"].map(descrever)
    df = df.sort_values("queda_roc_auc", ascending=False).reset_index(drop=True)

    if salvar:
        df.to_csv(METRICS_DIR / "importancia_permutacao.csv", index=False)
    return df


def _estimador_base(pipeline):
    """Extrai o estimador final, atravessando o wrapper de calibração."""
    modelo = pipeline
    if hasattr(modelo, "calibrated_classifiers_"):
        modelo = modelo.calibrated_classifiers_[0].estimator
    if hasattr(modelo, "named_steps"):
        return modelo, modelo.named_steps["modelo"]
    return modelo, modelo


def valores_shap(pipeline, X, n_amostras: int = 400, salvar: bool = True) -> dict:
    """Calcula valores SHAP sobre a matriz já pré-processada.

    Amostramos o conjunto porque o TreeExplainer cresce com n x features; 400
    municípios são suficientes para estabilizar o ranking médio e mantêm o
    notebook rodando em segundos.
    """
    try:
        import shap
    except ImportError:
        return {"disponivel": False,
                "motivo": "biblioteca shap não instalada (pip install shap)"}

    pipe_interno, estimador = _estimador_base(pipeline)

    amostra = X.sample(min(n_amostras, len(X)), random_state=RANDOM_STATE)
    X_transformado = transformar(pipe_interno, amostra)
    nomes = nomes_das_features(pipe_interno)

    # Cada família de modelo tem um explicador exato e barato; só caímos no
    # explicador genérico (por permutação, bem mais lento) se nenhum servir.
    if hasattr(estimador, "coef_"):
        explicador = shap.LinearExplainer(estimador, X_transformado)
        valores = explicador.shap_values(X_transformado)
    else:
        try:
            explicador = shap.TreeExplainer(estimador)
            valores = explicador.shap_values(X_transformado)
        except Exception:  # noqa: BLE001 - modelo sem explicador dedicado
            explicador = shap.Explainer(estimador.predict_proba, X_transformado)
            valores = explicador(X_transformado, max_evals=2 * X_transformado.shape[1] + 1).values

    # Classificadores binários podem devolver uma matriz por classe.
    valores = np.asarray(valores)
    if valores.ndim == 3:
        valores = valores[:, :, 1]

    ranking = pd.DataFrame({
        "variavel": nomes,
        "shap_medio_absoluto": np.abs(valores).mean(axis=0).round(5),
        "shap_medio": valores.mean(axis=0).round(5),
    })
    ranking["descricao"] = ranking["variavel"].map(descrever)
    ranking["direcao"] = np.where(
        ranking["shap_medio"] >= 0, "aumenta a chance", "reduz a chance")
    ranking = ranking.sort_values("shap_medio_absoluto", ascending=False).reset_index(drop=True)

    if salvar:
        ranking.to_csv(METRICS_DIR / "importancia_shap.csv", index=False)

    return {"disponivel": True, "ranking": ranking, "valores": valores,
            "X_transformado": X_transformado, "nomes": nomes}


def explicar_municipio(pipeline, X: pd.DataFrame, indice, shap_saida: dict,
                       top: int = 8) -> pd.DataFrame:
    """Decompõe a predição de um município específico em contribuições SHAP.

    É a peça que torna o modelo utilizável por um gestor: em vez de "risco
    alto", entrega "risco alto porque a taxa de 2023 estava 20 p.p. abaixo da
    meta e a alfabetização adulta da cidade é baixa".
    """
    if not shap_saida.get("disponivel"):
        raise RuntimeError("SHAP indisponível — instale a biblioteca shap.")

    X_t = shap_saida["X_transformado"]
    if indice not in X_t.index:
        raise KeyError(f"Município {indice} não está na amostra SHAP calculada.")

    posicao = X_t.index.get_loc(indice)
    contribuicoes = pd.DataFrame({
        "variavel": shap_saida["nomes"],
        "valor_padronizado": X_t.iloc[posicao].to_numpy().round(3),
        "contribuicao_shap": np.asarray(shap_saida["valores"])[posicao].round(4),
    })
    contribuicoes["descricao"] = contribuicoes["variavel"].map(descrever)
    contribuicoes["efeito"] = np.where(
        contribuicoes["contribuicao_shap"] >= 0, "puxa para alfabetizado",
        "puxa para nao alfabetizado")

    return (contribuicoes
            .reindex(contribuicoes["contribuicao_shap"].abs().sort_values(ascending=False).index)
            .head(top).reset_index(drop=True))


def ranking_de_risco(pipeline, abt: pd.DataFrame, limiar: float,
                     salvar: bool = True) -> pd.DataFrame:
    """Escora todos os municípios e ordena do maior para o menor risco.

    Entregável direto para a gestão: a lista de quem precisa de atenção, com a
    probabilidade calibrada e a comparação com a meta pactuada.
    """
    from src.preprocessing.features import FEATURES

    prob = pipeline.predict_proba(abt[FEATURES])[:, 1]

    ranking = abt[["id_municipio", "nome_municipio", "sigla_uf", "regiao",
                   "taxa_alf_2023", "taxa_alvo", "meta_2025", "populacao_2022"]].copy()
    ranking["probabilidade_alfabetizar"] = prob.round(4)
    ranking["risco"] = (1 - prob).round(4)
    ranking["classificacao"] = np.where(prob >= limiar, "Dentro do patamar",
                                        "Risco educacional")
    ranking["faixa_de_risco"] = pd.cut(
        ranking["risco"], bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=["Baixo", "Moderado", "Alto", "Critico"], include_lowest=True)
    ranking["distancia_meta_2025"] = (ranking["taxa_alvo"] - ranking["meta_2025"]).round(2)

    ranking = ranking.sort_values("risco", ascending=False).reset_index(drop=True)
    ranking.insert(0, "posicao", range(1, len(ranking) + 1))

    if salvar:
        ranking.to_csv(METRICS_DIR / "ranking_risco_municipios.csv", index=False)
    return ranking
