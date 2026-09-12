"""
Treino, otimização e validação do modelo supervisionado.

Fluxo:
  1. separação treino/teste estratificada (o teste só é tocado no fim);
  2. teste automático de vazamento sobre a lista de preditores;
  3. comparação de 5 algoritmos por validação cruzada estratificada;
  4. busca de hiperparâmetros (RandomizedSearchCV) no melhor candidato;
  5. calibração de probabilidades e escolha do limiar por custo de política;
  6. validação de generalização geográfica (GroupKFold por região);
  7. persistência do artefato e das métricas.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupKFold, RandomizedSearchCV, StratifiedKFold, cross_validate,
    learning_curve, train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

from src.config import CV_FOLDS, METRICS_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.data.build_abt import carregar_abt, validar_ausencia_de_vazamento
from src.evaluation.metrics import (
    calibracao, comparar_com_taxa_real, limiar_otimo, painel_de_metricas,
)
from src.preprocessing.features import ALVO_PRINCIPAL, FEATURES
from src.preprocessing.pipeline import montar_pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

ARQUIVO_MODELO = MODELS_DIR / "modelo_alfabetizacao.joblib"


# ---------------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------------

def candidatos() -> dict:
    """Modelos comparados, do mais simples ao mais flexível.

    A régua é a Regressão Logística: se um modelo complexo não a supera com
    folga, o ganho não compensa a perda de interpretabilidade.
    """
    return {
        "Regressao Logistica": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE),
        "Arvore de Decisao": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=30, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1,
            random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=5, n_jobs=-1,
            random_state=RANDOM_STATE),
        # Versão histogramada do gradient boosting: discretiza as variáveis em
        # 256 faixas antes de crescer as árvores, o que a torna uma ordem de
        # grandeza mais rápida que o GradientBoostingClassifier clássico com
        # desempenho equivalente neste volume de dados.
        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15),
    }


GRADES = {
    "Random Forest": {
        "modelo__n_estimators": [300, 500, 800],
        "modelo__max_depth": [None, 8, 12, 20],
        "modelo__min_samples_leaf": [1, 3, 5, 10, 20],
        "modelo__max_features": ["sqrt", "log2", 0.3, 0.5],
        "modelo__class_weight": [None, "balanced"],
    },
    "Extra Trees": {
        "modelo__n_estimators": [300, 500, 800],
        "modelo__max_depth": [None, 10, 16],
        "modelo__min_samples_leaf": [1, 3, 5, 10],
        "modelo__max_features": ["sqrt", "log2", 0.4],
    },
    "Gradient Boosting": {
        "modelo__max_iter": [200, 400, 800],
        "modelo__learning_rate": [0.02, 0.05, 0.1],
        "modelo__max_depth": [None, 3, 5, 8],
        "modelo__min_samples_leaf": [10, 20, 50],
        "modelo__l2_regularization": [0.0, 0.5, 2.0],
        "modelo__max_leaf_nodes": [15, 31, 63],
    },
    "Regressao Logistica": {
        "modelo__C": [0.01, 0.1, 0.5, 1, 5, 20],
        "modelo__penalty": ["l2"],
        "modelo__class_weight": [None, "balanced"],
    },
    "Arvore de Decisao": {
        "modelo__max_depth": [3, 4, 6, 8, 12],
        "modelo__min_samples_leaf": [10, 30, 60, 100],
        "modelo__criterion": ["gini", "entropy"],
    },
}


# ---------------------------------------------------------------------------
# Separação dos dados
# ---------------------------------------------------------------------------

def separar_dados(abt: pd.DataFrame | None = None):
    """Divide em treino e teste de forma estratificada e reprodutível.

    A estratificação mantém a proporção de municípios alfabetizados nas duas
    partes; sem ela, uma divisão azarada mudaria a régua das métricas.
    """
    abt = carregar_abt() if abt is None else abt
    # `regiao` já é uma feature categórica; os demais identificadores viajam
    # junto apenas para rotular relatórios (o ColumnTransformer os descarta).
    extras = [c for c in ("id_municipio", "nome_municipio") if c not in FEATURES]
    X = abt[FEATURES + extras].copy()
    y = abt[ALVO_PRINCIPAL].copy()
    taxa = abt["taxa_alvo"].copy()

    X_tr, X_te, y_tr, y_te, taxa_tr, taxa_te = train_test_split(
        X, y, taxa, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    return X_tr, X_te, y_tr, y_te, taxa_tr, taxa_te


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

def comparar_modelos(X_tr, y_tr) -> pd.DataFrame:
    """Validação cruzada estratificada de todos os candidatos.

    Reporta média e desvio no treino e na validação de cada fold: a diferença
    entre as duas colunas é a leitura direta de overfitting.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    linhas = []

    for nome, estimador in candidatos().items():
        pipe = montar_pipeline(estimador)
        scores = cross_validate(
            pipe, X_tr, y_tr, cv=cv,
            scoring=["roc_auc", "f1", "accuracy", "average_precision"],
            return_train_score=True, n_jobs=-1,
        )
        linhas.append({
            "modelo": nome,
            "roc_auc_val": round(scores["test_roc_auc"].mean(), 4),
            "roc_auc_desvio": round(scores["test_roc_auc"].std(), 4),
            "roc_auc_treino": round(scores["train_roc_auc"].mean(), 4),
            "overfitting": round(scores["train_roc_auc"].mean()
                                 - scores["test_roc_auc"].mean(), 4),
            "f1_val": round(scores["test_f1"].mean(), 4),
            "pr_auc_val": round(scores["test_average_precision"].mean(), 4),
            "acuracia_val": round(scores["test_accuracy"].mean(), 4),
            "tempo_fit_s": round(scores["fit_time"].mean(), 2),
        })

    return (pd.DataFrame(linhas)
            .sort_values("roc_auc_val", ascending=False)
            .reset_index(drop=True))


def escolher_modelo(ranking: pd.DataFrame) -> tuple[str, str]:
    """Seleciona o modelo pela regra do erro-padrão, não pelo maior AUC bruto.

    Diferenças de milésimos no ROC AUC entre algoritmos ficam dentro do desvio
    entre os folds — escolher o primeiro da lista seria escolher ruído. A regra
    aqui é a clássica *one standard error rule*: entre todos os modelos cujo AUC
    está a menos de um desvio-padrão do melhor, fica o que menos decorou o
    treino (menor diferença entre treino e validação).

    Em política pública isso importa duas vezes: o modelo mais simples generaliza
    melhor para municípios fora da amostra e é mais fácil de defender diante de
    um gestor, que precisa entender por que sua cidade foi classificada como
    prioritária.

    Retorna o nome escolhido e a justificativa em texto.
    """
    melhor_auc = ranking["roc_auc_val"].max()
    tolerancia = float(ranking.loc[ranking["roc_auc_val"].idxmax(), "roc_auc_desvio"])

    empatados = ranking[ranking["roc_auc_val"] >= melhor_auc - tolerancia]
    escolhido = empatados.sort_values("overfitting").iloc[0]

    if escolhido["modelo"] == ranking.loc[0, "modelo"]:
        justificativa = (f"{escolhido['modelo']} teve o maior ROC AUC "
                         f"({escolhido['roc_auc_val']:.4f}) e o menor overfitting "
                         f"entre os empatados.")
    else:
        justificativa = (
            f"{ranking.loc[0, 'modelo']} teve o maior ROC AUC "
            f"({melhor_auc:.4f}), mas {escolhido['modelo']} ficou dentro de um "
            f"desvio-padrão ({escolhido['roc_auc_val']:.4f}) com overfitting de "
            f"{escolhido['overfitting']:.4f} contra "
            f"{ranking.loc[0, 'overfitting']:.4f}. Pela regra do erro-padrão, "
            f"vence o modelo mais simples e mais estável.")

    return str(escolhido["modelo"]), justificativa


def otimizar(nome_modelo: str, X_tr, y_tr, n_iter: int = 40):
    """Busca aleatória de hiperparâmetros com validação cruzada.

    `RandomizedSearchCV` em vez de `GridSearchCV`: com 5 hiperparâmetros, a
    grade completa passaria de mil combinações; a busca aleatória cobre o
    espaço com uma fração do custo e, na prática, chega a resultados
    equivalentes.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipe = montar_pipeline(candidatos()[nome_modelo])

    busca = RandomizedSearchCV(
        pipe, param_distributions=GRADES[nome_modelo], n_iter=n_iter,
        scoring="roc_auc", cv=cv, n_jobs=-1, random_state=RANDOM_STATE,
        return_train_score=True, refit=True,
    )
    busca.fit(X_tr, y_tr)
    return busca


def validacao_geografica(pipeline, X, y, grupos) -> pd.DataFrame:
    """Treina deixando uma região do país de fora e testa exatamente nela.

    É um teste de generalização mais duro do que a validação cruzada aleatória:
    responde se o modelo aprendeu regularidades educacionais transferíveis ou
    apenas decorou o padrão de cada região. Um gestor federal precisa da
    primeira resposta.
    """
    cv = GroupKFold(n_splits=grupos.nunique())
    linhas = []

    for treino, teste in cv.split(X, y, groups=grupos):
        regiao = grupos.iloc[teste].unique()[0]
        modelo = clone(pipeline)          # mesma configuração, pesos zerados
        modelo.fit(X.iloc[treino], y.iloc[treino])
        prob = modelo.predict_proba(X.iloc[teste])[:, 1]

        metricas = painel_de_metricas(y.iloc[teste], prob)
        linhas.append({
            "regiao_de_teste": regiao,
            "n_municipios": len(teste),
            "roc_auc": metricas["roc_auc"],
            "f1": metricas["f1"],
            "acuracia": metricas["acuracia"],
        })

    return pd.DataFrame(linhas).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def curva_de_aprendizado(pipeline, X, y) -> pd.DataFrame:
    """Desempenho em função do volume de treino — diagnóstico de viés x variância.

    Curvas que se encontram no alto indicam modelo bem dimensionado; um vão
    persistente entre treino e validação indica variância (overfitting) e
    sugere mais regularização ou mais dados.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tamanhos, treino, validacao = learning_curve(
        pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8), random_state=RANDOM_STATE,
    )
    return pd.DataFrame({
        "n_treino": tamanhos,
        "roc_auc_treino": treino.mean(axis=1).round(4),
        "roc_auc_treino_desvio": treino.std(axis=1).round(4),
        "roc_auc_validacao": validacao.mean(axis=1).round(4),
        "roc_auc_validacao_desvio": validacao.std(axis=1).round(4),
    })


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def treinar(salvar: bool = True, n_iter: int = 40, verbose: bool = True) -> dict:
    """Executa a modelagem supervisionada de ponta a ponta."""
    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    abt = carregar_abt()

    log("[1/7] Verificando ausência de vazamento...")
    diagnostico_vazamento = validar_ausencia_de_vazamento(FEATURES, abt)
    log(f"      OK — maior correlação com o alvo: "
        f"{diagnostico_vazamento['variavel_mais_correlacionada']} "
        f"({diagnostico_vazamento['maior_correlacao_com_alvo']})")

    X_tr, X_te, y_tr, y_te, _, taxa_te = separar_dados(abt)
    log(f"[2/7] Treino: {len(X_tr)} municípios | Teste: {len(X_te)} municípios")

    log("[3/7] Comparando algoritmos por validação cruzada...")
    ranking = comparar_modelos(X_tr, y_tr)
    log(ranking.to_string(index=False))

    melhor, justificativa = escolher_modelo(ranking)
    log(f"\n      Escolha: {justificativa}")
    log(f"[4/7] Otimizando hiperparâmetros de '{melhor}' ({n_iter} combinações)...")
    busca = otimizar(melhor, X_tr, y_tr, n_iter=n_iter)
    log(f"      Melhor ROC AUC (CV): {busca.best_score_:.4f}")
    log(f"      Parâmetros: {busca.best_params_}")

    log("[5/7] Calibrando probabilidades...")
    # A calibração isotônica reajusta as probabilidades usando validação cruzada
    # interna ao treino — o conjunto de teste continua intocado.
    calibrado = CalibratedClassifierCV(busca.best_estimator_, method="isotonic", cv=5)
    calibrado.fit(X_tr, y_tr)

    prob_bruta = busca.best_estimator_.predict_proba(X_te)[:, 1]
    prob_calibrada = calibrado.predict_proba(X_te)[:, 1]

    limiar = limiar_otimo(y_te, prob_calibrada)
    log(f"      Limiar ótimo por custo de política pública: {limiar}")

    metricas = {
        "modelo_escolhido": melhor,
        "justificativa_da_escolha": justificativa,
        "melhores_parametros": {k: str(v) for k, v in busca.best_params_.items()},
        "roc_auc_cv_treino": round(float(busca.best_score_), 4),
        "teste_sem_calibracao": painel_de_metricas(y_te, prob_bruta),
        "teste_calibrado": painel_de_metricas(y_te, prob_calibrada),
        "teste_calibrado_limiar_otimo": painel_de_metricas(y_te, prob_calibrada, limiar),
        "limiar_otimo": limiar,
        "ponte_municipio_aluno": comparar_com_taxa_real(prob_calibrada, taxa_te, 60.0),
        "diagnostico_vazamento": diagnostico_vazamento,
    }
    log(f"      ROC AUC no teste: {metricas['teste_calibrado']['roc_auc']}")
    log(f"      Brier: {metricas['teste_calibrado']['brier']} "
        f"(sem calibração: {metricas['teste_sem_calibracao']['brier']})")

    log("[6/7] Validação de generalização geográfica (deixa uma região de fora)...")
    X_full = pd.concat([X_tr, X_te])
    y_full = pd.concat([y_tr, y_te])
    geo = validacao_geografica(busca.best_estimator_, X_full, y_full, X_full["regiao"])
    log(geo.to_string(index=False))

    log("[7/7] Curva de aprendizado...")
    curva = curva_de_aprendizado(busca.best_estimator_, X_tr, y_tr)

    tabela_calibracao = calibracao(y_te, prob_calibrada)

    resultado = {
        "ranking_modelos": ranking,
        "busca": busca,
        "modelo_final": calibrado,
        "modelo_nao_calibrado": busca.best_estimator_,
        "metricas": metricas,
        "validacao_geografica": geo,
        "curva_aprendizado": curva,
        "calibracao": tabela_calibracao,
        "dados": {"X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
                  "prob_te": prob_calibrada, "taxa_te": taxa_te},
    }

    if salvar:
        import joblib
        joblib.dump({"pipeline": calibrado, "limiar": limiar,
                     "features": FEATURES, "metricas": metricas},
                    ARQUIVO_MODELO)
        ranking.to_csv(METRICS_DIR / "comparacao_modelos.csv", index=False)
        geo.to_csv(METRICS_DIR / "validacao_geografica.csv", index=False)
        curva.to_csv(METRICS_DIR / "curva_aprendizado.csv", index=False)
        tabela_calibracao.to_csv(METRICS_DIR / "calibracao.csv", index=False)
        (METRICS_DIR / "metricas_modelo.json").write_text(
            json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"      Artefato salvo em {ARQUIVO_MODELO}")

    return resultado


def carregar_modelo():
    """Carrega o artefato treinado (pipeline completo + limiar de decisão)."""
    import joblib
    if not ARQUIVO_MODELO.exists():
        raise FileNotFoundError(
            f"{ARQUIVO_MODELO} não existe. Rode `python run_pipeline.py` primeiro.")
    return joblib.load(ARQUIVO_MODELO)


if __name__ == "__main__":
    treinar()
