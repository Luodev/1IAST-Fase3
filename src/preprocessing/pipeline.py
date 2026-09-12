"""
Pré-processamento integrado ao modelo (scikit-learn Pipeline).

Todo tratamento estatístico — imputação, padronização, encoding — vive dentro
de um `ColumnTransformer` que é a primeira etapa do `Pipeline`. Isso garante
duas coisas exigidas pelo enunciado:

  1. **Sem vazamento entre treino e teste.** Médias de imputação, médias e
     desvios do scaler e categorias do one-hot são aprendidos *apenas* no fold
     de treino de cada validação cruzada, nunca na base inteira.
  2. **Pré-processamento acoplado ao modelo.** O artefato salvo em disco recebe
     dados crus e devolve a predição: não existe um passo manual de preparação
     que possa divergir entre o notebook e a produção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.features import CATEGORICAS, NUMERICAS


def _onehot() -> OneHotEncoder:
    """OneHotEncoder compatível com versões diferentes do scikit-learn.

    O parâmetro que devolve matriz densa mudou de `sparse` para `sparse_output`
    na 1.2; testamos o nome correto em tempo de execução para que o projeto rode
    em ambientes de aula com versões distintas.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             min_frequency=10)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def construir_preprocessador(numericas: list[str] | None = None,
                             categoricas: list[str] | None = None,
                             indicar_faltantes: bool = True) -> ColumnTransformer:
    """Monta o ColumnTransformer do projeto.

    Numéricas
        Imputação pela **mediana** (robusta às caudas longas de população, área
        e PIB per capita) com `add_indicator=True`: cada coluna que tinha valor
        faltante ganha uma flag binária. Isso preserva a informação de que o
        dado *estava ausente* — relevante aqui, porque a ausência de rede
        estadual ou privada no município não é ruído, é uma característica do
        território. Em seguida, `StandardScaler` coloca tudo na mesma escala,
        o que é indispensável para modelos lineares e para o KMeans.

    Categóricas
        Imputação pela categoria mais frequente e one-hot com
        `handle_unknown="ignore"` — se aparecer uma UF ou faixa não vista no
        treino, o modelo não quebra. `min_frequency=10` agrupa categorias raras
        e evita colunas com pouquíssimas observações.

    `indicar_faltantes=False`
        Desliga as flags de ausência. Necessário na clusterização: lá as flags
        viravam a dimensão dominante e o KMeans formava um "perfil" que era só
        o grupo de municípios sem meta cadastrada — um artefato de cadastro, não
        uma realidade educacional.
    """
    numericas = NUMERICAS if numericas is None else numericas
    categoricas = CATEGORICAS if categoricas is None else categoricas

    pipe_num = Pipeline([
        ("imputacao", SimpleImputer(strategy="median", add_indicator=indicar_faltantes)),
        ("escala", StandardScaler()),
    ])

    pipe_cat = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("encoding", _onehot()),
    ])

    return ColumnTransformer(
        transformers=[
            ("numericas", pipe_num, numericas),
            ("categoricas", pipe_cat, categoricas),
        ],
        remainder="drop",          # nada entra sem estar declarado
        verbose_feature_names_out=False,
    )


def montar_pipeline(modelo, numericas: list[str] | None = None,
                    categoricas: list[str] | None = None) -> Pipeline:
    """Encadeia pré-processamento + estimador em um único objeto treinável."""
    return Pipeline([
        ("preprocessamento", construir_preprocessador(numericas, categoricas)),
        ("modelo", modelo),
    ])


def nomes_das_features(pipeline: Pipeline) -> list[str]:
    """Nomes das colunas na saída do pré-processamento, já ajustado.

    Necessário para rotular gráficos de importância e valores SHAP, que operam
    sobre a matriz transformada e não sobre as colunas originais.
    """
    pre = pipeline.named_steps["preprocessamento"]
    try:
        return list(pre.get_feature_names_out())
    except Exception:  # noqa: BLE001 - fallback para versões antigas
        return [f"feature_{i}" for i in range(pre.transform(pd.DataFrame()).shape[1])]


def transformar(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Aplica só a etapa de pré-processamento e devolve um DataFrame nomeado."""
    pre = pipeline.named_steps["preprocessamento"]
    matriz = pre.transform(X)
    if hasattr(matriz, "toarray"):
        matriz = matriz.toarray()
    return pd.DataFrame(np.asarray(matriz), columns=nomes_das_features(pipeline),
                        index=X.index)
