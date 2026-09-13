"""
Pipeline reproduzível de ponta a ponta.

Executa, em ordem, tudo o que o Tech Challenge pede — da reconstrução da camada
Gold da Fase 2 até os gráficos e as tabelas usadas no relatório:

    python run_pipeline.py                # roda tudo
    python run_pipeline.py --rapido       # menos iterações na busca (teste rápido)
    python run_pipeline.py --etapa dados  # só uma etapa

Etapas: dados | supervisionado | clusters | series | reforco | relatorio

O script é idempotente: rodar de novo produz os mesmos artefatos, porque todas
as fontes de aleatoriedade estão fixadas por `RANDOM_STATE`.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from src.config import ANO_ALVO, CORTE_ALFABETIZACAO, METRICS_DIR, REPORTS_DIR
from src.data.build_abt import construir_abt
from src.data.gold_loader import construir_lake
from src.preprocessing.features import NUMERICAS

ETAPAS = ["dados", "supervisionado", "clusters", "series", "reforco", "relatorio"]


def _titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}", flush=True)


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

def etapa_dados() -> pd.DataFrame:
    """Reconstrói a camada Gold, baixa o enriquecimento IBGE e monta a ABT."""
    _titulo("ETAPA 1/6 — DADOS: camada Gold (Fase 2) + enriquecimento IBGE")

    resumo_lake = construir_lake(forcar=True)
    for visao, n in resumo_lake["gold"].items():
        print(f"  Gold {visao:<32} {n:>7} registros")

    abt = construir_abt()
    print(f"\n  ABT: {abt.shape[0]} municípios x {abt.shape[1]} colunas")
    print(f"  Alvo (alfabetizado, taxa {ANO_ALVO} >= {CORTE_ALFABETIZACAO:.0f}%): "
          f"{abt['alfabetizado'].mean():.1%} positivos")
    return abt


def etapa_exploratoria(abt: pd.DataFrame) -> None:
    """Gera as figuras da análise exploratória."""
    from src.visualization import plots

    _titulo("ETAPA 2/6 — ANÁLISE EXPLORATÓRIA")
    caminhos = [
        plots.distribuicao_alvo(abt, CORTE_ALFABETIZACAO),
        plots.taxa_por_regiao(abt),
        plots.correlacoes(abt, NUMERICAS),
        plots.dispersao_historico(abt, CORTE_ALFABETIZACAO),
        plots.socioeconomico_vs_alfabetizacao(abt),
    ]
    for caminho in caminhos:
        print(f"  figura -> {caminho.name}")


def etapa_supervisionado(rapido: bool = False) -> dict:
    """Treina, otimiza, valida e interpreta o modelo supervisionado."""
    from src.evaluation.interpretability import (
        importancia_por_permutacao, ranking_de_risco, valores_shap,
    )
    from src.evaluation.metrics import otimizar_limiar
    from src.modeling.train import treinar
    from src.preprocessing.features import FEATURES
    from src.visualization import plots

    _titulo("ETAPA 3/6 — MODELAGEM SUPERVISIONADA")
    saida = treinar(n_iter=8 if rapido else 40)

    dados = saida["dados"]
    plots.comparacao_modelos(saida["ranking_modelos"])
    plots.curvas_de_desempenho(dados["y_te"], dados["prob_te"])
    plots.curva_de_calibracao(saida["calibracao"])
    plots.curva_de_aprendizado(saida["curva_aprendizado"])

    varredura = otimizar_limiar(dados["y_te"], dados["prob_te"])
    plots.limiar_e_custo(varredura, saida["metricas"]["limiar_otimo"])
    varredura.to_csv(METRICS_DIR / "varredura_limiar.csv", index=False)

    print("\n  Interpretabilidade...")
    importancia = importancia_por_permutacao(
        saida["modelo_final"], dados["X_te"][FEATURES], dados["y_te"])
    plots.importancia_variaveis(importancia)
    print(importancia.head(10)[["descricao", "queda_roc_auc"]].to_string(index=False))

    shap_saida = valores_shap(saida["modelo_nao_calibrado"], dados["X_te"][FEATURES])
    if shap_saida.get("disponivel"):
        plots.importancia_variaveis(
            shap_saida["ranking"], coluna="shap_medio_absoluto",
            titulo="Importância média pelos valores SHAP",
            nome="10b_importancia_shap")
        print("\n  SHAP — cinco variáveis mais influentes:")
        print(shap_saida["ranking"].head(5)[["descricao", "shap_medio_absoluto",
                                             "direcao"]].to_string(index=False))
    else:
        print(f"  SHAP indisponível: {shap_saida.get('motivo')}")

    from src.data.build_abt import carregar_abt
    ranking = ranking_de_risco(saida["modelo_final"], carregar_abt(),
                               saida["metricas"]["limiar_otimo"])
    plots.ranking_risco(ranking)
    print(f"\n  Municípios em risco educacional: "
          f"{(ranking['classificacao'] == 'Risco educacional').sum()}")

    saida["importancia"] = importancia
    saida["shap"] = shap_saida
    saida["ranking_risco"] = ranking
    return saida


def etapa_clusters() -> dict:
    """Segmenta os municípios em perfis comparáveis."""
    from src.modeling.clustering import segmentar
    from src.visualization import plots

    _titulo("ETAPA 4/6 — SEGMENTAÇÃO NÃO SUPERVISIONADA")
    saida = segmentar()
    print(f"  k escolhido: {saida['k_escolhido']} | "
          f"silhueta: {saida['silhueta_final']} | "
          f"outliers DBSCAN: {saida['n_outliers']}")
    print(saida["perfis"][["cluster", "rotulo", "municipios",
                           "taxa_alfabetizacao_2024"]].to_string(index=False))

    plots.diagnostico_k(saida["diagnostico_k"], saida["k_escolhido"])
    plots.mapa_clusters(saida["municipios"], saida["perfis"])
    return saida


def etapa_series() -> dict:
    """Projeta o indicador até 2030 e classifica o ritmo de cada município."""
    from src.modeling.forecasting import (
        projetar_municipios, projetar_nacional, resumo_por_uf, serie_nacional,
    )
    from src.visualization import plots

    _titulo("ETAPA 5/6 — SÉRIES TEMPORAIS E METAS")
    serie = serie_nacional()
    projecao_br = projetar_nacional(serie)
    print("  Brasil:")
    print(projecao_br.to_string(index=False))

    projecao = projetar_municipios()
    print("\n  Ritmo dos municípios:")
    print(projecao["classificacao_ritmo"].value_counts().to_string())

    resumo = resumo_por_uf(projecao)
    resumo.to_csv(METRICS_DIR / "projecao_resumo_uf.csv", index=False)

    plots.projecao_nacional(serie, projecao_br)
    plots.ritmo_municipios(projecao)
    return {"serie": serie, "projecao_nacional": projecao_br,
            "projecao_municipios": projecao, "resumo_uf": resumo}


def etapa_reforco(clusters: dict, ranking_risco: pd.DataFrame | None) -> dict:
    """Aprende a política de alocação de intervenções."""
    from src.modeling.rl_alocacao import alocar_orcamento, rodar_experimento
    from src.visualization import plots

    _titulo("ETAPA 6/6 — APRENDIZADO POR REFORÇO")
    saida = rodar_experimento(clusters["perfis"])
    print(saida["resumo"].to_string(index=False))
    print(f"\n  Política aprendida (agente {saida['agente_escolhido']}):")
    print(saida["politica"][["rotulo", "intervencao_recomendada",
                             "ganho_esperado_pp"]].to_string(index=False))

    municipios = clusters["municipios"]
    if ranking_risco is not None:
        municipios = municipios.merge(
            ranking_risco[["id_municipio", "probabilidade_alfabetizar"]],
            on="id_municipio", how="left")

    alocacao = alocar_orcamento(municipios, saida["politica"])
    print(f"\n  Alocação de orçamento: {alocacao.attrs['resumo']}")

    plots.convergencia_rl(saida["historico"])
    plots.politica_rl(saida["politica"])
    saida["alocacao"] = alocacao
    return saida


def etapa_relatorio(contexto: dict) -> None:
    """Consolida os números principais em um resumo executivo legível."""
    _titulo("CONSOLIDAÇÃO DOS RESULTADOS")

    resumo = {
        "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "municipios_analisados": int(contexto["abt"].shape[0]),
        "corte_do_alvo": CORTE_ALFABETIZACAO,
        "proporcao_alfabetizados": round(float(contexto["abt"]["alfabetizado"].mean()), 4),
    }

    if "supervisionado" in contexto:
        metricas = contexto["supervisionado"]["metricas"]
        resumo["modelo"] = {
            "algoritmo": metricas["modelo_escolhido"],
            "roc_auc_teste": metricas["teste_calibrado"]["roc_auc"],
            "f1_teste": metricas["teste_calibrado"]["f1"],
            "brier_teste": metricas["teste_calibrado"]["brier"],
            "limiar_otimo": metricas["limiar_otimo"],
            "correlacao_prob_x_taxa_real":
                metricas["ponte_municipio_aluno"]["correlacao_pearson"],
        }
        resumo["top_5_variaveis"] = (
            contexto["supervisionado"]["importancia"].head(5)["descricao"].tolist())
        resumo["municipios_em_risco"] = int(
            (contexto["supervisionado"]["ranking_risco"]["classificacao"]
             == "Risco educacional").sum())

    if "clusters" in contexto:
        resumo["perfis_municipais"] = contexto["clusters"]["perfis"][
            ["rotulo", "municipios", "taxa_alfabetizacao_2024"]].to_dict("records")

    if "series" in contexto:
        projecao = contexto["series"]["projecao_municipios"]
        resumo["projecao_2030"] = {
            "municipios_que_atingem_a_meta": int(projecao["atinge_meta_2030"].sum()),
            "proporcao": round(float(projecao["atinge_meta_2030"].mean()), 4),
            "em_retrocesso": int((projecao["classificacao_ritmo"] == "Em retrocesso").sum()),
        }

    if "reforco" in contexto:
        resumo["aprendizado_por_reforco"] = {
            "melhor_agente": contexto["reforco"]["agente_escolhido"],
            "alocacao": contexto["reforco"]["alocacao"].attrs["resumo"],
        }

    caminho = REPORTS_DIR / "resumo_executivo.json"
    caminho.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(f"\n  Resumo salvo em {caminho}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--etapa", choices=ETAPAS, help="executa apenas uma etapa")
    parser.add_argument("--rapido", action="store_true",
                        help="reduz a busca de hiperparâmetros (para testes)")
    args = parser.parse_args()

    inicio = time.time()
    contexto: dict = {}

    executar = ETAPAS if args.etapa is None else [args.etapa]

    contexto["abt"] = etapa_dados()
    if args.etapa in (None, "dados"):
        etapa_exploratoria(contexto["abt"])

    if "supervisionado" in executar:
        contexto["supervisionado"] = etapa_supervisionado(rapido=args.rapido)
    if "clusters" in executar or "reforco" in executar:
        contexto["clusters"] = etapa_clusters()
    if "series" in executar:
        contexto["series"] = etapa_series()
    if "reforco" in executar:
        contexto["reforco"] = etapa_reforco(
            contexto["clusters"],
            contexto.get("supervisionado", {}).get("ranking_risco"))
    if "relatorio" in executar:
        etapa_relatorio(contexto)

    print(f"\nConcluído em {time.time() - inicio:.1f}s")


if __name__ == "__main__":
    main()
