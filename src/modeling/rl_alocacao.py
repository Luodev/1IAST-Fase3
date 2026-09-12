"""
Aprendizagem por reforço para alocação de intervenções educacionais.

-----------------------------------------------------------------------------
O PROBLEMA DE DECISÃO
-----------------------------------------------------------------------------
O modelo supervisionado diz quais municípios estão em risco e a clusterização
diz de que tipo é esse risco. Falta a decisão: com orçamento limitado, **qual
intervenção mandar para qual perfil de município?**

Uma secretaria não sabe de antemão qual programa funciona melhor em cada
contexto — ela descobre aplicando, medindo e reaplicando. Esse é exatamente o
dilema exploração x explotação que o aprendizado por reforço formaliza.

-----------------------------------------------------------------------------
HONESTIDADE SOBRE O AMBIENTE
-----------------------------------------------------------------------------
Não existe base pública com o efeito causal de cada programa por município.
O ambiente aqui é um **simulador calibrado nos dados reais**, não uma avaliação
de impacto:

  * a magnitude dos ganhos vem da distribuição real da variação 2023 -> 2024
    observada em cada cluster;
  * o retorno é decrescente conforme a taxa se aproxima de 100% (é mais fácil
    tirar um município de 30% para 40% do que de 85% para 95%);
  * a matriz de efeito por (perfil x intervenção) é sorteada uma vez e fica
    oculta do agente — é o que ele precisa aprender.

O que o experimento demonstra é a **mecânica de decisão**: que um agente que
explora converge para a alocação certa e quanto ele perde no caminho. Com dados
reais de execução de programas, o mesmo código roda sem alteração.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, RANDOM_STATE

# Intervenções disponíveis (os "braços" do problema).
INTERVENCOES = [
    "Formacao continuada de professores",
    "Material estruturado de alfabetizacao",
    "Recomposicao de aprendizagem em contraturno",
    "Busca ativa e apoio a frequencia",
]

# Custo relativo de cada intervenção por município (unidades de orçamento).
CUSTO_INTERVENCAO = np.array([3.0, 1.5, 2.5, 2.0])


# ---------------------------------------------------------------------------
# Ambiente simulado
# ---------------------------------------------------------------------------

class AmbienteIntervencao:
    """Simulador do efeito de intervenções sobre a taxa de alfabetização.

    Estado observável pelo agente: o perfil (cluster) do município.
    Ação: qual das quatro intervenções aplicar.
    Recompensa: ganho em pontos percentuais na taxa de alfabetização.
    """

    def __init__(self, perfis: pd.DataFrame, random_state: int = RANDOM_STATE):
        self.rng = np.random.default_rng(random_state)
        self.perfis = perfis.reset_index(drop=True)
        self.n_perfis = len(perfis)
        self.n_acoes = len(INTERVENCOES)

        # Ganho médio de referência por perfil, ancorado na variação real
        # observada entre 2023 e 2024 naquele cluster (nunca negativo: aqui
        # medimos o efeito da intervenção, não a trajetória espontânea).
        base = np.clip(
            (perfis["taxa_alfabetizacao_2024"] - perfis["taxa_alfabetizacao_2023"]).to_numpy(),
            1.0, None,
        )

        # Efeito oculto de cada (perfil, intervenção): multiplicador sorteado
        # em torno de 1, criando heterogeneidade — é o que o agente descobre.
        multiplicador = self.rng.uniform(0.4, 1.8, size=(self.n_perfis, self.n_acoes))
        self.efeito_verdadeiro = base[:, None] * multiplicador

        # Ruído de execução por município: adesão local, rotatividade de
        # professores, diferenças entre escolas.
        self.ruido = 1.5

        # Uma rodada de decisão não é um município, é um LOTE: um programa é
        # contratado para dezenas de cidades do mesmo perfil e o gestor observa
        # o ganho médio do lote. A média de `tamanho_lote` municípios reduz o
        # desvio do ruído pela raiz do tamanho — sem isso, o ruído de execução
        # (1,5 p.p.) engoliria efeitos de menos de 1 p.p. e nenhum agente
        # conseguiria distinguir as intervenções, por melhor que fosse.
        self.tamanho_lote = 25

    def amostrar_perfil(self) -> int:
        """Sorteia um perfil proporcionalmente ao seu número de municípios."""
        pesos = self.perfis["municipios"].to_numpy(dtype=float)
        return int(self.rng.choice(self.n_perfis, p=pesos / pesos.sum()))

    def _saturacao(self, perfil: int) -> float:
        """Retorno decrescente: quanto mais alta a taxa, menor o ganho possível."""
        taxa_atual = float(self.perfis.loc[perfil, "taxa_alfabetizacao_2024"])
        return max(0.15, (100 - taxa_atual) / 100)

    def recompensa(self, perfil: int, acao: int) -> float:
        """Ganho médio (em p.p.) do lote em que `acao` foi aplicada."""
        esperado = self.efeito_verdadeiro[perfil, acao] * self._saturacao(perfil)
        ruido_do_lote = self.ruido / np.sqrt(self.tamanho_lote)
        return float(max(0.0, esperado + self.rng.normal(0, ruido_do_lote)))

    def ganho_esperado(self, perfil: int, acao: int) -> float:
        """Valor esperado da recompensa — usado para calcular o regret."""
        return float(self.efeito_verdadeiro[perfil, acao] * self._saturacao(perfil))

    def melhor_acao(self, perfil: int) -> int:
        """Ação ótima do perfil — conhecida só pelo ambiente, para medir regret."""
        return int(np.argmax(self.efeito_verdadeiro[perfil]))


# ---------------------------------------------------------------------------
# Agentes (bandit contextual)
# ---------------------------------------------------------------------------

class AgenteBandit:
    """Base: mantém contagens e médias de recompensa por (perfil, ação)."""

    nome = "base"

    def __init__(self, n_perfis: int, n_acoes: int, random_state: int = RANDOM_STATE):
        self.rng = np.random.default_rng(random_state)
        self.n_acoes = n_acoes
        self.contagem = np.zeros((n_perfis, n_acoes))
        self.media = np.zeros((n_perfis, n_acoes))
        self.passo = 0

    def escolher(self, perfil: int) -> int:
        raise NotImplementedError

    def atualizar(self, perfil: int, acao: int, recompensa: float) -> None:
        """Média incremental — não guarda o histórico inteiro em memória."""
        self.passo += 1
        self.contagem[perfil, acao] += 1
        n = self.contagem[perfil, acao]
        self.media[perfil, acao] += (recompensa - self.media[perfil, acao]) / n


class EpsilonGreedy(AgenteBandit):
    """Explora com probabilidade epsilon; no resto do tempo, explota o melhor.

    Epsilon decai com o tempo: no começo é preciso experimentar, depois já se
    sabe o suficiente e insistir em explorar vira desperdício de orçamento.
    """

    nome = "Epsilon-Greedy"

    def __init__(self, *args, epsilon_inicial: float = 0.3, decaimento: float = 0.999,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon_inicial
        self.decaimento = decaimento

    def escolher(self, perfil: int) -> int:
        self.epsilon = max(0.02, self.epsilon * self.decaimento)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_acoes))
        return int(np.argmax(self.media[perfil]))


class UCB(AgenteBandit):
    """Otimismo diante da incerteza: soma um bônus às ações pouco testadas.

    Ao contrário do epsilon-greedy, não explora ao acaso — explora justamente
    onde ainda não sabe.

    Usamos a variante **UCB-V**, em que o bônus é escalado pela variância
    observada das recompensas. O UCB1 original supõe recompensas em [0, 1]; aqui
    elas são ganhos em pontos percentuais, que variam de fração de ponto a vários
    pontos conforme o perfil. Com a constante fixa do UCB1, o bônus fica enorme
    perto de recompensas pequenas e o agente explora para sempre — foi o que
    observamos antes da correção. Escalar pela variância devolve a comparação
    justa com os outros dois agentes.
    """

    nome = "UCB"

    def __init__(self, *args, c: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.soma_quadrados = np.zeros_like(self.media)

    def atualizar(self, perfil: int, acao: int, recompensa: float) -> None:
        super().atualizar(perfil, acao, recompensa)
        self.soma_quadrados[perfil, acao] += recompensa ** 2

    def _desvio(self, perfil: int) -> np.ndarray:
        n = np.maximum(self.contagem[perfil], 1)
        variancia = np.maximum(self.soma_quadrados[perfil] / n - self.media[perfil] ** 2, 0)
        return np.sqrt(variancia)

    def escolher(self, perfil: int) -> int:
        nao_testadas = np.where(self.contagem[perfil] == 0)[0]
        if len(nao_testadas):
            return int(nao_testadas[0])
        total = self.contagem[perfil].sum()
        bonus = self.c * self._desvio(perfil) * np.sqrt(
            2 * np.log(total) / self.contagem[perfil])
        return int(np.argmax(self.media[perfil] + bonus))


class ThompsonSampling(AgenteBandit):
    """Amostra a crença sobre cada ação e age como se a amostra fosse verdade.

    Mantém uma distribuição Normal sobre o ganho médio de cada par
    (perfil, ação); a variância encolhe conforme a evidência se acumula, então
    a exploração diminui sozinha, sem parâmetro para ajustar.
    """

    nome = "Thompson Sampling"

    def escolher(self, perfil: int) -> int:
        desvio = 3.0 / np.sqrt(1 + self.contagem[perfil])
        amostra = self.rng.normal(self.media[perfil], desvio)
        return int(np.argmax(amostra))


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------

def rodar_experimento(perfis: pd.DataFrame, n_rodadas: int = 4000,
                      salvar: bool = True) -> dict:
    """Compara os três agentes pelo arrependimento (regret) acumulado.

    Regret = quanto se deixou de ganhar por não ter escolhido, desde o início, a
    melhor intervenção para cada perfil. É a métrica certa aqui porque mede o
    custo do aprendizado — em política pública, cada ponto de regret é ganho de
    alfabetização que não aconteceu.
    """
    historico = []
    resumo = []

    for classe in (EpsilonGreedy, UCB, ThompsonSampling):
        ambiente = AmbienteIntervencao(perfis)
        agente = classe(ambiente.n_perfis, ambiente.n_acoes)

        regret_acumulado = 0.0
        ganho_acumulado = 0.0
        acertos = 0

        for rodada in range(n_rodadas):
            perfil = ambiente.amostrar_perfil()
            acao = agente.escolher(perfil)
            recompensa = ambiente.recompensa(perfil, acao)
            agente.atualizar(perfil, acao, recompensa)

            otima = ambiente.melhor_acao(perfil)
            acertos += int(acao == otima)
            regret_acumulado += (ambiente.ganho_esperado(perfil, otima)
                                 - ambiente.ganho_esperado(perfil, acao))
            ganho_acumulado += recompensa

            if rodada % 25 == 0:
                historico.append({
                    "agente": agente.nome, "rodada": rodada,
                    "regret_acumulado": round(regret_acumulado, 3),
                    "ganho_acumulado": round(ganho_acumulado, 3),
                    "taxa_acerto": round(acertos / (rodada + 1), 4),
                })

        # Política final aprendida: melhor intervenção estimada por perfil.
        politica = np.argmax(agente.media, axis=1)
        resumo.append({
            "agente": agente.nome,
            "regret_final": round(regret_acumulado, 2),
            "ganho_total_pp": round(ganho_acumulado, 2),
            "taxa_acerto_final": round(acertos / n_rodadas, 4),
            "politica_correta": int(sum(
                politica[p] == ambiente.melhor_acao(p) for p in range(ambiente.n_perfis))),
            "perfis": ambiente.n_perfis,
        })

    df_historico = pd.DataFrame(historico)
    df_resumo = pd.DataFrame(resumo).sort_values("regret_final").reset_index(drop=True)

    # Política recomendada = a do agente com menor regret.
    melhor_nome = df_resumo.loc[0, "agente"]
    ambiente = AmbienteIntervencao(perfis)
    classe = {c.nome: c for c in (EpsilonGreedy, UCB, ThompsonSampling)}[melhor_nome]
    agente = classe(ambiente.n_perfis, ambiente.n_acoes)
    for _ in range(n_rodadas):
        perfil = ambiente.amostrar_perfil()
        acao = agente.escolher(perfil)
        agente.atualizar(perfil, acao, ambiente.recompensa(perfil, acao))

    politica = pd.DataFrame({
        "cluster": perfis["cluster"].to_numpy(),
        "rotulo": perfis["rotulo"].to_numpy(),
        "municipios": perfis["municipios"].to_numpy(),
        "intervencao_recomendada": [INTERVENCOES[a] for a in np.argmax(agente.media, axis=1)],
        "ganho_esperado_pp": np.round(agente.media.max(axis=1), 2),
        "custo_relativo": CUSTO_INTERVENCAO[np.argmax(agente.media, axis=1)],
    })
    politica["ganho_por_unidade_de_custo"] = (
        politica["ganho_esperado_pp"] / politica["custo_relativo"]).round(3)

    if salvar:
        df_historico.to_csv(METRICS_DIR / "rl_historico.csv", index=False)
        df_resumo.to_csv(METRICS_DIR / "rl_comparacao_agentes.csv", index=False)
        politica.to_csv(METRICS_DIR / "rl_politica_recomendada.csv", index=False)

    return {"historico": df_historico, "resumo": df_resumo,
            "politica": politica, "agente_escolhido": melhor_nome}


# ---------------------------------------------------------------------------
# Alocação de orçamento
# ---------------------------------------------------------------------------

def alocar_orcamento(municipios: pd.DataFrame, politica: pd.DataFrame,
                     orcamento: float = 3000.0, salvar: bool = True) -> pd.DataFrame:
    """Distribui um orçamento fixo usando a política aprendida.

    Critério de prioridade: ganho esperado da intervenção do perfil, ponderado
    pelo risco do município (1 - probabilidade prevista de alfabetizar) e
    dividido pelo custo. É uma heurística de mochila — atende primeiro quem tem
    maior retorno social por real investido.
    """
    df = municipios.merge(
        politica[["cluster", "intervencao_recomendada", "ganho_esperado_pp",
                  "custo_relativo"]],
        on="cluster", how="left")

    risco = 1 - df["probabilidade_alfabetizar"] if "probabilidade_alfabetizar" in df \
        else 1 - (df["taxa_alvo"] / 100)

    df["prioridade"] = (df["ganho_esperado_pp"] * risco / df["custo_relativo"]).round(4)
    df = df.sort_values("prioridade", ascending=False)

    df["custo_acumulado"] = df["custo_relativo"].cumsum()
    df["atendido"] = (df["custo_acumulado"] <= orcamento).astype(int)

    atendidos = df[df["atendido"] == 1]
    df.attrs["resumo"] = {
        "orcamento": orcamento,
        "municipios_atendidos": int(len(atendidos)),
        "custo_utilizado": round(float(atendidos["custo_relativo"].sum()), 2),
        "ganho_esperado_total_pp": round(float(atendidos["ganho_esperado_pp"].sum()), 2),
        "taxa_media_dos_atendidos": round(float(atendidos["taxa_alvo"].mean()), 2),
    }

    if salvar:
        df.to_csv(METRICS_DIR / "rl_alocacao_orcamento.csv", index=False)

    return df


if __name__ == "__main__":
    from src.modeling.clustering import segmentar

    seg = segmentar()
    saida = rodar_experimento(seg["perfis"])
    print(saida["resumo"].to_string(index=False))
    print()
    print(saida["politica"].to_string(index=False))

    aloc = alocar_orcamento(seg["municipios"], saida["politica"])
    print()
    print(aloc.attrs["resumo"])
