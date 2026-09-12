# Registro de decisões analíticas

Este documento registra as decisões que moldaram o projeto — incluindo as
alternativas descartadas e o motivo. Serve como memória técnica: daqui a seis
meses, é aqui que se descobre por que algo foi feito de um jeito e não de outro.

Cada decisão corresponde a uma branch do repositório (ver seção 14 do README).

---

## D1 — Grão da predição: município, não aluno

**Contexto.** O enunciado pede um modelo que preveja se **um aluno** será
alfabetizado. A camada Gold é agregada por município × rede × ano.

**Alternativas consideradas:**

| Alternativa | Por que foi descartada |
|---|---|
| Simular alunos individuais a partir das proporções por nível | Fabricar microdado a partir de agregado é inventar observação; o modelo aprenderia o ruído do sorteio |
| Buscar microdados do Saeb no BigQuery (Base dos Dados) | Exige projeto GCP autenticado; quebraria a reprodutibilidade do projeto |
| Modelo binomial ponderado (cada município = ensaio agregado) | Estatisticamente correto, mas o `sample_weight` fracionário complica a validação cruzada e a busca de hiperparâmetros no scikit-learn; ficou como evolução futura |

**Decisão.** Alvo binário no grão município, com a leitura no aluno sustentada
por **calibração** e validada pela correlação de 0,77 entre probabilidade prevista
e taxa real observada.

**Consequência.** O modelo não distingue dois alunos da mesma cidade. Está
declarado no README, no relatório e no cabeçalho de `build_abt.py`.

---

## D2 — Corte do alvo em 60%

**Alternativas consideradas:**

| Corte | Positivos | Avaliação |
|---|---|---|
| taxa ≥ 50% | 73,4% | "Um aluno ao acaso tem mais chance de ser alfabetizado" — semântica boa, mas desbalanceado |
| taxa ≥ meta municipal de 2025 | 43,6% | Alinhado à política, porém a meta varia por município e o alvo deixa de ser comparável entre cidades |
| **taxa ≥ 60%** | **56,9%** | Patamar nacional (resultado de 2024 = 59,2%, meta 2024 = 59,9%), comparável entre municípios, classes equilibradas |

**Decisão.** 60%, com o alvo secundário `atinge_meta_2025` calculado e disponível
na ABT para a aplicação estratégica — sem entrar no treino.

---

## D3 — Ano-alvo 2024 com features de 2023

**Contexto.** A base tem apenas 2023 e 2024.

**Alternativas consideradas:**

| Alternativa | Por que foi descartada |
|---|---|
| Usar os dois anos como linhas, com features contemporâneas | As metas do INEP são calculadas a partir da linha de base de 2023 — usá-las para prever 2023 seria vazamento temporal |
| Prever 2023 com features de 2022 | Não existem dados de 2022 para este indicador |

**Decisão.** Alvo = 2024, preditores = 2023 + estruturais + Censo 2022. A
separação temporal é a defesa mais forte contra vazamento, porque é física: o
preditor foi medido antes do alvo.

**Consequência.** Não há validação temporal disponível (treinar em um ano, testar
em outro). Substituímos por **validação geográfica** (deixar uma região de fora),
que é um teste de generalização até mais duro.

---

## D4 — Poda de multicolinearidade

**Contexto.** Seis dos preditores candidatos mediam essencialmente a mesma coisa,
com correlações de até 1,000.

**Alternativa descartada:** manter tudo e deixar a regularização L2 resolver.
Funciona para a acurácia — mas não para a interpretação, que é metade da entrega:
com variáveis sinônimas, a importância se reparte arbitrariamente e nenhuma
aparece como relevante.

**Decisão.** Poda com limite de 0,95, resolvendo cada empate a favor da variável
mais interpretável para um gestor. Documentada em `features.py` e reproduzível por
`build_abt.analisar_redundancia()`.

**Efeito prático.** Antes da poda, as três variáveis mais importantes eram metas
do INEP — que são, na prática, o resultado de 2023 reescalado. Depois, o topo do
ranking passou a ser proficiência em Português e percentil na UF: conceitos
educacionais que um gestor sabe o que fazer com.

---

## D5 — Escolha do algoritmo pela regra do erro-padrão

**Contexto.** Random Forest (0,9032), Extra Trees (0,9030), Gradient Boosting
(0,9012) e Regressão Logística (0,9008) empatam dentro do desvio entre folds
(0,011).

**Alternativa descartada:** pegar o maior ROC AUC. Seria escolher ruído — e
levaria a um modelo com overfitting 12 vezes maior (0,073 contra 0,006).

**Decisão.** *One standard error rule*: entre os empatados, vence o de menor
diferença entre treino e validação. Implementado em `train.escolher_modelo()`,
com a justificativa gravada no JSON de métricas.

**Consequência.** O modelo final é uma Regressão Logística — mais simples, 20x
mais rápida, com o melhor F1 do conjunto e coeficientes explicáveis. Para um
escore que decide onde a política pública chega, isso é vantagem, não concessão.

---

## D6 — Limiar por custo de política pública, não 0,50

**Contexto.** Os dois erros custam coisas diferentes: deixar um município em
risco sem apoio custa desenvolvimento de criança; mandar apoio a mais custa
orçamento recuperável.

**Decisão.** Ponderação 3:1 e varredura de limiares. O mínimo fica em 0,77.

**Trade-off explícito:** acurácia cai de 0,821 para 0,791, e a detecção de
municípios em risco sobe de 0,800 para 0,931 — de 93 para 32 municípios em risco
que passariam despercebidos.

---

## D7 — Clusterização: cotovelo em vez de silhueta

**Contexto.** A silhueta fica entre 0,18 e 0,24 para todo k de 2 a 10 — quase
plana, e sempre com máximo em k = 2.

**Diagnóstico.** Não é defeito do método. Dados socioeconômicos municipais formam
um **gradiente contínuo**, não ilhas separadas. Quando a silhueta não discrimina,
usá-la como critério é escolher ruído.

**Dois problemas corrigidos no caminho:**

1. Com `add_indicator=True` herdado do pipeline supervisionado, as flags de
   ausência viravam a dimensão dominante e o KMeans formava um cluster que era só
   "municípios sem meta cadastrada" — artefato de cadastro. Corrigido com
   `indicar_faltantes=False`.
2. Sem winsorização, a partir de k = 4 o menor grupo tinha menos de 1% dos
   municípios: cidades com PIB inflado por royalties e capitais densíssimas
   consumiam clusters inteiros. Corrigido truncando a 1%/99%.

**Decisão.** Critério principal = cotovelo da inércia (formalizado como distância
máxima à reta que liga os extremos), com restrição de negócio de mínimo 2% por
cluster. Resultado: k = 4, perfis com 121 a 2.269 municípios.

---

## D8 — Séries temporais: declarar a limitação em vez de mascará-la

**Contexto.** 2 pontos por município, 3 para o Brasil.

**Alternativa descartada:** ajustar ARIMA ou Prophet. Ambos exigem série longa
para estimar autocorrelação e sazonalidade; com 3 pontos, produziriam números com
aparência de rigor e nenhum conteúdo.

**Decisão.** Tendência explícita em três cenários para o Brasil (linear, Holt e
amortecido) e, por município, extrapolação da variação observada com dois freios:

- **encolhimento** em direção à média da UF, porque o desvio da variação anual é
  de 16,7 p.p.;
- **amortecimento** de 0,85 ao ano, porque os primeiros pontos percentuais são os
  mais fáceis e existe teto em 100%.

**Resultado que justifica a decisão.** A projeção linear leva o Brasil a 90,7% em
2030 — mais de 10 pontos acima da meta, patamar que nenhum sistema educacional
alcançou em seis anos. O cenário amortecido chega a 81,9%, margem estreita sobre
a meta de 80%. A diferença entre os dois é exatamente o valor de reconhecer a
saturação.

**Entrega.** Classificação de ritmo, não previsão pontual.

---

## D9 — Aprendizado por reforço: lote, não município

**Contexto.** Na primeira versão, cada rodada aplicava a intervenção a um único
município. O ruído de execução (σ = 1,5 p.p.) engolia efeitos de menos de 1 p.p.
e **nenhum agente conseguia aprender** — a taxa de acerto do UCB ficou em 0,0005.

**Diagnóstico.** O problema não era o algoritmo, era a formulação. Além disso, o
UCB1 clássico supõe recompensas em [0, 1]; com ganhos em pontos percentuais, o
bônus de exploração ficava desproporcional e o agente explorava indefinidamente.

**Duas correções:**

1. **Recompensa por lote de 25 municípios.** É o que acontece na prática — um
   programa é contratado para dezenas de cidades e o gestor observa a média. O
   desvio cai pela raiz do tamanho e o problema fica bem-posto.
2. **UCB-V**, com bônus escalado pela variância observada, em vez do UCB1 de
   constante fixa.

**Resultado.** O UCB passou de 0,0005 para 96,8% de acerto e o menor
arrependimento dos três agentes — a ordenação teórica esperada.

**Limitação mantida e declarada.** O ambiente é simulado. Sem dados de execução
de programas com alguma aleatorização, não há como estimar efeitos causais.

---

## D10 — Versionar `data/raw/` e `data/external/`, ignorar `data/lake/`

**Critério.** Versiona-se o que o código **não consegue** reproduzir sozinho;
ignora-se o que ele regenera em segundos.

- `data/raw/` e `data/external/` ficam no Git: garantem que o projeto rode
  offline e que os resultados não mudem se uma API do IBGE sair do ar ou revisar
  um número.
- `data/lake/`, `data/processed/` e `models/*.joblib` são derivados
  determinísticos de `run_pipeline.py` — versioná-los só inflaria o repositório e
  criaria risco de o binário ficar dessincronizado do código.

---

## Correções de bugs encontrados durante o desenvolvimento

| Onde | Problema | Correção |
|---|---|---|
| Visão Gold `comparacao_metas_nacionais` (herdada da Fase 2) | Agregava a rede "total" por UF, que existe em 1 linha da base — a visão saía com 1 registro | Passou a usar a taxa da rede pública da tabela de metas por UF; 50 registros |
| `separar_dados()` | `regiao` entrava duas vezes em `X` (é feature categórica e foi adicionada como identificador), quebrando o `ColumnTransformer` | Identificadores extras filtrados contra a lista de features |
| `validar_ausencia_de_vazamento()` | Colunas constantes geravam divisão por zero no cálculo da correlação | Constantes excluídas da checagem — elas não podem vazar nada |
| `plots.taxa_por_regiao()` | `boxplot(labels=...)` foi removido no matplotlib 3.9 | Tenta `tick_labels` e cai para `labels` em versões antigas |
| `ibge.baixar_localidades()` | Municípios criados após a última revisão da malha vêm sem o bloco `microrregiao`, causando `TypeError` | Navegação defensiva da árvore, com `regiao-imediata` como caminho alternativo |
| `plots.mapa_clusters()` | Rótulos longos do eixo y invadiam o gráfico vizinho | Nome do perfil desenhado dentro da barra |
