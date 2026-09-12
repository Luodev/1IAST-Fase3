# Tech Challenge — Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto integrador da Fase 3 da pós-graduação em IA para Devs (FIAP). Consome a
**camada Gold construída na Fase 2** ([1IAST-Fase2](https://github.com/Luodev/1IAST-Fase2))
e a transforma em inteligência analítica aplicada: um modelo supervisionado de
risco educacional, uma segmentação de perfis municipais, uma projeção das metas
até 2030 e um agente de aprendizado por reforço para alocar intervenções.

---

## Índice

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo analítico](#2-objetivo-analítico)
3. [Base de dados](#3-base-de-dados)
4. [Como executar](#4-como-executar)
5. [Estrutura do repositório](#5-estrutura-do-repositório)
6. [Etapas de modelagem](#6-etapas-de-modelagem)
7. [Escolha do algoritmo](#7-escolha-do-algoritmo)
8. [Métricas de avaliação](#8-métricas-de-avaliação)
9. [Interpretação dos resultados](#9-interpretação-dos-resultados)
10. [Insights encontrados](#10-insights-encontrados)
11. [Limitações do projeto](#11-limitações-do-projeto)
12. [Aplicação prática para políticas públicas](#12-aplicação-prática-para-políticas-públicas)
13. [Possíveis evoluções futuras](#13-possíveis-evoluções-futuras)
14. [Fluxo de trabalho Git](#14-fluxo-de-trabalho-git)

---

## 1. Contexto do problema

A alfabetização na idade certa é o alicerce de toda a trajetória escolar. Uma
criança que termina o 2º ano sem ler com fluência carrega a defasagem por anos,
e a conta chega em evasão, repetência e desigualdade de renda uma geração
depois.

O **Compromisso Nacional Criança Alfabetizada** estabeleceu metas anuais até
2030 — 80% dos alunos alfabetizados ao fim do 2º ano — e o INEP passou a medir o
**Indicador Criança Alfabetizada** a partir de 2023. O Brasil saiu de 55,9% em
2023 para 66,0% em 2025.

O problema é que a média nacional esconde realidades incompatíveis. Entre os
5.396 municípios analisados, a taxa vai de **4% a 100%**. Saber o número de hoje
não basta: o gestor precisa antecipar quem vai ficar para trás, entender quais
fatores pesam mais e decidir onde colocar um orçamento que nunca dá para todos.

É esse o espaço que a Ciência de Dados ocupa aqui — transformar dado público em
decisão.

---

## 2. Objetivo analítico

**Objetivo principal.** Desenvolver um modelo supervisionado capaz de prever se
um aluno será considerado alfabetizado ou não, usando variáveis educacionais,
territoriais e socioeconômicas.

### Nota metodológica sobre o grão da predição

A camada Gold é **agregada**: o menor grão publicado pelo INEP para o Indicador
Criança Alfabetizada é *município × rede × ano*. Não existe microdado por aluno.

Adotamos o grão mais fino disponível e mantemos a leitura no aluno:

- cada linha da base analítica representa o **aluno típico** de um município da
  rede municipal;
- o alvo vale 1 quando o município alfabetiza pelo menos **60%** dos seus alunos
  — patamar ancorado no resultado nacional de 2024 (59,2%) e na meta nacional
  daquele ano (59,9%);
- a probabilidade prevista é **calibrada** (calibração isotônica) e conferida
  contra a taxa real de cada município: a correlação entre probabilidade prevista
  e taxa observada é de **0,77**. É isso que autoriza lê-la como a chance de um
  aluno daquele município terminar o 2º ano alfabetizado.

A limitação está declarada na seção 11 e no relatório técnico.

### Perguntas de negócio respondidas

| Pergunta do enunciado | Onde é respondida |
|---|---|
| Quais fatores mais impactam a alfabetização? | Notebook 02 — permutação + SHAP |
| Quais municípios apresentam maior risco educacional? | Notebook 02 — `ranking_risco_municipios.csv` |
| Quais regiões possuem padrões semelhantes? | Notebook 03 — clusterização |
| Como prever municípios que podem não atingir metas futuras? | Notebook 04 — projeção até 2030 |
| Quais variáveis possuem maior influência nos modelos? | Notebook 02 — importância por permutação e SHAP |
| Onde alocar o orçamento de intervenção? | Notebook 05 — aprendizado por reforço |

---

## 3. Base de dados

### 3.1 Origem — camada Gold da Fase 2

Os dados vêm do repositório da Fase 2. A pipeline Medalhão que lá roda em AWS
Glue é **reexecutada localmente em pandas** por `src/data/gold_loader.py`, com o
mesmo particionamento Hive-style (`camada/entidade/ano=YYYY/`) e a mesma
semântica idempotente de escrita. Assim o projeto da Fase 3 é reproduzível sem
provisionar infraestrutura.

| Entidade (Silver) | Registros | Conteúdo |
|---|---|---|
| `indicador_municipio` | 23.995 | Taxa de alfabetização e proficiência por município, série, rede |
| `indicador_uf` | 145 | Mesmo indicador agregado por UF |
| `meta_municipio` | 10.704 | Trajetória de metas 2024–2030 por município |
| `meta_uf` | 54 | Metas por UF |
| `meta_brasil` | 3 | Série nacional e metas do Compromisso |

| Visão Gold | Registros | Uso na Fase 3 |
|---|---|---|
| `alfabetizacao_por_municipio` | 10.896 | Alvo e histórico municipal |
| `ranking_municipios` | 10.896 | Posição e percentil dentro da UF |
| `evolucao_temporal` | 49 | Série por UF (notebook 04) |
| `comparacao_metas_nacionais` | 50 | Comparação UF × meta nacional |

> **Correção aplicada à visão 4.** Na Fase 2, `comparacao_metas_nacionais`
> agregava a rede "total" (código 0) do indicador por UF — que aparece em apenas
> 1 linha da base, deixando a visão praticamente vazia. A taxa estadual passou a
> vir da tabela de metas por UF, que traz o resultado da rede pública de cada
> estado nos dois anos. Mesmo conteúdo analítico, visão utilizável.

### 3.2 Enriquecimento externo (IBGE)

Coletado por `src/data/ibge.py` e versionado em `data/external/`, para que o
projeto rode offline e os resultados não mudem se uma API sair do ar.

| Fonte | Variáveis |
|---|---|
| API de Localidades | Região, UF, mesorregião, microrregião, região intermediária |
| Censo 2022 — tabela 4714 | População residente, área territorial, densidade demográfica |
| Censo 2022 — tabela 9543 | Taxa de alfabetização da população de 15 anos ou mais |
| PIB dos Municípios — tabela 5938 | PIB municipal (base do PIB per capita) |

### 3.3 A ABT

`src/data/build_abt.py` monta a tabela analítica: **5.396 municípios × 35
colunas**, alvo de 2024, **56,9% de positivos**.

---

## 4. Como executar

```bash
pip install -r requirements.txt
```

```bash
python run_pipeline.py
```

O pipeline completo roda em cerca de 4 minutos e executa, em ordem: reconstrução
da camada Gold, download (ou leitura do cache) dos dados do IBGE, montagem da
ABT, análise exploratória, modelagem supervisionada, clusterização, projeção de
séries temporais e aprendizado por reforço. Todas as figuras vão para `images/`
e todas as tabelas para `reports/metrics/`.

Para rodar apenas uma etapa:

```bash
python run_pipeline.py --etapa supervisionado
```

Para uma execução rápida, com busca de hiperparâmetros reduzida:

```bash
python run_pipeline.py --rapido
```

Os notebooks reproduzem as mesmas etapas com narrativa completa:

```bash
jupyter notebook notebooks/
```

O projeto é **idempotente**: todas as fontes de aleatoriedade estão fixadas em
`RANDOM_STATE = 42`, e reexecutar produz exatamente os mesmos artefatos.

---

## 5. Estrutura do repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/          CSVs da camada Gold da Fase 2 (versionados)
│   ├── external/     enriquecimento IBGE (versionado)
│   ├── processed/    ABT gerada  (ignorada pelo Git — é derivada)
│   └── lake/         data lake local bronze/silver/gold (ignorado)
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_modelagem_supervisionada.ipynb
│   ├── 03_segmentacao_nao_supervisionada.ipynb
│   ├── 04_series_temporais_metas.ipynb
│   └── 05_aprendizado_por_reforco.ipynb
├── src/
│   ├── config.py            caminhos, constantes de negócio, semente
│   ├── data/                gold_loader · ibge · build_abt
│   ├── preprocessing/       features · pipeline (ColumnTransformer)
│   ├── modeling/            train · clustering · forecasting · rl_alocacao
│   ├── evaluation/          metrics · interpretability
│   └── visualization/       plots
├── reports/
│   ├── relatorio_tecnico.md
│   ├── roteiro_video.md
│   ├── resumo_executivo.json
│   └── metrics/             18 tabelas de resultado (CSV/JSON)
├── docs/
│   └── decisoes_analiticas.md
├── images/                  19 figuras geradas pelo pipeline
├── models/                  artefato treinado (.joblib)
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 6. Etapas de modelagem

### 6.1 Tratamento de data leakage

O alvo deriva da taxa de alfabetização de 2024. Nenhuma medida da mesma avaliação
pode entrar como preditor.

| Bloqueado | Motivo |
|---|---|
| `taxa_alfabetizacao` (2024) | é o próprio alvo |
| `media_portugues` (2024) | mesma prova que gerou o alvo |
| `proporcao_aluno_nivel_*` (2024) | decomposição do alvo |
| `nivel_alfabetizacao` (2024) | classificação do resultado de 2024 |
| `gap_meta` / `status_meta` (2024) | calculados a partir do alvo |

| Permitido | Motivo |
|---|---|
| Qualquer indicador de 2023 | histórico, medido um ano antes |
| Metas 2024–2030 | publicadas pelo INEP a partir da linha de base de 2023 |
| Contexto do Censo 2022 (IBGE) | anterior ao alvo |

A regra é **testada por código**, não prometida em texto:
`build_abt.validar_ausencia_de_vazamento()` levanta erro se alguma coluna
proibida aparecer entre os preditores ou se algum preditor tiver correlação acima
de 0,95 com o alvo. A variável mais correlacionada é a proficiência em Português
de **2023**, com 0,65 — histórico legítimo.

A segunda defesa é estrutural: a separação treino/teste acontece **antes** de
qualquer estatística, e todo o pré-processamento vive dentro do `Pipeline`, sendo
reajustado dentro de cada fold da validação cruzada.

### 6.2 Poda de multicolinearidade

A análise de correlação revelou que vários candidatos mediam a mesma coisa:

| Removida | Motivo |
|---|---|
| `meta_2030` | constante — 80% para todos os municípios |
| `inclinacao_meta_anual` | r = −1,000 com `meta_2025` (é `(80 − meta_2025)/5`) |
| `meta_2024` | r = 1,000 com `meta_2025` |
| `meta_2025` | r = 0,966 com `taxa_alf_2023` |
| `taxa_privada_2023` | r = 0,987 com `taxa_alf_2023` |
| `gap_2023_para_meta_2024` | r = 0,975 com `gap_2023_para_meta_2025` |

Duas remoções merecem nota analítica:

- **As metas do INEP são função da linha de base de 2023**, então carregam quase
  a mesma informação que o resultado daquele ano. Manter as duas dividia a
  importância entre variáveis sinônimas e escondia o desempenho histórico. O que
  sobra da meta e não está na taxa é a distância entre elas, preservada em
  `gap_2023_para_meta_2025`.
- **`taxa_privada_2023` é quase idêntica à municipal** porque, em municípios sem
  rede privada de 2º ano, a base do INEP repete o valor da rede municipal na
  linha da rede privada. Artefato de preenchimento, não coincidência.

Multicolinearidade não derruba a acurácia, mas arruína a interpretação — e metade
da entrega deste projeto é explicar por que um município foi priorizado.

### 6.3 Engenharia de atributos

Restaram **20 preditores**: 16 numéricos e 4 categóricos.

| Bloco | Variáveis |
|---|---|
| Histórico (2023) | taxa municipal, proficiência em Português, participação na avaliação, ranking e percentil na UF, taxa da rede estadual local, diferenças entre redes |
| Metas | distância entre o resultado de 2023 e a meta de 2025 |
| Socioeconômicas | log da população, área, densidade, alfabetização adulta (15+), log do PIB per capita, diferença entre alfabetização infantil e adulta, porte relativo na UF |
| Categóricas | UF, região, faixa populacional, nível INEP de 2023 |

Variáveis derivadas com intenção analítica explícita:

- `dif_crianca_adulto` — isola o efeito da escola do capital educacional das
  famílias;
- `dif_municipal_estadual_2023` — se a rede privada da cidade vai bem e a
  municipal vai mal, o problema é de gestão da rede, não do território;
- `percentil_uf_2023` — posição relativa, comparável entre UFs de tamanhos muito
  diferentes.

### 6.4 Pipeline de pré-processamento

```
ColumnTransformer
├── numéricas   → SimpleImputer(mediana, add_indicator=True) → StandardScaler
└── categóricas → SimpleImputer(moda) → OneHotEncoder(handle_unknown="ignore",
                                                      min_frequency=10)
```

- **Mediana** e não média: população, área e PIB per capita têm caudas longas.
- **`add_indicator=True`**: `taxa_estadual_2023` falta em ~80% dos municípios, e
  isso não é erro de coleta — a maioria das cidades não tem rede estadual
  oferecendo o 2º ano. A ausência é informação sobre o território, e vira uma
  coluna binária em vez de ser mascarada pela imputação.
- **`handle_unknown="ignore"`**: uma UF não vista no treino não quebra o modelo.
- **`remainder="drop"`**: nada entra sem estar declarado.

As 20 colunas originais viram **64 colunas transformadas**.

---

## 7. Escolha do algoritmo

Cinco candidatos comparados por validação cruzada estratificada de 5 folds:

| Modelo | ROC AUC validação | ROC AUC treino | Overfitting | F1 | Tempo |
|---|---|---|---|---|---|
| Random Forest | 0,9032 ± 0,0115 | 0,9760 | **0,0728** | 0,8447 | 12,3 s |
| Extra Trees | 0,9030 ± 0,0112 | 0,9503 | 0,0472 | 0,8443 | 10,1 s |
| Gradient Boosting | 0,9012 ± 0,0113 | 0,9680 | 0,0668 | 0,8359 | 2,2 s |
| **Regressão Logística** | **0,9008 ± 0,0139** | 0,9067 | **0,0060** | **0,8462** | **0,6 s** |
| Árvore de Decisão | 0,8798 ± 0,0110 | 0,9078 | 0,0280 | 0,8345 | 0,5 s |

### A regra do erro-padrão

Os quatro primeiros modelos ficam separados por **milésimos** de ROC AUC,
enquanto o desvio entre os folds é da ordem de 0,011. A diferença está dentro do
ruído: escolher o primeiro da lista seria escolher acaso.

Aplicamos a *one standard error rule* — entre todos os modelos a menos de um
desvio-padrão do melhor, fica o que **menos decorou o treino**. Os ensembles
chegam a 0,97 de AUC no treino contra 0,90 na validação. A Regressão Logística
entrega o mesmo desempenho de validação com uma diferença de 0,006 entre treino
e validação, e roda 20 vezes mais rápido.

Em política pública essa escolha vale duas vezes: o modelo mais simples
generaliza melhor fora da amostra e é defensável diante de um gestor que precisa
entender por que sua cidade entrou na lista de prioridades.

**Hiperparâmetros otimizados** por `RandomizedSearchCV` (40 combinações, métrica
ROC AUC): `C=5`, `penalty=l2`, `class_weight=balanced`.

**Calibração** isotônica com validação cruzada interna ao treino — o conjunto de
teste permanece intocado.

---

## 8. Métricas de avaliação

### Desempenho no conjunto de teste (1.080 municípios nunca vistos)

| Métrica | Valor | Leitura |
|---|---|---|
| ROC AUC | **0,9024** | separação entre municípios que alfabetizam e os que não |
| PR AUC | 0,9246 | desempenho sobre a classe de interesse |
| Acurácia | 0,8213 | — |
| F1 | 0,8422 | — |
| **Brier** | **0,1246** | erro da *probabilidade* — quanto menor, mais confiável o risco |
| Correlação prob. × taxa real | **0,77** | a ponte entre o grão municipal e a leitura no aluno |

### Escolha do limiar por custo de política pública

O limiar de 0,50 é convenção matemática, não decisão de gestão. Os dois erros
custam coisas diferentes:

- dizer que um município "vai bem" quando ele está em risco → **criança sem
  política pública**;
- dizer que um município está em risco quando ele já ia bem → **orçamento
  desperdiçado**, recuperável.

Ponderando o primeiro erro com peso 3 e o segundo com peso 1 e varrendo os
limiares, o custo mínimo fica em **0,77**. O limiar mais alto exige mais
evidência para declarar um município seguro, elevando a detecção de risco.

### Generalização — três testes

1. **Curva de aprendizado**: treino e validação convergem, sem vão persistente.
2. **Validação cruzada estratificada**: desvio de 0,014 entre folds.
3. **Validação geográfica** (deixar uma região inteira fora do treino):

| Região de teste | Municípios | ROC AUC |
|---|---|---|
| Nordeste | 1.779 | 0,780 |
| Norte | 413 | 0,770 |
| Sudeste | 1.612 | 0,738 |
| Centro-Oeste | 462 | 0,702 |
| Sul | 1.130 | 0,663 |

**Este é o resultado mais honesto do projeto.** O AUC cai de 0,90 para a faixa de
0,66–0,78 quando o modelo precisa opinar sobre uma região que nunca viu. A
explicação é direta: parte relevante do sinal está nas variáveis territoriais, e
ao remover a região inteira do treino o one-hot de UF e região fica todo em zero.

A consequência prática: **o modelo serve para priorizar municípios dentro do
Brasil de hoje, não para extrapolar a um território de estrutura desconhecida.**
Para uso federal isso basta — todas as regiões estão representadas na base.

---

## 9. Interpretação dos resultados

### Importância por permutação (queda no ROC AUC do conjunto de teste)

| # | Variável | Queda no AUC |
|---|---|---|
| 1 | Unidade da Federação | 0,183 |
| 2 | Proficiência média em Português em 2023 | 0,056 |
| 3 | Percentil do município dentro da UF em 2023 | 0,044 |
| 4 | Região do país | 0,030 |
| 5 | Distância entre o resultado de 2023 e a meta de 2025 | 0,013 |
| 6 | Participação dos alunos na avaliação de 2023 | 0,012 |
| 7 | Posição relativa de população dentro da UF | 0,009 |

Os valores SHAP confirmam o ranking e acrescentam o **sinal** de cada
contribuição — por exemplo, pertencer a determinadas UFs reduz a probabilidade
prevista mesmo controlando por histórico e contexto socioeconômico.

### Leitura

**O território pesa mais que tudo.** A UF sozinha responde por três vezes a
importância da segunda variável. Isso não é um defeito do modelo, é um retrato do
país: a mesma criança, com o mesmo perfil familiar, tem chances muito diferentes
de ser alfabetizada dependendo do estado em que nasce.

**O desempenho passado é o melhor preditor individual controlável.** Proficiência
em Português e percentil na UF ocupam o 2º e o 3º lugar. Desempenho educacional é
persistente — o que é má notícia para quem está mal e boa notícia para o
planejamento, porque torna o risco previsível com um ano de antecedência.

**A participação na avaliação aparece entre as sete primeiras.** Municípios com
baixa adesão à prova tendem a ir pior. É um sinal operacional: rede desorganizada
para aplicar a avaliação costuma ser rede desorganizada para alfabetizar.

---

## 10. Insights encontrados

### 10.1 Quatro perfis municipais, e o mais surpreendente são as grandes cidades

A clusterização (KMeans, k = 4 escolhido pelo cotovelo da inércia com restrição
de tamanho mínimo de 2%) separou:

| Perfil | Municípios | Taxa 2024 | PIB per capita | População mediana |
|---|---|---|---|---|
| Risco crítico — baixa renda | 1.650 | 46,5% | R$ 13,3 mil | 13,2 mil |
| Desempenho intermediário — interior | 2.269 | 65,4% | R$ 37,9 mil | 13,0 mil |
| Alto desempenho — interior | 1.356 | 79,4% | R$ 20,2 mil | 6,9 mil |
| **Risco moderado — centros urbanos** | **121** | **53,2%** | **R$ 37,6 mil** | **264 mil** |

O quarto perfil é o achado contraintuitivo: municípios grandes, com PIB per
capita alto e **96,3% de alfabetização adulta**, mas com taxa infantil de 53% —
abaixo da média nacional. Renda e escolaridade dos pais não estão bastando
nessas cidades. O gargalo é de rede, escala e gestão.

Na direção oposta, o perfil de **alto desempenho é predominantemente nordestino e
de baixa renda**. Os dois achados juntos derrubam a leitura simplista de que
alfabetização é consequência direta de riqueza.

Validação da segmentação: embora construída sem olhar para o alvo, ela separa a
proporção de municípios no patamar nacional de menos de 20% a quase 90%.

### 10.2 O Brasil avança e 4 em cada 10 municípios andam para trás

| Classificação de ritmo | Municípios | % |
|---|---|---|
| No ritmo da meta | 2.541 | 47,1% |
| **Em retrocesso** | **2.250** | **41,7%** |
| Ritmo crítico | 365 | 6,8% |
| Ritmo insuficiente | 240 | 4,4% |

Enquanto o indicador nacional sobe mais de 5 pontos por ano, **41,7% dos
municípios pioraram** entre 2023 e 2024, e apenas **45,3% chegam à meta de 2030**
na projeção. O país avança porque os municípios grandes puxam o agregado.

**Política pública desenhada sobre a média nacional não alcança justamente quem
mais precisa.**

### 10.3 A meta nacional é alcançável, mas não está garantida

| Cenário | Brasil em 2030 | Meta |
|---|---|---|
| Tendência linear | 90,7% | 80% |
| Holt (suavização exponencial) | 90,7% | 80% |
| **Amortecido (referência)** | **81,9%** | 80% |

A extrapolação linear leva o Brasil a mais de 10 pontos acima da meta — nenhum
sistema educacional do mundo avançou tanto em seis anos. O cenário amortecido,
que reconhece que os primeiros pontos percentuais são os mais fáceis, chega a
81,9%: acima da meta, mas por margem estreita. Qualquer desaceleração do ritmo
atual coloca a meta em risco.

### 10.4 Intervenção rende mais onde o indicador está pior

O agente de aprendizado por reforço (UCB com bônus escalado pela variância)
convergiu para a intervenção ótima em **96,8%** das decisões, com arrependimento
acumulado de 19,8 pontos percentuais contra 217 do epsilon-greedy.

A política aprendida diferencia a intervenção por perfil, e o retorno por unidade
de orçamento é sistematicamente maior nos perfis de risco crítico — eles estão
longe do teto de saturação. Investir onde o indicador já está alto rende pouco,
matematicamente e na prática.

Na simulação de alocação, um orçamento fixo alcança 1.200 municípios com taxa
média de 42%, contra uma média geral de 63% — a regra de prioridade se comporta
como política redistributiva sem que isso tenha sido imposto por regra.

---

## 11. Limitações do projeto

**Sobre o grão da predição.** O alvo é municipal, não individual. Não existe
microdado por aluno para o Indicador Criança Alfabetizada. A calibração e a
correlação de 0,77 com a taxa real sustentam a leitura no aluno, mas o modelo não
distingue dois alunos do mesmo município.

**Sobre a extrapolação territorial.** O ROC AUC cai de 0,90 para 0,66–0,78 quando
uma região inteira fica fora do treino. O modelo prioriza dentro do Brasil
conhecido; não extrapola para territórios de estrutura desconhecida.

**Sobre a série temporal.** Dois pontos por município e três para o Brasil. A
projeção é uma **classificação de ritmo**, não uma previsão pontual confiável
para uma cidade específica, e não tem intervalo de confiança formal — um
intervalo estimado sobre dois pontos daria falsa sensação de rigor. O fator de
amortecimento (0,85) é uma escolha ancorada na saturação esperada, não estimada
dos dados.

**Sobre a variação 2023 → 2024.** Ela pode conter mudança metodológica do INEP,
não apenas aprendizado real. Uma terceira medição municipal tornaria o método
muito mais sólido.

**Sobre o aprendizado por reforço.** O ambiente é um **simulador calibrado nos
dados reais**, não uma avaliação de impacto. Não existe base pública com o efeito
causal de cada programa por município. O experimento demonstra a mecânica de
decisão; não estima o impacto de nenhum programa real.

**Sobre causalidade.** Todas as relações identificadas são **associativas**.
Dizer que a alfabetização adulta "impacta" a infantil é abuso de linguagem —
o que o modelo mostra é correlação condicional, e políticas públicas exigem
desenho causal para afirmar mais que isso.

**Sobre cobertura.** 5.396 dos 5.570 municípios brasileiros. Faltam os que não
têm rede municipal com 2º ano ou não participaram da avaliação — e municípios
ausentes de uma avaliação censitária são, muito provavelmente, os mais frágeis.
O viés existe e vai no sentido de subestimar o problema.

---

## 12. Aplicação prática para políticas públicas

**Três entregáveis operacionais**, todos em `reports/metrics/`:

1. **`ranking_risco_municipios.csv`** — os 5.396 municípios ordenados por risco,
   com probabilidade calibrada, faixa de risco e distância para a meta de 2025.
   É a lista de chamada da política: quem precisa de atenção agora.

2. **`clusters_perfis.csv` e `clusters_municipios.csv`** — o perfil de cada
   município. Define **que tipo** de programa faz sentido, não só se ele precisa
   de programa.

3. **`projecao_municipios_2030.csv`** — a classificação de ritmo por município,
   com projeção anual até 2030. Permite montar o plano plurianual e monitorar
   desvio de rota.

**Como isso entra na rotina de uma secretaria.** O modelo usa apenas dados do ano
anterior, todos públicos. Assim que o INEP divulga o resultado de um ano, o
ranking do ano seguinte sai em minutos — dando ao gestor uma janela de doze meses
para agir antes da próxima medição, em vez de reagir ao resultado ruim depois que
ele aconteceu.

**Por que a interpretabilidade importa aqui.** Um prefeito que recebe a
informação de que sua cidade está na faixa crítica vai perguntar por quê. O modelo
responde caso a caso pelos valores SHAP: "porque o resultado de 2023 ficou 18
pontos abaixo da meta, a participação na avaliação foi de 71% e o município está
no último quartil da UF". Sem essa resposta, a lista vira alvo de disputa política
em vez de instrumento de gestão.

**Sobre o uso responsável.** O escore prioriza apoio, nunca penaliza. Usar um
modelo de risco educacional para condicionar repasse ou ranquear gestores
publicamente inverteria o incentivo: municípios passariam a manipular a
participação na avaliação — variável que, não por acaso, o próprio modelo
identificou como preditiva.

---

## 13. Possíveis evoluções futuras

**Dados**
- Incorporar o **Censo Escolar** (infraestrutura, formação docente, razão
  aluno/turma) e o **FUNDEB** (gasto por aluno) — as variáveis controláveis pela
  gestão, ausentes na base atual.
- Cruzar com o **Cadastro Único** para medir vulnerabilidade familiar direta, no
  lugar do PIB per capita municipal.
- Incluir a medição de 2025 por município, que vai destravar séries temporais de
  verdade.

**Modelagem**
- **Modelos hierárquicos** (município aninhado em UF) para tratar o efeito
  territorial explicitamente, em vez de via one-hot.
- **Regressão sobre a taxa** como tarefa complementar à classificação, entregando
  a estimativa contínua junto com a probabilidade.
- **Modelo binomial ponderado** no grão do aluno, tratando cada município como
  ensaio agregado — a formulação estatisticamente correta para dados de proporção.

**Decisão**
- **LinUCB ou bandit contextual completo** usando as variáveis do município em
  vez do rótulo do cluster, permitindo personalização mais fina.
- **Q-learning plurianual**, tratando a alocação como decisão sequencial em que
  investir hoje muda o estado do município amanhã.
- Integrar dados reais de execução de programas com alguma aleatorização, para
  que o agente aprenda efeitos causais em vez de simulados.

**Produto**
- API de escoragem e painel para as secretarias estaduais.
- Monitoramento de *drift*: reavaliar o modelo a cada divulgação do INEP.

---

## 14. Fluxo de trabalho Git

O histórico do repositório segue um fluxo de branches por etapa analítica, com
merge sem fast-forward para preservar a topologia:

```
main
 ├── feat/estrutura-e-camada-gold
 ├── feat/enriquecimento-ibge-e-abt
 ├── feat/modelagem-supervisionada
 ├── feat/segmentacao-nao-supervisionada
 ├── feat/series-temporais-metas
 ├── feat/aprendizado-por-reforco
 └── docs/relatorio-e-readme
```

Cada branch corresponde a uma decisão analítica documentada em
`docs/decisoes_analiticas.md`. Para inspecionar a topologia:

```bash
git log --oneline --graph --all
```

---

## Documentação complementar

- [`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md) — relatório técnico completo
- [`docs/decisoes_analiticas.md`](docs/decisoes_analiticas.md) — registro das decisões e alternativas descartadas
- [`reports/roteiro_video.md`](reports/roteiro_video.md) — roteiro do vídeo executivo de 5 minutos
- [`data/README.md`](data/README.md) — dicionário de dados

---

## Fontes

- INEP — Indicador Criança Alfabetizada e metas do Compromisso Nacional Criança Alfabetizada
- IBGE — Censo Demográfico 2022 (tabelas 4714 e 9543), PIB dos Municípios (tabela 5938), API de Localidades
- Camada Gold da Fase 2 — [github.com/Luodev/1IAST-Fase2](https://github.com/Luodev/1IAST-Fase2)
