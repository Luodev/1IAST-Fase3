# Relatório Técnico
## Predição e Inteligência Analítica para Alfabetização no Brasil
### Tech Challenge — Fase 3

---

## Sumário executivo

Este projeto transforma a camada Gold construída na Fase 2 em quatro produtos
analíticos sobre o Indicador Criança Alfabetizada: um **modelo supervisionado de
risco educacional** (ROC AUC de 0,902 em teste), uma **segmentação em quatro
perfis municipais**, uma **projeção das metas até 2030** e um **agente de
aprendizado por reforço** para alocar intervenções sob restrição orçamentária.

Três conclusões orientam a leitura:

1. **O território é o fator mais determinante.** A Unidade da Federação sozinha
   responde por três vezes a importância da segunda variável mais influente. A
   mesma criança, com o mesmo perfil familiar, tem chances muito diferentes de
   ser alfabetizada dependendo do estado em que nasce.

2. **A média nacional esconde o problema.** Enquanto o Brasil avança mais de 5
   pontos percentuais por ano, **41,7% dos municípios retrocederam** entre 2023 e
   2024 e apenas 45,3% chegam à meta de 2030 no ritmo projetado.

3. **Renda não explica alfabetização.** O perfil de melhor desempenho é
   predominantemente nordestino e de baixa renda; o perfil mais surpreendente são
   121 centros urbanos com PIB alto, 96% de alfabetização adulta e taxa infantil
   de 53%.

---

## 1. Definição do problema analítico

### 1.1 Alvo

O enunciado pede a previsão de **se um aluno será considerado alfabetizado ou
não**. A camada Gold é agregada — o menor grão publicado pelo INEP é *município ×
rede × ano*, e não existe microdado por aluno para este indicador.

A solução adotada mantém a pergunta e ajusta o grão:

```
alfabetizado = 1  se  taxa_alfabetizacao(municipio, rede municipal, 2024) >= 60%
```

**Justificativa do corte em 60%.** A camada Gold traz a série nacional: o Brasil
fechou 2024 com 59,2% e a meta nacional pactuada para aquele ano era 59,9%.
Arredondar para 60% significa rotular como ALFABETIZADO o município que opera no
patamar nacional ou acima dele. O corte também deixa as classes equilibradas
(56,9% de positivos), o que evita métricas infladas por desbalanceamento.

**A ponte para o aluno.** A probabilidade prevista é calibrada por regressão
isotônica e validada contra a taxa real de cada município. A correlação de
Pearson entre probabilidade prevista e taxa observada é de **0,766** (Spearman
0,782), e a curva de calibração cai sobre a diagonal. Isso autoriza ler a saída
como "chance de um aluno daquele município terminar o 2º ano alfabetizado", com
a ressalva de que o modelo não distingue dois alunos da mesma cidade.

### 1.2 Unidade de análise

Rede municipal, 2º ano do ensino fundamental, ano-alvo 2024. A rede municipal
concentra a matrícula dessa etapa e é onde o Compromisso Nacional Criança
Alfabetizada efetivamente atua.

**5.396 municípios** entram na análise — os que têm resultado publicado em 2023 e
2024 para a rede municipal.

---

## 2. Dados

### 2.1 Reconstrução da camada Gold

A pipeline Medalhão da Fase 2 roda em AWS Glue e grava Parquet particionado no
S3. Para que a Fase 3 seja reproduzível sem provisionar infraestrutura,
`src/data/gold_loader.py` reexecuta a mesma lógica em pandas sobre os CSVs
originais, preservando:

- o particionamento Hive-style `camada/entidade/ano=YYYY/`;
- a semântica de escrita `partitionOverwriteMode=dynamic` (apenas as partições do
  lote são sobrescritas — a pipeline continua idempotente);
- as regras de qualidade da Silver, com roteamento `pass`/`quarentena`;
- as quatro visões analíticas da Gold.

**Uma correção foi aplicada.** A visão `comparacao_metas_nacionais` agregava a
rede "total" (código 0) do indicador por UF, que existe em apenas 1 linha da base
— a visão saía com 1 registro. A taxa estadual passou a vir da tabela de metas
por UF, que traz o resultado da rede pública de cada estado em 2023 e 2024. Mesmo
conteúdo analítico, 50 registros, visão utilizável.

### 2.2 Enriquecimento externo

Quatro fontes públicas do IBGE, coletadas por `src/data/ibge.py` e versionadas em
`data/external/` para garantir reprodutibilidade offline:

| Fonte | Variáveis obtidas |
|---|---|
| API de Localidades | região, UF, mesorregião, microrregião, região intermediária |
| Censo 2022, tabela 4714 | população residente, área territorial, densidade |
| Censo 2022, tabela 9543 | taxa de alfabetização da população de 15+ anos |
| PIB dos Municípios, tabela 5938 | PIB municipal → PIB per capita |

A **alfabetização adulta** merece destaque: é um proxy do capital educacional das
famílias. Em municípios onde a geração adulta tem baixa alfabetização, a criança
recebe menos apoio de letramento em casa. E não é vazamento — mede os adultos de
2022, não as crianças avaliadas em 2024.

### 2.3 Qualidade dos dados

| Variável | Faltantes | Interpretação |
|---|---|---|
| `taxa_estadual_2023` | 80,3% | a maioria dos municípios não tem rede estadual com 2º ano — **a ausência é informação sobre o território** |
| `taxa_privada_2023` | 10,1% | municípios sem rede privada nessa etapa |
| `participacao_2023`, `nivel_alfabetizacao_2023` | 3,0% | municípios fora da tabela de metas |
| `meta_2025` e derivadas | 1,7% | idem |

Nenhum valor foi descartado por ausência. A imputação usa a **mediana** (robusta
às caudas longas de população, área e PIB) com `add_indicator=True`, que cria uma
coluna binária sinalizando a ausência — preservando a informação de que o dado
*não existia*.

---

## 3. Tratamento de data leakage

O alvo deriva da taxa de 2024. Toda medida da mesma avaliação está bloqueada.

| Bloqueado | Motivo |
|---|---|
| `taxa_alfabetizacao` (2024) | é o próprio alvo |
| `media_portugues` (2024) | mesma prova |
| `proporcao_aluno_nivel_*` (2024) | decomposição do alvo |
| `nivel_alfabetizacao` (2024) | classificação do resultado de 2024 |
| `gap_meta_2025`, `status_meta_2025` | calculados a partir do alvo |

| Permitido | Motivo |
|---|---|
| Indicadores de 2023 | histórico, medido um ano antes |
| Metas 2024–2030 | publicadas a partir da linha de base de 2023 |
| Censo 2022 (IBGE) | anterior ao alvo |

**Três defesas, todas executáveis:**

1. `validar_ausencia_de_vazamento()` levanta exceção se alguma coluna proibida
   aparecer entre os preditores ou se algum preditor tiver correlação superior a
   0,95 com o alvo. Resultado: **nenhuma coluna proibida**, maior correlação =
   0,651 (`media_portugues_2023`, histórico legítimo).
2. A separação treino/teste acontece **antes** de qualquer estatística ser
   calculada.
3. Todo o pré-processamento vive dentro do `Pipeline` e é reajustado dentro de
   cada fold da validação cruzada — imputador, scaler e encoder nunca veem o
   conjunto de validação.

---

## 4. Engenharia de atributos e poda de redundância

### 4.1 Variáveis derivadas

| Variável | Construção | Intenção analítica |
|---|---|---|
| `gap_2023_para_meta_2025` | taxa 2023 − meta 2025 | tamanho do esforço pela frente |
| `dif_municipal_estadual_2023` | taxa municipal − taxa estadual | isola gestão da rede do efeito do território |
| `dif_crianca_adulto` | taxa infantil − alfabetização 15+ | isola o efeito da escola do capital familiar |
| `percentil_uf_2023` | posição relativa dentro da UF | comparável entre UFs de tamanhos distintos |
| `porte_relativo_na_uf` | rank percentual de população na UF | escala de atendimento |
| `log_populacao`, `log_pib_per_capita` | log(1 + x) | comprime caudas longas |

### 4.2 A poda

A análise de correlação encontrou redundâncias severas:

| Removida | Correlação | Diagnóstico |
|---|---|---|
| `meta_2030` | — | **constante**: 80% para todos |
| `inclinacao_meta_anual` | −1,000 com `meta_2025` | é `(80 − meta_2025)/5` |
| `meta_2024` | 1,000 com `meta_2025` | duplicata |
| `meta_2025` | 0,966 com `taxa_alf_2023` | metas do INEP são função da linha de base de 2023 |
| `taxa_privada_2023` | 0,987 com `taxa_alf_2023` | artefato de preenchimento da base |
| `gap_2023_para_meta_2024` | 0,975 com `gap_..._2025` | duplicata |

**Sobre as metas.** As metas municipais são calculadas pelo INEP a partir do
resultado de 2023, então carregam quase a mesma informação que ele. Mantê-las
dividia a importância entre variáveis sinônimas e escondia o desempenho
histórico — que é o preditor conceitualmente central. O que sobra da meta e não
está na taxa é a distância entre as duas, preservada em
`gap_2023_para_meta_2025`.

**Sobre a rede privada.** Em municípios sem rede privada de 2º ano, a base do
INEP repete o valor da rede municipal na linha da rede privada. É preenchimento,
não coincidência — e usá-la seria duplicar a variável histórica.

Multicolinearidade não derruba a acurácia, mas arruína a interpretação. Como
metade da entrega deste projeto é explicar *por que* um município foi priorizado,
a poda é obrigatória.

**Conjunto final: 20 preditores** (16 numéricos, 4 categóricos), que o
pré-processamento expande para 64 colunas.

---

## 5. Modelagem supervisionada

### 5.1 Pipeline

```
Pipeline
├── ColumnTransformer
│   ├── numéricas   → SimpleImputer(mediana, add_indicator) → StandardScaler
│   └── categóricas → SimpleImputer(moda) → OneHotEncoder(ignore, min_freq=10)
└── Estimador
```

`remainder="drop"`: nada entra sem estar declarado.

### 5.2 Comparação dos candidatos

Validação cruzada estratificada, 5 folds, sobre 4.316 municípios de treino.

| Modelo | ROC AUC validação | ROC AUC treino | Overfitting | F1 | PR AUC | Tempo |
|---|---|---|---|---|---|---|
| Random Forest | 0,9032 ± 0,0115 | 0,9760 | 0,0728 | 0,8447 | 0,9210 | 12,3 s |
| Extra Trees | 0,9030 ± 0,0112 | 0,9503 | 0,0472 | 0,8443 | 0,9211 | 10,1 s |
| Gradient Boosting | 0,9012 ± 0,0113 | 0,9680 | 0,0668 | 0,8359 | 0,9205 | 2,2 s |
| **Regressão Logística** | **0,9008 ± 0,0139** | 0,9067 | **0,0060** | **0,8462** | 0,9152 | 0,6 s |
| Árvore de Decisão | 0,8798 ± 0,0110 | 0,9078 | 0,0280 | 0,8345 | 0,8837 | 0,5 s |

### 5.3 A escolha: regra do erro-padrão

Os quatro primeiros ficam separados por milésimos de ROC AUC, enquanto o desvio
entre folds é de ~0,011. A diferença está dentro do ruído.

Aplicamos a *one standard error rule*: entre os modelos a menos de um
desvio-padrão do melhor, fica o que **menos decorou o treino**. Os ensembles
chegam a 0,97 no treino contra 0,90 na validação. A Regressão Logística entrega
o mesmo desempenho de validação com diferença de 0,006, o melhor F1 do conjunto,
e roda 20 vezes mais rápido.

Em política pública essa escolha vale duas vezes: o modelo mais simples
generaliza melhor fora da amostra e é defensável diante de um gestor.

**Hiperparâmetros** (RandomizedSearchCV, 40 combinações, métrica ROC AUC):
`C = 5`, `penalty = l2`, `class_weight = balanced`.

### 5.4 Calibração

Regressão isotônica com validação cruzada interna ao treino. O Brier melhora de
0,1256 para **0,1246** e a curva de calibração passa a cair sobre a diagonal.

| | Sem calibração | Calibrado |
|---|---|---|
| ROC AUC | 0,9029 | 0,9024 |
| Brier | 0,1256 | **0,1246** |
| F1 | 0,8412 | 0,8422 |

O ROC AUC praticamente não muda — era esperado, porque a calibração preserva o
ordenamento. O ganho está na **utilizabilidade da probabilidade**: sem ela, o
escore serve para ranquear; com ela, serve para dizer "este município tem 30% de
chance de alcançar o patamar".

### 5.5 Escolha do limiar

O limiar de 0,50 é convenção matemática, não decisão de gestão:

- classificar como "vai bem" um município em risco → **criança sem política**;
- classificar como "em risco" um município que ia bem → **orçamento
  desperdiçado**, recuperável.

Ponderando o primeiro erro com peso 3 e o segundo com peso 1, o custo mínimo fica
em **0,77**:

| Limiar | Acurácia | Precisão | Recall | F1 | Detecção de risco | Municípios em risco não detectados |
|---|---|---|---|---|---|---|
| 0,50 (padrão) | 0,8213 | 0,8470 | 0,8374 | 0,8422 | 0,800 | 93 |
| **0,77 (custo de política)** | 0,7907 | 0,9294 | 0,6846 | 0,7884 | **0,931** | **32** |

A troca é explícita: perde-se 3 pontos de acurácia e ganha-se 13 pontos de
detecção de risco — de 93 para 32 municípios em risco que passariam despercebidos.
Para uma política de priorização, esse é o lado certo do trade-off.

### 5.6 Generalização

**Curva de aprendizado.** Treino e validação convergem sem vão persistente — o
modelo está bem dimensionado, sem sinal de variância excessiva.

**Validação geográfica.** Treinar deixando uma região inteira de fora e testar
exatamente nela:

| Região de teste | Municípios | ROC AUC | F1 | Acurácia |
|---|---|---|---|---|
| Nordeste | 1.779 | 0,780 | 0,600 | 0,721 |
| Norte | 413 | 0,770 | 0,528 | 0,697 |
| Sudeste | 1.612 | 0,738 | 0,750 | 0,670 |
| Centro-Oeste | 462 | 0,702 | 0,747 | 0,654 |
| Sul | 1.130 | 0,663 | 0,765 | 0,681 |

**Este é o resultado mais importante do relatório.** O AUC cai de 0,90 para
0,66–0,78. A explicação é direta: parte relevante do sinal está nas variáveis
territoriais, e ao remover a região inteira do treino o one-hot de UF e região
fica todo em zero — sobram apenas o histórico e o contexto socioeconômico.

**Consequência prática:** o modelo serve para priorizar municípios dentro do
Brasil de hoje, não para extrapolar a um território de estrutura desconhecida.
Para uso federal isso basta, porque todas as regiões estão representadas.

---

## 6. Interpretabilidade

### 6.1 Importância por permutação (conjunto de teste)

| # | Variável | Queda no ROC AUC | Desvio |
|---|---|---|---|
| 1 | Unidade da Federação | 0,1831 | 0,0149 |
| 2 | Proficiência em Português em 2023 | 0,0556 | 0,0074 |
| 3 | Percentil do município dentro da UF em 2023 | 0,0439 | 0,0069 |
| 4 | Região do país | 0,0300 | 0,0025 |
| 5 | Distância entre 2023 e a meta de 2025 | 0,0132 | 0,0035 |
| 6 | Participação na avaliação de 2023 | 0,0125 | 0,0026 |
| 7 | Posição relativa de população na UF | 0,0088 | 0,0018 |

Usamos permutação em vez da importância nativa por dois motivos: ela é calculada
no **conjunto de teste** (mede generalização, não ajuste) e não sofre o viés que
infla variáveis de alta cardinalidade.

### 6.2 Valores SHAP

Os valores SHAP confirmam o ranking e acrescentam o **sinal** de cada
contribuição, além de permitir explicar predições individuais. Para um município
específico, o modelo responde: "risco alto porque o resultado de 2023 ficou 18
pontos abaixo da meta, a participação na avaliação foi de 71% e o município está
no último quartil da sua UF".

Sem essa resposta caso a caso, a lista de prioridades vira alvo de disputa
política em vez de instrumento de gestão.

### 6.3 Leitura substantiva

**O território pesa mais que tudo.** A UF responde por três vezes a importância
da segunda variável. Não é defeito do modelo — é retrato do país.

**O desempenho passado é o melhor preditor controlável.** Proficiência e percentil
na UF ocupam 2º e 3º lugares. Desempenho educacional é persistente: má notícia
para quem está mal, boa notícia para o planejamento, porque torna o risco
previsível com um ano de antecedência.

**A participação na avaliação aparece entre as sete primeiras.** Sinal
operacional: rede desorganizada para aplicar a avaliação costuma ser rede
desorganizada para alfabetizar.

---

## 7. Segmentação não supervisionada

### 7.1 Método e decisões

KMeans sobre o perfil municipal padronizado, com duas diferenças deliberadas em
relação ao modelo supervisionado:

- **sem indicador de ausência** — com ele, as flags viravam a dimensão dominante
  e o KMeans formava um "perfil" que era só o grupo de municípios sem meta
  cadastrada: artefato de cadastro, não realidade educacional;
- **com winsorização a 1%/99%** — sem truncar as caudas, a partir de k = 4 o menor
  grupo ficava com menos de 1% dos municípios, porque cidades com PIB inflado por
  royalties e capitais densíssimas consumiam clusters inteiros.

**Escolha de k.** A silhueta fica entre 0,18 e 0,24 para todo k testado — quase
plana. Isso não é defeito do método: dados socioeconômicos municipais formam um
gradiente contínuo, não ilhas separadas. Quando a silhueta não discrimina, usá-la
como critério é escolher ruído, e ela sempre puxa para k = 2, grosseiro demais
para orientar política.

O critério principal passou a ser o **cotovelo da inércia** (formalizado como o k
mais distante da reta que liga os extremos da curva), com a restrição de negócio
de que nenhum perfil tenha menos de 2% dos municípios. Resultado: **k = 4**.

### 7.2 Os perfis

| Perfil | Municípios | Taxa 2024 | Taxa 2023 | PIB per capita | Pop. mediana | Alfab. adulta | Região |
|---|---|---|---|---|---|---|---|
| Risco crítico — baixa renda | 1.650 | 46,5% | 39,4% | R$ 13,3 mil | 13,2 mil | 81,7% | Nordeste |
| Desempenho intermediário — interior | 2.269 | 65,4% | 62,0% | R$ 37,9 mil | 13,0 mil | 93,5% | Sudeste |
| Alto desempenho — interior | 1.356 | 79,4% | 83,3% | R$ 20,2 mil | 6,9 mil | 86,3% | Nordeste |
| Risco moderado — centros urbanos | 121 | 53,2% | 52,2% | R$ 37,6 mil | 264 mil | 96,3% | Sudeste |

### 7.3 Achados

**O quarto perfil é o contraintuitivo.** 121 municípios grandes, com PIB per
capita alto e 96,3% de alfabetização adulta, mas taxa infantil de 53% — abaixo da
média nacional. Renda e escolaridade dos pais não estão bastando nessas cidades:
o gargalo é de rede, escala e gestão.

**O perfil de melhor desempenho é nordestino e de baixa renda.** Com R$ 20,2 mil
de PIB per capita e 86,3% de alfabetização adulta, atinge 79,4% de alfabetização
infantil — muito acima de municípios com o dobro da renda.

Os dois achados juntos derrubam a leitura de que alfabetização é consequência
direta de riqueza.

**Validação.** Embora construída sem olhar para o alvo, a segmentação separa a
proporção de municípios no patamar nacional de menos de 20% a quase 90%. É essa
validação que autoriza usar os clusters como contexto do agente de RL.

**Outliers.** O DBSCAN identificou 7 municípios que não cabem em nenhum perfil —
casos para análise individual, não para programa padronizado.

---

## 8. Séries temporais e projeção de metas

### 8.1 A limitação que define o método

| Nível | Pontos observados |
|---|---|
| Brasil | 3 (2023, 2024, 2025) |
| UF | 2 (2023, 2024) |
| Município | 2 (2023, 2024) |

Com dois ou três pontos não há sazonalidade a estimar nem autocorrelação a
modelar. ARIMA e Prophet seriam teatro estatístico. Declarar isso é parte da
entrega.

### 8.2 Projeção nacional — três cenários

| Ano | Amortecido (referência) | Holt | Linear | Meta |
|---|---|---|---|---|
| 2026 | 70,3% | 70,5% | 70,5% | 67% |
| 2027 | 73,9% | 75,5% | 75,5% | 71% |
| 2028 | 77,0% | 80,6% | 80,6% | 74% |
| 2029 | 79,7% | 85,6% | 85,6% | 77% |
| **2030** | **81,9%** | 90,7% | 90,7% | **80%** |

**Holt e linear produzem a mesma reta.** Com três observações, o modelo de série
temporal não acrescenta informação à tendência simples — resultado que precisa
ser reportado, não escondido.

**A extrapolação linear leva o Brasil a 90,7% em 2030**, mais de 10 pontos acima
da meta. Nenhum sistema educacional do mundo avançou tanto em seis anos. O número
não é previsão: é o que acontece ao projetar uma reta ignorando o teto de 100%.

**O cenário amortecido é a referência**: o ganho anual encolhe a cada ano
projetado, refletindo que os primeiros pontos percentuais são os mais fáceis.
Chega a 81,9% — acima da meta, por margem estreita.

**Conclusão para o gestor: a meta nacional é alcançável, mas não está garantida.**

### 8.3 Projeção municipal

Dois ajustes obrigatórios:

- **Encolhimento (shrinkage).** A variação individual 2023→2024 tem desvio de
  16,7 p.p., muito por conta de municípios pequenos onde poucas turmas mudam o
  percentual inteiro. Metade do peso vai para o comportamento médio da UF.
- **Amortecimento (damping).** O ganho projetado cai a cada ano
  (`delta × 0,85^t`). Sem o freio, um município que subiu 12 p.p. em um ano
  passaria de 100% em 2030.

| Classificação de ritmo | Municípios | % |
|---|---|---|
| No ritmo da meta | 2.541 | 47,1% |
| **Em retrocesso** | **2.250** | **41,7%** |
| Ritmo crítico | 365 | 6,8% |
| Ritmo insuficiente | 240 | 4,4% |

**Municípios que atingem a meta de 2030 na projeção: 2.444 (45,3%).**

**O número que deve abrir a apresentação executiva.** O Brasil supera a meta
nacional no agregado enquanto 4 em cada 10 municípios andam para trás. O país
avança porque os municípios grandes puxam a média. **Política pública desenhada
sobre a média nacional não alcança justamente quem mais precisa.**

---

## 9. Aprendizado por reforço para alocação

### 9.1 Formulação

| Elemento | Neste projeto |
|---|---|
| Estado (contexto) | Perfil do município (cluster do capítulo 7) |
| Ações | 4 intervenções educacionais, com custos relativos diferentes |
| Recompensa | Ganho médio, em p.p., no lote em que a intervenção foi aplicada |
| Objetivo | Maximizar ganho acumulado = minimizar arrependimento |

**Decisão de modelagem que mudou o resultado.** Uma rodada não é um município, é
um **lote**: um programa é contratado para dezenas de cidades do mesmo perfil e o
gestor observa o ganho médio. Isso importa numericamente — o ruído de execução
por município é de 1,5 p.p., e se cada decisão observasse um único município esse
ruído engoliria efeitos de menos de 1 p.p., impedindo qualquer agente de
distinguir as intervenções. Com lotes de 25, o desvio cai pela raiz do tamanho e
o problema fica bem-posto.

### 9.2 Os agentes

| Agente | Arrependimento final | Ganho total (p.p.) | Taxa de acerto | Perfis com política correta |
|---|---|---|---|---|
| **UCB (variante UCB-V)** | **19,8** | 11.443 | **96,8%** | 3 de 4 |
| Epsilon-Greedy | 217,2 | 11.248 | 91,9% | 3 de 4 |
| Thompson Sampling | 364,3 | 11.110 | 72,1% | 4 de 4 |

**Sobre o UCB.** Usamos a variante com bônus escalado pela variância observada.
O UCB1 original supõe recompensas em [0, 1]; aqui são ganhos em pontos
percentuais. Com a constante fixa do UCB1, o bônus ficava enorme perto de
recompensas pequenas e o agente explorava indefinidamente — comportamento que
observamos antes da correção e que tornava a comparação injusta.

O UCB converge mais rápido porque concentra a exploração onde a incerteza está,
em vez de espalhá-la ao acaso como o epsilon-greedy.

### 9.3 A política aprendida

A política difere por perfil, que é o ponto: uma secretaria nacional que manda o
mesmo programa para todo mundo desperdiça orçamento onde aquele programa rende
pouco.

O retorno por unidade de orçamento é sistematicamente maior nos perfis de risco
crítico — eles estão longe do teto de saturação. **Investir onde o indicador já
está alto rende pouco, matematicamente e na prática.**

### 9.4 Alocação

Com a política aprendida, a alocação vira um problema de mochila: a prioridade de
cada município combina o ganho esperado da intervenção do seu perfil, o risco
previsto pelo modelo supervisionado e o custo.

Um orçamento fixo alcança **1.200 municípios com taxa média de 42,0%**, contra
uma média geral de 62,9%. A regra de prioridade se comporta como política
redistributiva sem que isso tenha sido imposto por regra.

### 9.5 Honestidade sobre o ambiente

Não existe base pública com o efeito causal de cada programa por município. O
ambiente é um **simulador calibrado nos dados reais** — a magnitude dos ganhos
vem da variação real 2023→2024 de cada cluster, o retorno é decrescente conforme
a taxa se aproxima de 100%, e a matriz de efeito por (perfil × intervenção) é
sorteada uma vez e fica oculta do agente.

O experimento demonstra a **mecânica de decisão**, não estima o impacto de
nenhum programa real. Com dados de execução e alguma aleatorização na alocação,
o mesmo código roda sem alteração.

---

## 10. Limitações

| Limitação | Impacto | Mitigação adotada |
|---|---|---|
| Grão municipal, não individual | Não distingue dois alunos da mesma cidade | Calibração + validação contra a taxa real (r = 0,77) |
| Queda do AUC fora da região treinada | Não extrapola a territórios desconhecidos | Declarado; uso restrito à priorização dentro do Brasil |
| Dois pontos por município na série | Sem previsão pontual confiável | Entrega classificação de ritmo, não previsão; encolhimento + amortecimento |
| Sem intervalo de confiança na projeção | Incerteza não quantificada formalmente | Três cenários nacionais em vez de intervalo falsamente preciso |
| Ambiente de RL simulado | Não estima impacto real de programas | Declarado no código, no notebook e aqui |
| Relações associativas, não causais | "Impacta" é abuso de linguagem | Declarado; evoluções propõem desenho causal |
| 5.396 de 5.570 municípios | Faltam os que não participaram | Viés provavelmente subestima o problema — ausentes tendem a ser os mais frágeis |
| Amortecimento de 0,85 é escolha | Sensibilidade não testada | Ancorado na saturação esperada; série mais longa permitiria estimar |

---

## 11. Conclusão

O projeto entrega três artefatos operacionais — o ranking de risco, os perfis
municipais e a projeção de ritmo até 2030 — construídos sobre dados públicos, com
pipeline reproduzível e limitações declaradas.

O modelo alcança ROC AUC de 0,902 com probabilidades calibradas, e escolhemos
deliberadamente o algoritmo **mais simples** entre os estatisticamente empatados,
porque um escore que decide onde a política pública vai chegar precisa ser
explicável ao gestor cuja cidade aparece na lista.

O achado central não é técnico: **o Brasil está cumprindo a meta nacional
enquanto 4 em cada 10 municípios retrocedem.** O agregado melhora porque as
cidades grandes puxam a média. Qualquer política desenhada sobre esse agregado
vai passar ao largo de quem mais precisa — e é exatamente para isso que serve um
modelo de risco no grão do município.
