# Roteiro do vídeo executivo — 5 minutos

> **Apresentação:** [`apresentacao_executiva.pptx`](apresentacao_executiva.pptx) — 10 slides.
> A mesma fala abaixo está nas **notas do apresentador** de cada slide, com o tempo.
> Total de 5:00, a cerca de 100 palavras por minuto, o que deixa folga para pausas.

**Cenário:** reunião executiva com gestores públicos de educação.
**Tom:** direto, sem jargão. Quem está do outro lado decide orçamento, não escolhe algoritmo.

| # | Slide | Início | Duração |
|---|---|---|---|
| 1 | Quem vai ficar para trás? | 0:00 | 15 s |
| 2 | O Brasil melhora. 4 em cada 10 municípios pioram. | 0:15 | 30 s |
| 3 | A base: dados públicos do INEP e do IBGE | 0:45 | 40 s |
| 4 | Quatro técnicas, quatro respostas | 1:25 | 40 s |
| 5 | Supervisionado: 5 algoritmos, 1 escolhido | 2:05 | 40 s |
| 6 | O estado onde a criança nasce pesa mais que tudo | 2:45 | 25 s |
| 7 | Dinheiro não explica alfabetização | 3:10 | 30 s |
| 8 | A meta de 2030 é alcançável, mas não está garantida | 3:40 | 25 s |
| 9 | Como isso vira decisão de gestão | 4:05 | 30 s |
| 10 | A pergunta é o que fazemos com ela | 4:35 | 25 s |

---

## Bloco 1 — O problema

### Slide 1 · Capa (0:00 – 0:15)

> Bom dia. Sou o Lucas Oliveira. Vou mostrar como prever quais municípios vão
> ficar para trás na alfabetização das crianças — e o que fazer antes que isso
> aconteça.

### Slide 2 · O paradoxo (0:15 – 0:45)

**Na tela:** evolução do Brasil (55,9% → 59,2% → 66,0%) e o destaque de 41,7%.

> Primeiro, o contexto. O Brasil saiu de 55,9% das crianças alfabetizadas em 2023
> para 66% em 2025. Mas a média esconde o problema: entre 2023 e 2024, 41,7% dos
> municípios pioraram. Uma política desenhada sobre a média nacional não alcança
> quem mais precisa.

---

## Bloco 2 — Base de dados e modelos

### Slide 3 · A base de dados (0:45 – 1:25)

**Na tela:** fluxo Camada Gold (INEP) → IBGE → base analítica; o alvo e a regra
anti-vazamento.

> De onde vêm os dados. A base é a camada Gold que construímos na Fase 2, com o
> Indicador Criança Alfabetizada do INEP: resultados e metas por município,
> estado e Brasil. Enriquecemos com o IBGE — população, renda e alfabetização dos
> adultos, do Censo 2022. O resultado é uma base com 5.396 municípios e 20
> variáveis. O alvo é saber se o município alfabetiza pelo menos 60% das crianças
> em 2024, usando só informação de 2023 — assim o modelo nunca vê a resposta.

| Componente | Conteúdo |
|---|---|
| Camada Gold — Fase 2 (INEP) | 5 tabelas: indicador e metas por município, UF e Brasil; 23.995 registros municipais |
| Enriquecimento IBGE | Censo 2022 (população, área, densidade, alfabetização 15+) e PIB municipal |
| Base analítica | 5.396 municípios × 20 variáveis; alvo = taxa 2024 ≥ 60% (57% sim, 43% não) |
| Divisão | Treino com 4.316 municípios, teste com 1.080 nunca vistos |

### Slide 4 · Quatro técnicas, quatro respostas (1:25 – 2:05)

**Na tela:** quatro cartões, um por técnica, com a pergunta e o resultado.

> Com essa base, aplicamos quatro técnicas, cada uma respondendo uma pergunta. O
> modelo supervisionado prevê quem fica abaixo do patamar, com 90% de capacidade
> de separação. A clusterização encontrou quatro perfis de município. A projeção
> de séries temporais mostra que só 45% chegam à meta de 2030. E o aprendizado
> por reforço acertou a melhor intervenção em 96,8% das decisões.

| Técnica | Pergunta | Método | Resultado |
|---|---|---|---|
| Supervisionado | Quem vai ficar abaixo do patamar? | Regressão Logística calibrada | ROC AUC 0,90 |
| Não supervisionado | Quais municípios são parecidos? | KMeans com PCA | 4 perfis |
| Séries temporais | Quem chega à meta de 2030? | Tendência com amortecimento | 45% chegam |
| Aprendizado por reforço | Onde a verba rende mais? | Bandit contextual (UCB) | 96,8% de escolhas ótimas |

### Slide 5 · Resultados do modelo supervisionado (2:05 – 2:45)

**Na tela:** ROC AUC de treino x validação dos 5 algoritmos e as métricas no teste.

> No modelo supervisionado, comparamos cinco algoritmos. Repare que os mais
> complexos, como o Random Forest, vão muito bem no treino mas caem na validação:
> decoraram os dados. A Regressão Logística teve praticamente a mesma nota na
> validação, com doze vezes menos overfitting. No teste, com municípios que o
> modelo nunca viu, ela chegou a 0,90 de AUC, 84% de F1 e probabilidades
> calibradas.

| Algoritmo | ROC AUC treino | ROC AUC validação |
|---|---|---|
| Random Forest | 0,98 | 0,90 |
| Extra Trees | 0,95 | 0,90 |
| Gradient Boosting | 0,97 | 0,90 |
| **Regressão Logística** | **0,91** | **0,90** |
| Árvore de Decisão | 0,91 | 0,88 |

Métricas no conjunto de teste: **ROC AUC 0,90 · F1 0,84 · Acurácia 0,82 · Brier 0,125**.

### Slide 6 · Fatores mais influentes (2:45 – 3:10)

**Na tela:** influência relativa dos cinco principais fatores (estado = 100).

> E o que o modelo aprendeu? O fator mais influente, de longe, é o estado: pesa
> três vezes mais que o segundo, a proficiência em Português do ano anterior. A
> mesma criança tem chances muito diferentes dependendo de onde nasce.

---

## Bloco 3 — O que os resultados revelam

### Slide 7 · Dinheiro não explica alfabetização (3:10 – 3:40)

**Na tela:** taxa por perfil e o contraste interior de alto desempenho x centros urbanos.

> A clusterização trouxe a descoberta mais surpreendente: dinheiro não explica
> alfabetização. O melhor grupo, com 79% de crianças alfabetizadas, é de
> municípios pequenos e de baixa renda, a maioria no Nordeste. Já 121 centros
> urbanos, ricos e com adultos escolarizados, alfabetizam só 53%. Ali o gargalo é
> a rede de ensino.

### Slide 8 · Meta de 2030 (3:40 – 4:05)

**Na tela:** projeção do Brasil x meta pactuada e o destaque de 45%.

> Olhando para 2030: o Brasil deve chegar a 81,9%, menos de dois pontos acima da
> meta de 80%. A meta é alcançável, mas não está garantida — e só 45% dos
> municípios chegam lá no ritmo atual.

---

## Bloco 4 — Decisão e fechamento

### Slide 9 · Como isso vira decisão (4:05 – 4:35)

**Na tela:** três colunas — priorizar, personalizar, alocar verba.

> Isso vira decisão de três formas. Priorizar os 3.121 municípios abaixo do
> patamar, cada um com a explicação do porquê. Personalizar o programa conforme o
> perfil. E alocar a verba: na simulação, o recurso vai para municípios com taxa
> média de 42%, contra 63% no país.

### Slide 10 · Fechamento (4:35 – 5:00)

**Na tela:** a pergunta final e as duas ressalvas.

> Duas ressalvas: o modelo prioriza, não prediz destino, e mostra associação, não
> causa. Mas a informação para agir com um ano de antecedência já existe. A
> pergunta é o que fazemos com ela. Obrigado.

---

## Checklist de gravação

- [ ] Abrir o PPT no modo apresentador para ver as notas e o tempo de cada slide
- [ ] Ensaiar uma vez cronometrando: os slides 3 a 5 são os mais densos
- [ ] Desligar a tradução automática do navegador se mostrar o repositório
- [ ] Ter à mão, para perguntas: 5.396 municípios · 20 variáveis · alvo ≥ 60% ·
      treino 4.316 / teste 1.080 · ROC AUC 0,90 · limiar 0,77 por custo assimétrico
      3:1 · AUC cai para 0,66–0,78 quando uma região inteira fica fora do treino
