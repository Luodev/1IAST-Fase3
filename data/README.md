# Dicionário de dados

## Organização das pastas

| Pasta | Versionada? | Conteúdo |
|---|---|---|
| `raw/` | Sim | CSVs originais que alimentaram a camada Gold da Fase 2 |
| `external/` | Sim | Enriquecimento coletado das APIs do IBGE |
| `lake/` | Não | Data lake local (bronze/silver/gold) — regenerado por `run_pipeline.py` |
| `processed/` | Não | ABT em Parquet — regenerada por `run_pipeline.py` |

Versiona-se o que o código não consegue reproduzir sozinho. `lake/` e
`processed/` são derivados determinísticos e seriam apenas peso morto no
repositório.

---

## `raw/` — camada Gold da Fase 2

Origem: [github.com/Luodev/1IAST-Fase2](https://github.com/Luodev/1IAST-Fase2),
que por sua vez consome as publicações do INEP sobre o Indicador Criança
Alfabetizada.

### `br_inep_avaliacao_alfabetizacao_municipio.csv` — 23.995 linhas

| Coluna | Tipo | Descrição |
|---|---|---|
| `ano` | int | 2023 ou 2024 |
| `id_municipio` | texto(7) | Código IBGE do município |
| `serie` | int | Série avaliada (2 = 2º ano do fundamental) |
| `rede` | texto | 0 = total, 2 = estadual, 3 = municipal, 5 = privada |
| `taxa_alfabetizacao` | float | % de alunos no nível esperado |
| `media_portugues` | float | Proficiência média em Língua Portuguesa (escala Saeb) |
| `proporcao_aluno_nivel_0..8` | float | Distribuição dos alunos por nível (disponível só em parte das linhas) |

### `br_inep_avaliacao_alfabetizacao_uf.csv` — 145 linhas

Mesmo esquema, com `sigla_uf` no lugar de `id_municipio`.

### `br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv` — 10.704 linhas

| Coluna | Tipo | Descrição |
|---|---|---|
| `ano` | int | 2023 ou 2024 |
| `id_municipio` | texto(7) | Código IBGE |
| `rede` | texto | Sempre "Municipal" |
| `taxa_alfabetizacao` | float | Resultado da rede municipal naquele ano |
| `meta_alfabetizacao_2024..2030` | float | Trajetória pactuada — **constante por município** |
| `nivel_alfabetizacao` | int | Classificação INEP do resultado (0 a 5) |
| `percentual_participacao` | float | % de alunos que fizeram a avaliação |

> As metas são calculadas pelo INEP a partir da linha de base de 2023, o que
> explica a correlação de 0,97 entre `meta_2025` e a taxa daquele ano. Ver a
> seção 6.2 do README.

### `br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv` — 54 linhas
### `br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv` — 3 linhas

Série nacional (2023: 55,9% · 2024: 59,2% · 2025: 66,0%) e as metas do
Compromisso Nacional Criança Alfabetizada até 2030 (80%).

---

## `external/` — enriquecimento IBGE

### `ibge_localidades.csv` — 5.571 municípios

Hierarquia territorial completa: `nome_municipio`, `sigla_uf`, `nome_uf`,
`regiao`, `mesorregiao`, `microrregiao`, `regiao_intermediaria`.
Fonte: API de Localidades do IBGE.

### `ibge_demografia_censo2022.csv` — 5.571 municípios

`populacao_2022`, `area_km2`, `densidade_hab_km2`.
Fonte: Censo Demográfico 2022, tabela SIDRA 4714.

### `ibge_alfabetizacao_adulta_censo2022.csv` — 5.571 municípios

`taxa_alfabetizacao_15mais` — taxa de alfabetização da população de 15 anos ou
mais. Proxy do capital educacional das famílias.
Fonte: Censo Demográfico 2022, tabela SIDRA 9543.

### `ibge_pib_2021.csv` — 5.571 municípios

`pib_mil_reais` — PIB municipal a preços correntes.
Fonte: PIB dos Municípios, tabela SIDRA 5938.

### `contexto_municipal.csv`

Consolidação das quatro fontes acima, com `pib_per_capita` calculado.

---

## `processed/abt_alfabetizacao.parquet` — a ABT

**5.396 linhas × 35 colunas.** Uma linha por município com rede municipal e
resultado publicado em 2023 e 2024.

### Identificação

| Coluna | Descrição |
|---|---|
| `id_municipio`, `nome_municipio`, `sigla_uf`, `regiao`, `regiao_intermediaria` | Identificação e território |

### Alvos

| Coluna | Descrição |
|---|---|
| `taxa_alvo` | Taxa de alfabetização da rede municipal em 2024 (%) |
| `alfabetizado` | **Alvo principal** — 1 se `taxa_alvo >= 60` |
| `atinge_meta_2025` | Alvo secundário — 1 se `taxa_alvo >= meta_2025`. Não entra no treino |

### Preditores usados no modelo (20)

**Histórico de 2023**

| Coluna | Descrição |
|---|---|
| `taxa_alf_2023` | Taxa da rede municipal em 2023 (%) |
| `media_portugues_2023` | Proficiência média em Português (escala Saeb) |
| `participacao_2023` | % de alunos que fizeram a avaliação |
| `ranking_uf_2023` | Posição do município dentro da UF |
| `percentil_uf_2023` | Posição relativa (0 = pior, 1 = melhor) |
| `taxa_estadual_2023` | Taxa da rede estadual local (ausente em 80,3% — a maioria não tem) |
| `dif_municipal_estadual_2023` | Diferença entre as duas redes (p.p.) |
| `dif_municipal_privada_2023` | Diferença entre municipal e privada (p.p.) |
| `nivel_alfabetizacao_2023` | Classificação INEP de 2023 (categórica) |

**Metas**

| Coluna | Descrição |
|---|---|
| `gap_2023_para_meta_2025` | `taxa_alf_2023 − meta_2025` (p.p.) |

**Socioeconômicas**

| Coluna | Descrição |
|---|---|
| `log_populacao` | log(1 + população do Censo 2022) |
| `area_km2` | Área territorial |
| `densidade_hab_km2` | Densidade demográfica |
| `taxa_alfabetizacao_15mais` | Alfabetização adulta (Censo 2022, %) |
| `log_pib_per_capita` | log(1 + PIB per capita de 2021) |
| `dif_crianca_adulto` | `taxa_alf_2023 − taxa_alfabetizacao_15mais` (p.p.) |
| `porte_relativo_na_uf` | Rank percentual de população dentro da UF |

**Categóricas**

`sigla_uf` (27) · `regiao` (5) · `porte_municipio` (7 faixas) ·
`nivel_alfabetizacao_2023` (6 níveis)

### Colunas presentes na ABT mas **fora** do modelo

Removidas por redundância (ver seção 6.2 do README): `meta_2024`, `meta_2025`,
`meta_2030`, `inclinacao_meta_anual`, `taxa_privada_2023`,
`gap_2023_para_meta_2024`. Permanecem na ABT porque a clusterização e a projeção
de metas as utilizam.

---

## `lake/` — data lake local

Estrutura gerada por `src/data/gold_loader.py`, espelhando a Fase 2:

```
lake/
├── bronze/<entidade>/ano=YYYY/dados.parquet
├── silver/
│   ├── pass/<entidade>/ano=YYYY/dados.parquet
│   └── quarentena/<entidade>/anomesdia=YYYYMMDD/dados.parquet
└── gold/<visao>/ano=YYYY/dados.parquet
```

Entidades: `indicador_municipio`, `indicador_uf`, `meta_brasil`, `meta_uf`,
`meta_municipio`.
Visões Gold: `alfabetizacao_por_municipio`, `evolucao_temporal`,
`ranking_municipios`, `comparacao_metas_nacionais`.

Colunas com prefixo `_` são metadados de linhagem e qualidade
(`_record_hash`, `_ingestion_timestamp`, `_dq_*`, `_gold_processed_at`).
