# Investigação empírica: estimação de intensidades com localização faltante

Baixado e analisado em 2026-07-18.

## Fontes

| dataset | fonte | conteúdo | por que serve |
|---|---|---|---|
| `nyc_collisions.csv` | NYC Open Data (Socrata `h9gi-nx95`) | colisões de trânsito NYC, data/hora, borough, lat/long, feridos/mortos | lat/long faltante em fração relevante; borough presente em parte das linhas sem coordenada → **teste MNAR** (p depende da zona?); severidade = tipos |
| `chicago_crimes.csv` | Chicago Data Portal (`ijzp-q8t2`), 2019+ | crimes, data/hora, tipo, lat/long, community area | community area quase sempre presente mesmo sem lat/long → **teste MNAR direto** |
| `seattle_fire911.csv` | Seattle Open Data (`kzjm-xkqj`) | chamados 911 do corpo de bombeiros (EMS/Fire), data/hora, tipo, lat/long | análogo mais próximo do problema EMS do artigo |
| `LASPATED_Replication/Missing_Data/Rect10x10` | GitHub vguigues/LASPATED_Replication | **os dados reais do artigo** (Rio 2016–2018): chegadas reportadas por (janela 30min × dia × zona 10×10 × prioridade × semana) + chamadas sem localização por (janela × dia × prioridade × semana) | benchmark no dado original do paper |

## Scripts

- `pipeline.py {nyc,chicago,seattle}` — perfil de missingness ($\hat p$ global/por hora-da-semana/por tipo,
  testes de deviance), **teste MNAR** ($p$ depende da zona rotulada?), viés de descarte, validação
  out-of-sample por mês (prever intensidade total por hora-da-semana: não corrigido vs corrigido com
  $p$ único vs corrigido com $p_t$), e modelo 2 penalizado (grade 10×10, T=168) com EM vs PG.
- `rio_paper_data.py` — replica no dado do Rio: $\hat p_{c,t}$ (Fig. 3 do artigo), subestimação por
  prioridade, EM vs PG no problema regularizado real (T=336, I=100), OOS nas últimas ~20% semanas.

Resultados em `resultados_<ds>.json` e consolidados em `../RELATORIO_EMPIRICO.md`.

## Desenho da validação (o que é honestamente testável)

1. **Estrutura de p**: o modelo do artigo assume $p_{c,t}$ (depende de tipo e tempo, não de zona).
   Testamos cada eixo com teste de deviance binomial.
2. **Teste da hipótese central (MAR vs MNAR espacial)**: onde existe rótulo de zona sem coordenada,
   testamos se a taxa de faltante varia por zona. Se varia (esperado), o modelo do artigo está
   mal-especificado nesses dados — quantificamos o quanto. Esta é a motivação empírica do "Ensaio 3".
3. **Correção fora da amostra**: a previsão do total (reportado + não reportado) por janela é
   observável no fold de teste; descartar dados sem local subestima por fator $(1-\hat p)$ —
   medimos o MSE relativo. (Nota: para prever *apenas os reportados* por zona, corrigido e não
   corrigido coincidem por construção — o ganho da correção está no processo latente total.)
4. **EM vs gradiente projetado em dado real** e ganho (ou não) da regularização fora da amostra.
