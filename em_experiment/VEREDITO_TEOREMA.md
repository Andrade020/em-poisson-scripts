# Veredito da falsificação empírica do Teorema 1 (21/07/2026)

Scripts: `verifica_teorema.py` (fase 1) e `verifica_teorema_fase2.py` (fase 2).
Resultados brutos: `resultados_verifica_teorema.json`, `resultados_verifica_teorema_fase2.json`.
Desenho: I=6, T=4 (24 parâmetros λ + 4 p), Q = Laplaciano conexo (ker Q = span{1}),
verdades fundida e não fundida conhecidas, R=4000 réplicas/cenário (1000 nos auxiliares),
~100 mil ajustes convexos, N ∈ {50, ..., 12800}. Tempo total ≈ 9 min.

## Resultado por regime — NENHUMA REFUTAÇÃO

| regime | previsão testada | veredito |
|---|---|---|
| (i) $a_N=N^{1/4}$ | normalidade + cobertura 90% + cov → $I^{-1}$ | **Confirmado**: cobertura 0,932→0,892 (→0,90); erro de cov 0,70→0,10 (decai ~$N^{-1/2}$); viés residual decai na razão $(1/4)^{1/4}$=0,707 por 4×N, exatamente a taxa $a_N/\sqrt N$ prevista; na fase 2, viés bate com o alvo-γ finito (corr 0,9985, z máx 2,4) |
| (ii) $a_N=c\sqrt N$ | viés-limite $-c\,I^{-1}Q\theta^0$ | **Confirmado no mecanismo**: em amostra finita o viés é $\sqrt N(\theta^*_{\gamma_N}-\theta^0)$ com $\gamma_N=c/\sqrt N$ — a fase 2 mostra correlação 0,99996–1,00000 e normas idênticas a 3–4 casas (ex. 16,753 vs 16,760) em todos os 6 cenários (c∈{1,4}, N∈{200,800,3200}); a fórmula-limite é o γ→0 disso, com convergência lenta quando $\|I^{-1}Q\theta^0\|$ é grande (direção converge: corr 0,94→0,99) |
| (iii-a) $a_N=aN$, fundida | cov → $(I+aQ)^{-1}I(I+aQ)^{-1}$ | **Confirmado e discriminante**: erro 4–5% contra o sanduíche vs 95% contra o $I^{-1}$ ingênuo; viés z máx 1,0–3,3 |
| (iii-b) $a_N=aN$, não fundida | $\hat\theta\to\theta^*_a$; normal no alvo com $\bar H_a^{-1}\Sigma_a\bar H_a^{-1}$ | **Confirmado** (o passo reescrito da prova): dist. ao alvo 0,005→0,0006 (alvo a 1,377 de $\theta^0$); cov sanduíche analítica a 3–5%; z máx 4,3→2,2 |
| (iv-a) $a_N=wN^2$, fundida | colapso ortogonal + var(ker Q) = $[\Pi(\Pi'I\Pi)^{-1}\Pi']$ | **Confirmado**: razão 0,984/0,998/1,015; var ortogonal 0,028→0,0003 |
| (iv-b) $a_N=wN^2$, não fundida | inconsistência; acumulação em ker Q | **Confirmado**: dist. a $\theta^0$ 1,68→1,86 (não cai); dist. a ker Q 0,36→0,037 |
| intermediário $a_N=N^{3/4}$ | consistente, taxa lenta | **Consistente** (dist. cai 1,18→0,74 até N=12800); a taxa exata $N^{-1/4}$ é assintótica demais para confirmar nestes N ($\|I^{-1}Q\theta^0\|=17$ ⇒ regime linear só para N≫10^5) — sem contradição |

## Leitura honesta

1. A tentativa de refutação foi genuína (previsões quantitativas exatas, componente a
   componente, com poder estatístico para detectar erro de sinal, constante ou matriz)
   e o teorema sobreviveu em todas as frentes, incluindo o passo da prova que havia
   sido corrigido após a primeira revisão adversarial (iii-b).
2. Os "desvios" da fase 1 nos regimes (i)/(ii) são o efeito de segunda ordem
   $\gamma_N = a_N/N$ previsto pela própria teoria — e o fato de a previsão de amostra
   finita bater a 4 casas decimais é evidência mais forte do que a fórmula-limite
   sozinha daria.
3. O que a simulação NÃO cobre: dimensão crescente, desenhos não balanceados,
   penalidade no bloco p (testado só no bloco λ), e a taxa exata do regime
   intermediário. Nenhum desses é afirmado pelo teorema além do que foi testado,
   exceto o bloco p (coberto por simetria da estrutura, mas não simulado).
