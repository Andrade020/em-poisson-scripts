# em-poisson-scripts

Replication package for:

> **Penalized Estimation of Spatiotemporal Poisson Processes with Missing
> Marks: EM Algorithms, Asymptotic Theory, and Empirical Validation.**
> Lucas Rafael de Andrade (PPGCE/UERJ). Working paper, 2026.

The paper builds on the models of Guigues, Kleywegt, Nascimento & de Andrade,
*Estimation of spatiotemporal Poisson processes with missing data*,
Scientific Reports 16:19120 (2026), DOI 10.1038/s41598-026-46520-y.

Contact: `lucas_r_andrade@hotmail.com` / `rafael.lucas@posgraduacao.uerj.br`.

## Environment

Python 3.10.11, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, matplotlib
(see `requirements.txt`). Reported timings: Intel Core i7-1355U, 32 GB RAM,
Windows 11; wall-clock values in the paper are medians of 3 runs. All random
seeds are fixed inside the scripts.

## Data acquisition (not tracked in this repo)

All data are public. From `dados/`, download:

```bash
# NYC motor-vehicle collisions (Socrata)
curl -sL -o nyc_collisions.csv "https://data.cityofnewyork.us/resource/h9gi-nx95.csv?\$select=crash_date,crash_time,borough,latitude,longitude,number_of_persons_injured,number_of_persons_killed&\$limit=3000000"
# Chicago crimes 2019+
curl -sL -o chicago_crimes.csv "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?\$select=date,primary_type,latitude,longitude,community_area&\$where=date>'2019-01-01T00:00:00'&\$limit=3000000"
# Seattle Fire 911
curl -sL -o seattle_fire911.csv "https://data.seattle.gov/resource/kzjm-xkqj.csv?\$select=datetime,type,latitude,longitude&\$limit=2500000"
# Rio EMS data of the base paper (aggregated files, public repository)
git clone --depth 1 https://github.com/vguigues/LASPATED_Replication.git
# Brazilian federal-highway accidents (DATATRAN/PRF, yearly files via
# Google Drive links scraped from the PRF open-data page)
python probe_datatran.py    # requires drive_ids.txt built from the PRF page; see script header
```

Note: open-data snapshots evolve; the `resultados_*.json` files preserve the
exact outputs used in the paper (downloaded 2026-07-17/18).

## Script-to-table mapping

| Paper item | Script (working dir) |
|---|---|
| Table 1, Figure 1 (synthetic, penalized EM vs PG) | `em_experiment/em_vs_pg_model2.py` |
| Sec. 5.2 (exact mixture EM, scales + multistart) | `em_experiment/em_vs_pg_model4.py [I T N]` |
| Table 2 (four-solver benchmark, real data) | `dados/bench_solvers.py` |
| Coverage for lambda and p (Sec. 6.2) | `em_experiment/coverage_sim.py` |
| Table 3 (amputation + coverage) | `dados/semisintetico.py`, `dados/datatran_analise.py` |
| Penalized amputation, sparse RJ grid | `dados/amputacao_rj_pen.py` |
| Table 4 (dose-response OOS) | `dados/pipeline.py {nyc,chicago,seattle}`, `dados/rio_oos_random.py`; fold SEs in `dados/resultados_dose_se.json` |
| Table 5, Figure 2 (natural-mechanism MNAR validation) | `dados/validacao_natural.py`, `dados/validacao_natural_oos.py` |
| Permutation test (zone-dependent missingness) | `dados/resultados_perm_mnar.json` (inline script documented in the JSON) |
| Sec. 6.4 (reparametrized cross-validation) | `dados/cv_repar.py` |
| Figures (PDF) | `paper/figures/make_figs.py` |

`dados/rio_paper_data.py` parses the aggregated Rio files from
`LASPATED_Replication/Missing_Data/Rect10x10`.

Two data-processing cautions documented in the paper's appendix and
implemented here: fold exposure must be measured as observed days / 7 (ISO
week counting deflates out-of-sample targets by up to 20%), and the final
observation periods of the Rio replication file have degenerate exposure, so
random week folds are used there.

## License

Code released for academic replication. Contact the author for other uses.
