# Core benchmark: glossdex on public text collections (test splits)

Same EmbeddingGemma-300M vectors for every system. glossdex: result-set settings chosen once on three other collections (SciDocs, ArguAna, CQADupStack-webmasters; text_profile.json), then frozen. The threshold and Adaptive-k baselines' settings were chosen the same way (baseline_settings.json). No test collection was used for any setting. Probes: 50 queries from an unrelated collection per test collection, which the corpus cannot answer. BEIR judgments are sparse, so precision is a lower bound for every system alike.


## SciFact (test): 300 answerable queries, 50 probes

| system · cutoff | task | F2 | recall | precision | wrong/q | shown/q | probes: nothing shown | answerable: nothing shown | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **glossdex** | 70.2 | 65.2 | 71.5 | 65.2 | 1.06 | 1.85 | 100.0 | 10.0 | 78.2 |
| dense · top5 + threshold | 56.8 | 49.6 | 53.6 | 67.5 | 0.51 | 1.12 | 100.0 | 34.7 | 77.3 |
| dense · threshold | 55.2 | 47.7 | 51.6 | 70.4 | 0.51 | 1.10 | 100.0 | 39.0 | 77.3 |
| adaptive-k · B=1 (set on dev) | 53.5 | 62.4 | 81.5 | 35.4 | 2.19 | 3.10 | 0.0 | 0.0 | 77.3 |
| dense · top5 | 42.2 | 49.2 | 85.6 | 19.1 | 4.05 | 5.00 | 0.0 | 0.0 | 77.3 |
| hybrid · top5 | 40.0 | 46.6 | 81.5 | 17.9 | 4.10 | 5.00 | 0.0 | 0.0 | 75.3 |
| adaptive-k · B=5 (published) | 36.7 | 42.8 | 88.4 | 14.5 | 6.11 | 7.10 | 0.0 | 0.0 | 77.3 |
| bm25 · top5 | 36.1 | 42.1 | 73.9 | 16.0 | 4.20 | 5.00 | 0.0 | 0.0 | 67.7 |
| dense · top10 | 29.4 | 34.3 | 90.5 | 10.2 | 8.98 | 10.00 | 0.0 | 0.0 | 77.3 |
| hybrid · top10 | 28.3 | 33.0 | 87.6 | 9.8 | 9.02 | 10.00 | 0.0 | 0.0 | 75.3 |
| bm25 · top10 | 25.9 | 30.2 | 80.6 | 8.9 | 9.11 | 10.00 | 0.0 | 0.0 | 67.7 |

Paired differences, `glossdex` minus each (points, 95% CI):

| vs | task | F2 (answerable only) | nDCG@10 |
|---|---|---|---|
| dense · top5 + threshold | +13.4 (+9.9 to +17.0) | +15.6 (+11.7 to +19.8) | +0.9 (-0.2 to +2.0) |
| dense · threshold | +15.0 (+11.5 to +18.8) | +17.5 (+13.4 to +21.6) | +0.9 (-0.2 to +2.0) |
| adaptive-k · B=1 (set on dev) | +16.7 (+12.3 to +21.1) | +2.8 (-0.2 to +5.8) | +0.9 (-0.1 to +2.0) |
| dense · top5 | +28.0 (+23.4 to +32.6) | +16.0 (+12.0 to +19.9) | +0.9 (-0.2 to +2.0) |
| hybrid · top5 | +30.2 (+25.8 to +34.6) | +18.6 (+14.7 to +22.2) | +2.9 (+0.7 to +5.1) |
| adaptive-k · B=5 (published) | +33.5 (+29.0 to +37.9) | +22.4 (+18.4 to +26.4) | +0.9 (-0.1 to +2.0) |
| bm25 · top5 | +34.1 (+29.8 to +38.4) | +23.2 (+19.3 to +27.0) | +10.5 (+7.2 to +13.8) |
| dense · top10 | +40.9 (+36.2 to +45.3) | +31.0 (+26.5 to +35.3) | +0.9 (-0.1 to +2.0) |
| hybrid · top10 | +41.9 (+37.4 to +46.4) | +32.2 (+27.8 to +36.5) | +2.9 (+0.8 to +5.2) |
| bm25 · top10 | +44.3 (+40.0 to +48.5) | +35.1 (+30.8 to +39.2) | +10.5 (+7.3 to +13.8) |

## NFCorpus (test): 323 answerable queries, 50 probes

| system · cutoff | task | F2 | recall | precision | wrong/q | shown/q | probes: nothing shown | answerable: nothing shown | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| glossdex | 25.7 | 14.2 | 13.6 | 52.9 | 2.62 | 5.37 | 100.0 | 21.7 | 39.7 |
| dense · top5 + threshold | 16.9 | 4.0 | 3.7 | 68.7 | 0.29 | 0.74 | 100.0 | 70.6 | 39.3 |
| dense · threshold | 16.8 | 3.9 | 3.5 | 70.2 | 0.36 | 0.86 | 100.0 | 72.1 | 39.3 |
| dense · top10 | 14.2 | 16.4 | 20.0 | 29.3 | 7.07 | 10.00 | 0.0 | 0.0 | 39.3 |
| adaptive-k · B=5 (published) | 14.2 | 16.3 | 18.7 | 31.7 | 5.57 | 8.40 | 0.0 | 0.0 | 39.3 |
| hybrid · top10 | 12.7 | 14.7 | 17.7 | 26.9 | 7.31 | 10.00 | 0.0 | 0.0 | 36.7 |
| dense · top5 | 12.3 | 14.2 | 15.2 | 36.7 | 3.17 | 5.00 | 0.0 | 0.0 | 39.3 |
| adaptive-k · B=1 (set on dev) | 12.0 | 13.8 | 13.9 | 42.6 | 2.48 | 4.40 | 0.0 | 0.0 | 39.3 |
| hybrid · top5 | 11.5 | 13.3 | 14.3 | 34.7 | 3.27 | 5.00 | 0.0 | 0.0 | 36.7 |
| bm25 · top10 | 10.4 | 12.0 | 14.5 | 23.0 | 7.70 | 10.00 | 0.0 | 0.0 | 31.7 |
| bm25 · top5 | 9.8 | 11.3 | 12.1 | 30.0 | 3.50 | 5.00 | 0.0 | 0.0 | 31.7 |

Paired differences, `glossdex` minus each (points, 95% CI):

| vs | task | F2 (answerable only) | nDCG@10 |
|---|---|---|---|
| dense · top5 + threshold | +8.8 (+7.0 to +10.7) | +10.2 (+8.2 to +12.3) | +0.4 (-0.2 to +1.1) |
| dense · threshold | +9.0 (+7.2 to +11.0) | +10.4 (+8.3 to +12.6) | +0.4 (-0.2 to +1.1) |
| dense · top10 | +11.6 (+7.8 to +15.4) | -2.1 (-3.8 to -0.4) | +0.4 (-0.2 to +1.1) |
| adaptive-k · B=5 (published) | +11.6 (+7.9 to +15.3) | -2.1 (-3.5 to -0.7) | +0.4 (-0.2 to +1.1) |
| hybrid · top10 | +13.0 (+9.3 to +16.8) | -0.4 (-2.0 to +1.2) | +3.0 (+1.8 to +4.3) |
| dense · top5 | +13.5 (+9.9 to +17.1) | +0.1 (-1.4 to +1.5) | +0.4 (-0.3 to +1.1) |
| adaptive-k · B=1 (set on dev) | +13.7 (+10.2 to +17.5) | +0.4 (-0.9 to +1.7) | +0.4 (-0.2 to +1.1) |
| hybrid · top5 | +14.2 (+10.6 to +17.9) | +0.9 (-0.5 to +2.4) | +3.0 (+1.8 to +4.3) |
| bm25 · top10 | +15.3 (+11.8 to +19.0) | +2.2 (+0.7 to +3.7) | +8.0 (+6.1 to +9.9) |
| bm25 · top5 | +15.9 (+12.5 to +19.6) | +2.9 (+1.7 to +4.3) | +8.0 (+6.1 to +9.9) |

## FiQA (test): 648 answerable queries, 50 probes

| system · cutoff | task | F2 | recall | precision | wrong/q | shown/q | probes: nothing shown | answerable: nothing shown | nDCG@10 |
|---|---|---|---|---|---|---|---|---|---|
| **glossdex** | 39.7 | 35.4 | 45.3 | 37.3 | 6.30 | 7.29 | 96.0 | 0.6 | 46.8 |
| dense · top5 + threshold | 38.1 | 33.3 | 40.3 | 31.0 | 2.75 | 3.68 | 100.0 | 8.5 | 46.6 |
| dense · threshold | 35.9 | 31.0 | 45.0 | 28.6 | 8.10 | 9.24 | 100.0 | 11.1 | 46.6 |
| adaptive-k · B=1 (set on dev) | 31.1 | 33.5 | 39.2 | 29.8 | 2.52 | 3.40 | 0.0 | 0.0 | 46.6 |
| dense · top5 | 31.0 | 33.4 | 45.5 | 20.7 | 3.96 | 5.00 | 0.0 | 0.0 | 46.6 |
| adaptive-k · B=5 (published) | 29.7 | 32.0 | 50.1 | 16.4 | 6.23 | 7.40 | 0.0 | 0.0 | 46.6 |
| dense · top10 | 27.5 | 29.7 | 54.3 | 12.9 | 8.71 | 10.00 | 0.0 | 0.0 | 46.6 |
| hybrid · top5 | 25.8 | 27.7 | 37.9 | 17.2 | 4.14 | 5.00 | 0.0 | 0.0 | 38.2 |
| hybrid · top10 | 23.2 | 25.0 | 46.2 | 10.8 | 8.92 | 10.00 | 0.0 | 0.0 | 38.2 |
| bm25 · top5 | 16.3 | 17.5 | 24.3 | 10.5 | 4.48 | 5.00 | 0.0 | 0.0 | 24.7 |
| bm25 · top10 | 15.0 | 16.2 | 30.8 | 6.8 | 9.32 | 10.00 | 0.0 | 0.0 | 24.7 |

Paired differences, `glossdex` minus each (points, 95% CI):

| vs | task | F2 (answerable only) | nDCG@10 |
|---|---|---|---|
| dense · top5 + threshold | +1.7 (+0.0 to +3.3) | +2.1 (+0.4 to +3.8) | +0.2 (-0.6 to +1.0) |
| dense · threshold | +3.8 (+2.0 to +5.6) | +4.4 (+2.5 to +6.2) | +0.2 (-0.6 to +1.0) |
| adaptive-k · B=1 (set on dev) | +8.6 (+6.3 to +10.9) | +1.8 (+0.4 to +3.3) | +0.2 (-0.6 to +1.0) |
| dense · top5 | +8.7 (+6.4 to +11.1) | +2.0 (+0.4 to +3.7) | +0.2 (-0.6 to +1.0) |
| adaptive-k · B=5 (published) | +10.0 (+7.6 to +12.5) | +3.4 (+1.7 to +5.1) | +0.2 (-0.6 to +1.0) |
| dense · top10 | +12.2 (+9.7 to +14.7) | +5.7 (+3.8 to +7.7) | +0.2 (-0.5 to +1.0) |
| hybrid · top5 | +14.0 (+11.5 to +16.4) | +7.6 (+5.7 to +9.5) | +8.6 (+6.8 to +10.4) |
| hybrid · top10 | +16.5 (+14.0 to +19.0) | +10.4 (+8.4 to +12.4) | +8.6 (+6.8 to +10.4) |
| bm25 · top5 | +23.5 (+21.0 to +26.0) | +17.9 (+15.8 to +20.0) | +22.2 (+19.9 to +24.5) |
| bm25 · top10 | +24.7 (+22.2 to +27.1) | +19.2 (+17.1 to +21.3) | +22.2 (+19.8 to +24.6) |
