# EPLB +32, 1000-iteration retest

| arm/source | size | n | TTFT median | total median | Δ TTFT vs original |
|---|---:|---:|---:|---:|---:|
| original (prior bf94982) | 8K | 10 | 2.802466 s | 3.323588 s | 0.00% |
| original (prior bf94982) | 16K | 10 | 5.714676 s | 6.240125 s | 0.00% |
| original (prior bf94982) | 32K | 10 | 11.787459 s | 12.316519 s | 0.00% |
| eplb (this retest) | 8K | 10 | 2.596853 s | 3.166467 s | -7.34% |
| eplb (this retest) | 16K | 10 | 5.307194 s | 5.881423 s | -7.09% |
| eplb (this retest) | 32K | 10 | 11.218345 s | 11.781440 s | -4.85% |
| k2 (prior bf94982) | 8K | 10 | 2.935765 s | 3.477432 s | 0.00% |
| k2 (prior bf94982) | 16K | 10 | 5.978486 s | 6.497243 s | 0.00% |
| k2 (prior bf94982) | 32K | 10 | 12.301329 s | 12.858562 s | 0.00% |

EPLB rows are this retest; original and K=2 rows are reused from bf94982 and are not rerun.
