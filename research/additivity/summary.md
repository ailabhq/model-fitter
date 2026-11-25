## Planned footprints

| Candidate | Planned total GiB | On the GPU GiB |
|---|---|---|
| ngl58-c8704 | 10.82 | 9.20 |
| gpu-c8704 | 10.73 | 10.29 |
| gpu-c17408 | 11.31 | 10.86 |
| gpu-c26112 | 11.89 | 11.43 |

## All runs

| Rep | Load GiB | Candidate | Outcome | Gen tok/s @8K | Prefill tok/s | Peak wired GiB | Swap-outs GiB | Page-ins GiB | Critical share | Window max ms | Holder RSS first / min GiB | Desktop committed GiB | Headroom GiB | Why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0 | ngl58-c8704 | too slow | 3.69 | 60.41 | 11.51 | 0.78 | 48.54 | 0.0 | 8 | – | 5.06 | 0.12 |  |
| 1 | 0.0 | gpu-c8704 | Usable | 7.13 | 60.40 | 12.52 | 0.76 | 18.34 | 0.0 | 16 | – | 5.29 | -0.02 |  |
| 1 | 0.0 | gpu-c17408 | Usable | 7.12 | 58.69 | 12.98 | 1.51 | 12.79 | 0.009 | 60 | – | 4.70 | -0.01 |  |
| 1 | 0.0 | gpu-c26112 | Usable | 7.12 | 57.07 | 13.79 | 6.66 | 23.64 | 0.024 | 331 | – | 4.86 | -0.75 |  |
| 1 | 0.5 | ngl58-c8704 | killed | – | – | 11.51 | 0.0 | 19.36 | 0.0 | 931 | 0.51 / 0.0 | 5.80 | -0.62 | window check stuck for >= 500 ms at t=56s in phase gen@8k |
| 1 | 0.5 | gpu-c8704 | Usable | 7.07 | 58.34 | 13.01 | 2.05 | 25.59 | 0.0 | 464 | 0.51 / 0.0 | 6.40 | -1.13 |  |
| 1 | 0.5 | gpu-c17408 | Usable | 7.15 | 58.20 | 13.1 | 2.52 | 23.11 | 0.002 | 50 | 0.51 / 0.0 | 5.30 | -0.61 |  |
| 1 | 0.5 | gpu-c26112 | Usable | 7.12 | 58.36 | 13.58 | 5.56 | 27.76 | 0.006 | 24 | 0.51 / 0.0 | 5.71 | -1.60 |  |
| 1 | 1.0 | ngl58-c8704 | Usable | 5.24 | 59.02 | 11.13 | 1.39 | 33.09 | 0.0 | 22 | 1.01 / 0.0 | 5.30 | -0.12 |  |
| 1 | 1.0 | gpu-c8704 | Usable | 7.21 | 58.61 | 12.35 | 2.18 | 16.85 | 0.002 | 38 | 0.98 / 0.0 | 7.49 | -2.22 |  |
| 1 | 1.0 | gpu-c17408 | killed | – | – | 12.92 | 1.19 | 21.55 | 0.0 | 19 | 1.01 / 0.0 | 6.15 | -1.46 | timer check took 1397 ms at t=157s in phase gen@8k |
| 1 | 1.0 | gpu-c26112 | Usable | 7.20 | 58.78 | 13.49 | 3.2 | 23.41 | 0.013 | 47 | 1.01 / 0.0 | 6.14 | -2.03 |  |
| 1 | 1.5 | ngl58-c8704 | Usable | 5.24 | 59.41 | 11.14 | 1.95 | 31.5 | 0.0 | 38 | 1.51 / 0.0 | 5.57 | -0.39 |  |
| 1 | 1.5 | gpu-c8704 | Usable | 7.21 | 58.48 | 12.71 | 2.35 | 23.55 | 0.0 | 238 | 1.51 / 0.0 | 5.61 | -0.34 |  |
| 1 | 1.5 | gpu-c17408 | Usable | 7.21 | 58.71 | 12.82 | 2.46 | 22.87 | 0.004 | 43 | 1.51 / 0.0 | 6.36 | -1.67 |  |
| 1 | 1.5 | gpu-c26112 | Usable | 7.22 | 58.42 | 13.45 | 3.51 | 23.87 | 0.006 | 39 | 1.51 / 0.0 | 5.49 | -1.38 |  |
| 1 | 2.0 | ngl58-c8704 | killed | – | – | 11.57 | 2.96 | 80.26 | 0.012 | 42 | 2.01 / 0.0 | 9.34 | -4.16 | timer check took 677 ms at t=255s in phase gen@8k |
| 1 | 2.0 | gpu-c8704 | killed | – | – | 11.41 | 0.74 | 21.19 | 0.073 | 826 | 2.01 / 0.0 | 7.44 | -2.17 | window check stuck for >= 500 ms at t=17s in phase load |
| 1 | 2.0 | gpu-c17408 | killed | – | – | 11.17 | 1.05 | 20.77 | 0.152 | 1568 | 2.01 / 0.0 | 11.53 | -6.84 | spawn check stuck for >= 500 ms at t=18s in phase load |
| 1 | 2.0 | gpu-c26112 | killed | – | – | 11.36 | 2.96 | 21.57 | 0.287 | 5541 | 2.01 / 0.0 | 9.53 | -5.42 | window check stuck for >= 500 ms at t=21s in phase load |
| 1 | 4.0 | ngl58-c8704 | killed | – | – | 10.61 | 1.12 | 19.46 | 0.186 | 744 | 4.01 / 0.0 | 9.56 | -4.38 | window check stuck for >= 500 ms at t=18s in phase load |
| 1 | 4.0 | gpu-c8704 | killed | – | – | 9.84 | 2.4 | 21.8 | 0.301 | 6222 | 4.01 / 0.0 | 10.57 | -5.30 | timer check took 701 ms at t=19s in phase load |
| 1 | 4.0 | gpu-c17408 | killed | – | – | 9.8 | 2.22 | 21.76 | 0.308 | 6473 | 4.01 / 0.0 | 9.36 | -4.67 | timer check took 523 ms at t=21s in phase load |
| 1 | 4.0 | gpu-c26112 | killed | – | – | 10.3 | 1.99 | 21.31 | 0.247 | 4963 | 4.01 / 0.0 | 9.36 | -5.25 | window check stuck for >= 500 ms at t=17s in phase load |
| 2 | 0.0 | ngl58-c8704 | killed | – | – | 11.45 | 0.02 | 26.75 | 0.0 | 13 | – | 7.53 | -2.35 | timer check took 758 ms at t=59s in phase gen@8k |
| 2 | 0.0 | gpu-c8704 | Usable | 7.16 | 58.31 | 12.38 | 0.6 | 11.15 | 0.0 | 10 | – | 4.86 | 0.41 |  |
| 2 | 0.0 | gpu-c17408 | Usable | 7.17 | 58.27 | 12.93 | 1.77 | 21.02 | 0.0 | 36 | – | 6.06 | -1.37 |  |
| 2 | 0.0 | gpu-c26112 | killed | 7.12 | 58.71 | 13.61 | 2.38 | 10.67 | 0.0 | 1194 | – | 5.94 | -1.83 | window check stuck for >= 500 ms at t=202s in phase gen@8k |
| 2 | 0.5 | ngl58-c8704 | too slow | 3.34 | 59.57 | 11.47 | 1.36 | 78.43 | 0.0 | 19 | 0.51 / 0.0 | 5.68 | -0.50 |  |
| 2 | 0.5 | gpu-c8704 | Usable | 7.21 | 58.86 | 12.61 | 1.3 | 14.12 | 0.0 | 29 | 0.51 / 0.0 | 4.95 | 0.32 |  |
| 2 | 0.5 | gpu-c17408 | Usable | 7.21 | 58.63 | 13.06 | 2.21 | 15.64 | 0.002 | 43 | 0.51 / 0.0 | 5.31 | -0.62 |  |
| 2 | 0.5 | gpu-c26112 | Usable | 7.19 | 58.56 | 13.77 | 5.49 | 18.9 | 0.004 | 51 | 0.51 / 0.0 | 6.76 | -2.65 |  |
| 2 | 1.0 | ngl58-c8704 | Usable | 5.02 | 58.50 | 11.53 | 1.85 | 37.18 | 0.0 | 11 | 1.01 / 0.0 | 5.83 | -0.65 |  |
| 2 | 1.0 | gpu-c8704 | Usable | 7.18 | 59.09 | 12.52 | 1.94 | 21.72 | 0.0 | 43 | 1.01 / 0.0 | 6.06 | -0.79 |  |
| 2 | 1.0 | gpu-c17408 | Usable | 7.21 | 58.77 | 13.08 | 2.26 | 22.9 | 0.002 | 12 | 1.01 / 0.0 | 5.95 | -1.26 |  |
| 2 | 1.0 | gpu-c26112 | Usable | 7.15 | 58.72 | 13.64 | 3.67 | 23.23 | 0.004 | 33 | 1.01 / 0.0 | 5.56 | -1.45 |  |
| 2 | 1.5 | ngl58-c8704 | too slow | 4.65 | 58.71 | 11.49 | 2.9 | 46.84 | 0.007 | 76 | 1.51 / 0.0 | 7.33 | -2.15 |  |
| 2 | 1.5 | gpu-c8704 | Usable | 7.21 | 58.65 | 12.59 | 2.53 | 22.91 | 0.002 | 8 | 1.51 / 0.0 | 6.20 | -0.93 |  |
| 2 | 1.5 | gpu-c17408 | Usable | 7.17 | 59.00 | 13.2 | 3.14 | 23.13 | 0.006 | 29 | 1.51 / 0.0 | 6.41 | -1.72 |  |
| 2 | 1.5 | gpu-c26112 | killed | – | – | 13.56 | 2.01 | 20.62 | 0.207 | 4141 | 1.51 / 0.0 | 7.77 | -3.66 | window check stuck for >= 500 ms at t=17s in phase load |
| 2 | 2.0 | ngl58-c8704 | too slow | 3.76 | 59.20 | 11.57 | 3.38 | 67.21 | 0.006 | 18 | 2.01 / 0.0 | 7.03 | -1.85 |  |
| 2 | 2.0 | gpu-c8704 | killed | – | – | 10.6 | 1.26 | 21.0 | 0.172 | 1076 | 2.01 / 0.0 | 12.44 | -7.17 | spawn check stuck for >= 500 ms at t=19s in phase load |
| 2 | 2.0 | gpu-c17408 | killed | – | – | 11.86 | 0.99 | 20.03 | 0.059 | 1015 | 2.01 / 0.0 | 7.17 | -2.48 | window check stuck for >= 500 ms at t=17s in phase load |
| 2 | 2.0 | gpu-c26112 | killed | – | – | 11.36 | 1.83 | 20.93 | 0.184 | 2684 | 2.01 / 0.0 | 6.82 | -2.71 | window check stuck for >= 500 ms at t=19s in phase load |
| 2 | 4.0 | ngl58-c8704 | killed | – | – | 10.37 | 0.86 | 19.35 | 0.122 | 481 | 4.01 / 0.0 | 9.01 | -3.83 | spawn check stuck for >= 500 ms at t=17s in phase load |
| 2 | 4.0 | gpu-c8704 | killed | – | – | 10.26 | 1.92 | 21.09 | 0.264 | 1542 | 4.01 / 0.0 | 9.08 | -3.81 | timer check took 582 ms at t=21s in phase load |
| 2 | 4.0 | gpu-c17408 | killed | – | – | 10.52 | 1.95 | 21.14 | 0.258 | 4779 | 4.01 / 0.0 | 9.20 | -4.51 | timer check took 916 ms at t=19s in phase load |
| 2 | 4.0 | gpu-c26112 | killed | – | – | 10.0 | 2.04 | 21.21 | 0.26 | 2392 | 4.01 / 0.0 | 9.43 | -5.32 | window check stuck for >= 500 ms at t=21s in phase load |

## Outcome by load (is failure monotone in the planned footprint?)

| Rep | Load GiB | ngl58-c8704 | gpu-c8704 | gpu-c17408 | gpu-c26112 | Failures upward-closed |
|---|---|---|---|---|---|---|
| 1 | 0.0 | too slow | Usable | Usable | Usable | yes |
| 1 | 0.5 | killed | Usable | Usable | Usable | yes |
| 1 | 1.0 | Usable | Usable | killed | Usable | **no** |
| 1 | 1.5 | Usable | Usable | Usable | Usable | yes |
| 1 | 2.0 | killed | killed | killed | killed | yes |
| 1 | 4.0 | killed | killed | killed | killed | yes |
| 2 | 0.0 | killed | Usable | Usable | killed | yes |
| 2 | 0.5 | too slow | Usable | Usable | Usable | yes |
| 2 | 1.0 | Usable | Usable | Usable | Usable | yes |
| 2 | 1.5 | too slow | Usable | Usable | killed | yes |
| 2 | 2.0 | too slow | killed | killed | killed | yes |
| 2 | 4.0 | killed | killed | killed | killed | yes |

## Outcome against headroom (RAM − desktop committed − planned total), all runs

| Headroom GiB | Candidate | Rep | Load GiB | Outcome | Window max ms | Critical share |
|---|---|---|---|---|---|---|
| -7.17 | gpu-c8704 | 2 | 2 | killed | 1076 | 0.172 |
| -6.84 | gpu-c17408 | 1 | 2 | killed | 1568 | 0.152 |
| -5.42 | gpu-c26112 | 1 | 2 | killed | 5541 | 0.287 |
| -5.32 | gpu-c26112 | 2 | 4 | killed | 2392 | 0.26 |
| -5.30 | gpu-c8704 | 1 | 4 | killed | 6222 | 0.301 |
| -5.25 | gpu-c26112 | 1 | 4 | killed | 4963 | 0.247 |
| -4.67 | gpu-c17408 | 1 | 4 | killed | 6473 | 0.308 |
| -4.51 | gpu-c17408 | 2 | 4 | killed | 4779 | 0.258 |
| -4.38 | ngl58-c8704 | 1 | 4 | killed | 744 | 0.186 |
| -4.16 | ngl58-c8704 | 1 | 2 | killed | 42 | 0.012 |
| -3.83 | ngl58-c8704 | 2 | 4 | killed | 481 | 0.122 |
| -3.81 | gpu-c8704 | 2 | 4 | killed | 1542 | 0.264 |
| -3.66 | gpu-c26112 | 2 | 1.5 | killed | 4141 | 0.207 |
| -2.71 | gpu-c26112 | 2 | 2 | killed | 2684 | 0.184 |
| -2.65 | gpu-c26112 | 2 | 0.5 | Usable | 51 | 0.004 |
| -2.48 | gpu-c17408 | 2 | 2 | killed | 1015 | 0.059 |
| -2.35 | ngl58-c8704 | 2 | 0 | killed | 13 | 0.0 |
| -2.22 | gpu-c8704 | 1 | 1 | Usable | 38 | 0.002 |
| -2.17 | gpu-c8704 | 1 | 2 | killed | 826 | 0.073 |
| -2.15 | ngl58-c8704 | 2 | 1.5 | too slow | 76 | 0.007 |
| -2.03 | gpu-c26112 | 1 | 1 | Usable | 47 | 0.013 |
| -1.85 | ngl58-c8704 | 2 | 2 | too slow | 18 | 0.006 |
| -1.83 | gpu-c26112 | 2 | 0 | killed | 1194 | 0.0 |
| -1.72 | gpu-c17408 | 2 | 1.5 | Usable | 29 | 0.006 |
| -1.67 | gpu-c17408 | 1 | 1.5 | Usable | 43 | 0.004 |
| -1.60 | gpu-c26112 | 1 | 0.5 | Usable | 24 | 0.006 |
| -1.46 | gpu-c17408 | 1 | 1 | killed | 19 | 0.0 |
| -1.45 | gpu-c26112 | 2 | 1 | Usable | 33 | 0.004 |
| -1.38 | gpu-c26112 | 1 | 1.5 | Usable | 39 | 0.006 |
| -1.37 | gpu-c17408 | 2 | 0 | Usable | 36 | 0.0 |
| -1.26 | gpu-c17408 | 2 | 1 | Usable | 12 | 0.002 |
| -1.13 | gpu-c8704 | 1 | 0.5 | Usable | 464 | 0.0 |
| -0.93 | gpu-c8704 | 2 | 1.5 | Usable | 8 | 0.002 |
| -0.79 | gpu-c8704 | 2 | 1 | Usable | 43 | 0.0 |
| -0.75 | gpu-c26112 | 1 | 0 | Usable | 331 | 0.024 |
| -0.65 | ngl58-c8704 | 2 | 1 | Usable | 11 | 0.0 |
| -0.62 | ngl58-c8704 | 1 | 0.5 | killed | 931 | 0.0 |
| -0.62 | gpu-c17408 | 2 | 0.5 | Usable | 43 | 0.002 |
| -0.61 | gpu-c17408 | 1 | 0.5 | Usable | 50 | 0.002 |
| -0.50 | ngl58-c8704 | 2 | 0.5 | too slow | 19 | 0.0 |
| -0.39 | ngl58-c8704 | 1 | 1.5 | Usable | 38 | 0.0 |
| -0.34 | gpu-c8704 | 1 | 1.5 | Usable | 238 | 0.0 |
| -0.12 | ngl58-c8704 | 1 | 1 | Usable | 22 | 0.0 |
| -0.02 | gpu-c8704 | 1 | 0 | Usable | 16 | 0.0 |
| -0.01 | gpu-c17408 | 1 | 0 | Usable | 60 | 0.009 |
| 0.12 | ngl58-c8704 | 1 | 0 | too slow | 8 | 0.0 |
| 0.32 | gpu-c8704 | 2 | 0.5 | Usable | 29 | 0.0 |
| 0.41 | gpu-c8704 | 2 | 0 | Usable | 10 | 0.0 |

## Interaction ratio rho = Var(obs_j - obs_i) / Var(obs_j) over (rep, load), completed runs only

| Metric | i → j | n | Var gap | Var obs_j | rho | MAE: obs_i + footprint Δ | MAE: obs_i + quiet gap | MAE: j's quiet run |
|---|---|---|---|---|---|---|---|---|
| Peak wired GiB | ngl58-c8704 → gpu-c8704 | 6 | 0.0377 | 0.0121 | 3.12 | 0.15 | 0.20 | 0.10 |
| Peak wired GiB | ngl58-c8704 → gpu-c17408 | 5 | 0.0076 | 0.0158 | 0.48 | 0.06 | 0.16 | 0.14 |
| Peak wired GiB | ngl58-c8704 → gpu-c26112 | 5 | 0.0073 | 0.0195 | 0.37 | 0.10 | 0.07 | 0.20 |
| Peak wired GiB | gpu-c8704 → ngl58-c8704 | 6 | 0.0377 | 0.0299 | 1.26 | 0.15 | 0.20 | 0.17 |
| Peak wired GiB | gpu-c8704 → gpu-c17408 | 7 | 0.0398 | 0.0134 | 2.97 | 0.22 | 0.21 | 0.15 |
| Peak wired GiB | gpu-c8704 → gpu-c26112 | 6 | 0.0642 | 0.0165 | 3.88 | 0.20 | 0.32 | 0.20 |
| Peak wired GiB | gpu-c17408 → ngl58-c8704 | 5 | 0.0076 | 0.0211 | 0.36 | 0.06 | 0.16 | 0.11 |
| Peak wired GiB | gpu-c17408 → gpu-c8704 | 7 | 0.0398 | 0.0341 | 1.17 | 0.22 | 0.21 | 0.24 |
| Peak wired GiB | gpu-c17408 → gpu-c26112 | 5 | 0.0132 | 0.0158 | 0.83 | 0.07 | 0.21 | 0.18 |
| Peak wired GiB | gpu-c26112 → ngl58-c8704 | 5 | 0.0073 | 0.0329 | 0.22 | 0.10 | 0.07 | 0.20 |
| Peak wired GiB | gpu-c26112 → gpu-c8704 | 6 | 0.0642 | 0.0422 | 1.52 | 0.20 | 0.32 | 0.19 |
| Peak wired GiB | gpu-c26112 → gpu-c17408 | 5 | 0.0132 | 0.0105 | 1.26 | 0.07 | 0.21 | 0.11 |
| Swap-outs GiB | ngl58-c8704 → gpu-c8704 | 6 | 0.1364 | 0.3866 | 0.35 | 0.37 | 0.35 | 1.30 |
| Swap-outs GiB | ngl58-c8704 → gpu-c17408 | 5 | 0.0479 | 0.2727 | 0.18 | 0.18 | 0.29 | 1.01 |
| Swap-outs GiB | ngl58-c8704 → gpu-c26112 | 5 | 2.8891 | 1.8009 | 1.60 | 1.26 | 3.55 | 2.69 |
| Swap-outs GiB | gpu-c8704 → ngl58-c8704 | 6 | 0.1364 | 0.4305 | 0.32 | 0.37 | 0.35 | 1.11 |
| Swap-outs GiB | gpu-c8704 → gpu-c17408 | 7 | 0.1109 | 0.2410 | 0.46 | 0.24 | 0.48 | 0.88 |
| Swap-outs GiB | gpu-c8704 → gpu-c26112 | 6 | 3.1608 | 1.6550 | 1.91 | 1.22 | 3.58 | 2.37 |
| Swap-outs GiB | gpu-c17408 → ngl58-c8704 | 5 | 0.0479 | 0.4928 | 0.10 | 0.18 | 0.29 | 1.24 |
| Swap-outs GiB | gpu-c17408 → gpu-c8704 | 7 | 0.1109 | 0.5036 | 0.22 | 0.24 | 0.48 | 1.35 |
| Swap-outs GiB | gpu-c17408 → gpu-c26112 | 5 | 2.1608 | 1.4592 | 1.48 | 1.61 | 2.96 | 2.10 |
| Swap-outs GiB | gpu-c26112 → ngl58-c8704 | 5 | 2.8891 | 0.1739 | 16.62 | 1.26 | 3.55 | 0.86 |
| Swap-outs GiB | gpu-c26112 → gpu-c8704 | 6 | 3.1608 | 0.3088 | 10.24 | 1.22 | 3.58 | 1.20 |
| Swap-outs GiB | gpu-c26112 → gpu-c17408 | 5 | 2.1608 | 0.1299 | 16.64 | 1.61 | 2.96 | 0.85 |
| Critical share | ngl58-c8704 → gpu-c8704 | 6 | 0.0000 | 0.0000 | 5.16 | – | 0.00 | 0.00 |
| Critical share | ngl58-c8704 → gpu-c17408 | 5 | 0.0000 | 0.0000 | 1.56 | – | 0.01 | 0.01 |
| Critical share | ngl58-c8704 → gpu-c26112 | 5 | 0.0001 | 0.0001 | 1.00 | – | 0.02 | 0.02 |
| Critical share | gpu-c8704 → ngl58-c8704 | 6 | 0.0000 | 0.0000 | 0.67 | – | 0.00 | 0.00 |
| Critical share | gpu-c8704 → gpu-c17408 | 7 | 0.0000 | 0.0000 | 0.89 | – | 0.00 | 0.00 |
| Critical share | gpu-c8704 → gpu-c26112 | 6 | 0.0000 | 0.0001 | 0.97 | – | 0.02 | 0.02 |
| Critical share | gpu-c17408 → ngl58-c8704 | 5 | 0.0000 | 0.0000 | 1.40 | – | 0.01 | 0.00 |
| Critical share | gpu-c17408 → gpu-c8704 | 7 | 0.0000 | 0.0000 | 14.42 | – | 0.00 | 0.00 |
| Critical share | gpu-c17408 → gpu-c26112 | 5 | 0.0000 | 0.0001 | 0.44 | – | 0.01 | 0.02 |
| Critical share | gpu-c26112 → ngl58-c8704 | 5 | 0.0001 | 0.0000 | nan | – | 0.02 | 0.00 |
| Critical share | gpu-c26112 → gpu-c8704 | 6 | 0.0000 | 0.0000 | 89.05 | – | 0.02 | 0.00 |
| Critical share | gpu-c26112 → gpu-c17408 | 5 | 0.0000 | 0.0000 | 3.48 | – | 0.01 | 0.01 |
| Gen tok/s @8K | ngl58-c8704 → gpu-c8704 | 6 | 0.5474 | 0.0009 | 621.71 | – | 1.11 | 0.07 |
| Gen tok/s @8K | ngl58-c8704 → gpu-c17408 | 5 | 0.5362 | 0.0013 | 424.23 | – | 1.01 | 0.08 |
| Gen tok/s @8K | ngl58-c8704 → gpu-c26112 | 5 | 0.6484 | 0.0013 | 497.21 | – | 1.16 | 0.07 |
| Gen tok/s @8K | gpu-c8704 → ngl58-c8704 | 6 | 0.5474 | 0.5641 | 0.97 | – | 1.11 | 1.15 |
| Gen tok/s @8K | gpu-c8704 → gpu-c17408 | 7 | 0.0012 | 0.0010 | 1.14 | – | 0.03 | 0.05 |
| Gen tok/s @8K | gpu-c8704 → gpu-c26112 | 6 | 0.0007 | 0.0015 | 0.45 | – | 0.02 | 0.06 |
| Gen tok/s @8K | gpu-c17408 → ngl58-c8704 | 5 | 0.5362 | 0.5559 | 0.96 | – | 1.01 | 1.05 |
| Gen tok/s @8K | gpu-c17408 → gpu-c8704 | 7 | 0.0012 | 0.0024 | 0.51 | – | 0.03 | 0.06 |
| Gen tok/s @8K | gpu-c17408 → gpu-c26112 | 5 | 0.0006 | 0.0016 | 0.38 | – | 0.03 | 0.05 |
| Gen tok/s @8K | gpu-c26112 → ngl58-c8704 | 5 | 0.6484 | 0.6734 | 0.96 | – | 1.16 | 1.20 |
| Gen tok/s @8K | gpu-c26112 → gpu-c8704 | 6 | 0.0007 | 0.0027 | 0.25 | – | 0.02 | 0.07 |
| Gen tok/s @8K | gpu-c26112 → gpu-c17408 | 5 | 0.0006 | 0.0014 | 0.42 | – | 0.03 | 0.07 |

## Replicate noise (same load, same candidate: |rep1 - rep2|)

| Metric | mean | max |
|---|---|---|
| Peak wired GiB | 0.255 | 0.810 |
| Swap-outs GiB | 0.695 | 4.280 |
| Critical share | 0.030 | 0.201 |
| Gen tok/s @8K | 0.107 | 0.590 |
