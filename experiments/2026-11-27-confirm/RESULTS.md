# Confirmatory run of 2026-11-27 (234 processes, sealed at 5968229)

Sealed file: `arms.json`. Band 1.03, three pairs; ratio = baseline time / candidate time.

- not confirmed (28/30): G-pull/candidate: rows pulled from peer memory against All-to-All, same tuned staged consumer; expected faster: PULLW_SLT
- not confirmed (27/30): G-pull-blocks/candidate: the same, pulled as one block per source in rank-rotated order; expected faster: PULLR_SLT
- confirmed (17/17): G-inplace/candidate: one-launch consumer reading peer memory in place against the tuned staged path; expected faster: D_MRG16_Q8_G2_K64_W
- confirmed (12/12): G-inplace/baseline: one-launch consumer reading peer memory in place against the tuned staged path; expected faster: SLT
- not confirmed (7/8): G-inplace-or-pull/candidate: reading in place against pulling first, both from peer memory; expected faster: D_MRG16_Q8_G2_K64_W
- not confirmed (18/19): G-inplace-or-pull/baseline: reading in place against pulling first, both from peer memory; expected faster: PULLW_SLT
- confirmed (6/6): S-pull/candidate: rows pulled from peer memory, then a library GEMM per expert, against the native path; expected faster: PULL_BLAS_R
- confirmed (6/6): S-pull-or-stage/candidate: the same library GEMM on rows pulled from peer memory against rows staged by All-to-All; expected faster: PULL_BLAS_R
- confirmed (3/3): S-inplace-small/candidate: one-launch consumer in place, small tiles, against the native path; expected faster: D_GATHER16_Q8_G2_K64_W_R
- confirmed (3/3): S-inplace-small/baseline: one-launch consumer in place, small tiles, against the native path; expected faster: NATIVE
- not confirmed (4/6): S-inplace-large/candidate: one-launch consumer in place, large tiles, against the native path; expected faster: D_GATHER64_Q1_G4_K64_W_R

## PULLW_SLT against SLT

| unit | SLT ms | PULLW_SLT ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_M0001 | 0.207 | 0.145 | 1.50 | 1.41–1.65 | candidate | candidate ✓ |
| GLM_M0002 | 0.215 | 0.155 | 1.39 | 1.39–1.45 | candidate | candidate ✓ |
| GLM_M0004 | 0.194 | 0.134 | 1.43 | 1.39–1.48 | candidate | candidate ✓ |
| GLM_M0008 | 0.201 | 0.134 | 1.50 | 1.45–1.63 | candidate | candidate ✓ |
| GLM_M0016 | 0.212 | 0.135 | 1.47 | 1.37–1.56 | candidate | candidate ✓ |
| GLM_M0032 | 0.198 | 0.139 | 1.41 | 1.36–1.42 | candidate | candidate ✓ |
| GLM_M0064 | 0.211 | 0.148 | 1.37 | 1.34–1.44 | candidate | candidate ✓ |
| GLM_M0128 | 0.243 | 0.178 | 1.36 | 1.32–1.39 | candidate | candidate ✓ |
| GLM_M0256 | 0.311 | 0.255 | 1.22 | 1.21–1.23 | candidate | candidate ✓ |
| GLM_M0512 | 0.489 | 0.443 | 1.11 | 1.10–1.11 | candidate | candidate ✓ |
| GLM_M1024 | 0.750 | 0.746 | 0.98 | 0.97–1.01 | undecided |  |
| QWEN35_M0001 | 0.208 | 0.175 | 1.18 | 1.02–1.58 | undecided | candidate ✗ |
| QWEN35_M0002 | 0.203 | 0.145 | 1.38 | 1.37–1.41 | candidate | candidate ✓ |
| QWEN35_M0004 | 0.195 | 0.135 | 1.44 | 1.43–1.46 | candidate | candidate ✓ |
| QWEN35_M0008 | 0.216 | 0.135 | 1.55 | 1.40–1.67 | candidate | candidate ✓ |
| QWEN35_M0016 | 0.197 | 0.136 | 1.45 | 1.42–1.50 | candidate | candidate ✓ |
| QWEN35_M0032 | 0.193 | 0.134 | 1.44 | 1.40–1.44 | candidate | candidate ✓ |
| QWEN35_M0064 | 0.195 | 0.137 | 1.42 | 1.36–1.45 | candidate | candidate ✓ |
| QWEN35_M0128 | 0.219 | 0.155 | 1.41 | 1.35–1.41 | candidate | candidate ✓ |
| QWEN35_M0256 | 0.252 | 0.189 | 1.33 | 1.30–1.40 | candidate | candidate ✓ |
| QWEN35_M0512 | 0.321 | 0.272 | 1.19 | 1.16–1.19 | candidate | candidate ✓ |
| QWEN35_M1024 | 0.469 | 0.486 | 0.97 | 0.96–0.98 | undecided |  |
| DSV2L_M0001 | 0.205 | 0.143 | 1.42 | 1.42–1.48 | candidate | candidate ✓ |
| DSV2L_M0002 | 0.212 | 0.143 | 1.47 | 1.38–1.50 | candidate | candidate ✓ |
| DSV2L_M0004 | 0.189 | 0.132 | 1.40 | 1.40–1.46 | candidate | candidate ✓ |
| DSV2L_M0008 | 0.193 | 0.131 | 1.46 | 1.35–1.50 | candidate | candidate ✓ |
| DSV2L_M0016 | 0.194 | 0.144 | 1.35 | 1.03–1.43 | undecided | candidate ✗ |
| DSV2L_M0032 | 0.197 | 0.137 | 1.43 | 1.13–1.44 | candidate | candidate ✓ |
| DSV2L_M0064 | 0.208 | 0.182 | 1.16 | 1.14–1.45 | candidate | candidate ✓ |
| DSV2L_M0128 | 0.244 | 0.178 | 1.35 | 1.31–1.37 | candidate | candidate ✓ |
| DSV2L_M0256 | 0.319 | 0.261 | 1.22 | 1.21–1.26 | candidate | candidate ✓ |
| DSV2L_M0512 | 0.470 | 0.419 | 1.11 | 1.11–1.13 | candidate | candidate ✓ |
| DSV2L_M1024 | 0.726 | 0.722 | 1.00 | 0.98–1.02 | tie |  |

## PULLR_SLT against SLT

| unit | SLT ms | PULLR_SLT ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_M0001 | 0.207 | 0.153 | 1.45 | 1.35–1.55 | candidate | candidate ✓ |
| GLM_M0002 | 0.215 | 0.154 | 1.40 | 1.35–1.40 | candidate | candidate ✓ |
| GLM_M0004 | 0.194 | 0.141 | 1.37 | 1.17–1.45 | candidate | candidate ✓ |
| GLM_M0008 | 0.201 | 0.134 | 1.47 | 1.44–1.63 | candidate | candidate ✓ |
| GLM_M0016 | 0.212 | 0.139 | 1.42 | 1.37–1.53 | candidate | candidate ✓ |
| GLM_M0032 | 0.198 | 0.147 | 1.32 | 1.02–1.40 | undecided | candidate ✗ |
| GLM_M0064 | 0.211 | 0.158 | 1.30 | 1.07–1.34 | candidate | candidate ✓ |
| GLM_M0128 | 0.243 | 0.194 | 1.26 | 1.24–1.27 | candidate | candidate ✓ |
| GLM_M0256 | 0.311 | 0.283 | 1.11 | 1.09–1.12 | candidate | candidate ✓ |
| GLM_M0512 | 0.489 | 0.439 | 1.11 | 1.08–1.16 | candidate |  |
| GLM_M1024 | 0.750 | 0.689 | 1.10 | 1.04–1.10 | candidate |  |
| QWEN35_M0001 | 0.208 | 0.182 | 1.14 | 0.96–1.56 | undecided | candidate ✗ |
| QWEN35_M0002 | 0.203 | 0.145 | 1.39 | 1.36–1.40 | candidate | candidate ✓ |
| QWEN35_M0004 | 0.195 | 0.139 | 1.42 | 1.40–1.42 | candidate | candidate ✓ |
| QWEN35_M0008 | 0.216 | 0.138 | 1.52 | 1.36–1.59 | candidate | candidate ✓ |
| QWEN35_M0016 | 0.197 | 0.141 | 1.40 | 1.24–1.45 | candidate | candidate ✓ |
| QWEN35_M0032 | 0.193 | 0.158 | 1.36 | 1.20–1.44 | candidate | candidate ✓ |
| QWEN35_M0064 | 0.195 | 0.144 | 1.35 | 1.33–1.36 | candidate | candidate ✓ |
| QWEN35_M0128 | 0.219 | 0.170 | 1.29 | 1.27–1.39 | candidate | candidate ✓ |
| QWEN35_M0256 | 0.252 | 0.217 | 1.16 | 1.13–1.24 | candidate | candidate ✓ |
| QWEN35_M0512 | 0.321 | 0.292 | 1.10 | 1.08–1.12 | candidate | candidate ✓ |
| QWEN35_M1024 | 0.469 | 0.428 | 1.11 | 1.09–1.11 | candidate | candidate ✓ |
| DSV2L_M0001 | 0.205 | 0.147 | 1.39 | 1.39–1.40 | candidate | candidate ✓ |
| DSV2L_M0002 | 0.212 | 0.145 | 1.41 | 1.36–1.50 | candidate | candidate ✓ |
| DSV2L_M0004 | 0.189 | 0.133 | 1.39 | 1.20–1.48 | candidate | candidate ✓ |
| DSV2L_M0008 | 0.193 | 0.179 | 1.08 | 0.99–1.45 | undecided | candidate ✗ |
| DSV2L_M0016 | 0.194 | 0.149 | 1.30 | 1.14–1.41 | candidate | candidate ✓ |
| DSV2L_M0032 | 0.197 | 0.139 | 1.40 | 1.32–1.42 | candidate | candidate ✓ |
| DSV2L_M0064 | 0.208 | 0.172 | 1.21 | 1.11–1.37 | candidate | candidate ✓ |
| DSV2L_M0128 | 0.244 | 0.197 | 1.24 | 1.24–1.26 | candidate | candidate ✓ |
| DSV2L_M0256 | 0.319 | 0.288 | 1.11 | 1.11–1.15 | candidate | candidate ✓ |
| DSV2L_M0512 | 0.470 | 0.406 | 1.10 | 1.10–1.16 | candidate | candidate ✓ |
| DSV2L_M1024 | 0.726 | 0.662 | 1.10 | 1.10–1.11 | candidate |  |

## D_MRG16_Q8_G2_K64_W against SLT

| unit | SLT ms | D_MRG16_Q8_G2_K64_W ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_M0001 | 0.207 | 0.112 | 1.88 | 1.84–2.05 | candidate | candidate ✓ |
| GLM_M0002 | 0.215 | 0.115 | 1.84 | 1.83–1.89 | candidate | candidate ✓ |
| GLM_M0004 | 0.194 | 0.124 | 1.56 | 1.36–1.62 | candidate | candidate ✓ |
| GLM_M0008 | 0.201 | 0.137 | 1.47 | 1.43–1.57 | candidate | candidate ✓ |
| GLM_M0016 | 0.212 | 0.161 | 1.24 | 1.23–1.33 | candidate | candidate ✓ |
| GLM_M0032 | 0.198 | 0.244 | 0.79 | 0.78–0.86 | baseline |  |
| GLM_M0064 | 0.211 | 0.297 | 0.71 | 0.67–0.74 | baseline |  |
| GLM_M0128 | 0.243 | 0.453 | 0.53 | 0.51–0.54 | baseline | baseline ✓ |
| GLM_M0256 | 0.311 | 0.736 | 0.42 | 0.42–0.43 | baseline | baseline ✓ |
| GLM_M0512 | 0.489 | 1.293 | 0.38 | 0.37–0.39 | baseline | baseline ✓ |
| GLM_M1024 | 0.750 | 2.507 | 0.30 | 0.29–0.30 | baseline | baseline ✓ |
| QWEN35_M0001 | 0.208 | 0.140 | 1.48 | 1.45–1.98 | candidate | candidate ✓ |
| QWEN35_M0002 | 0.203 | 0.111 | 1.84 | 1.77–1.84 | candidate | candidate ✓ |
| QWEN35_M0004 | 0.195 | 0.112 | 1.76 | 1.73–1.77 | candidate | candidate ✓ |
| QWEN35_M0008 | 0.216 | 0.120 | 1.75 | 1.42–1.86 | candidate | candidate ✓ |
| QWEN35_M0016 | 0.197 | 0.128 | 1.54 | 1.52–1.54 | candidate | candidate ✓ |
| QWEN35_M0032 | 0.193 | 0.157 | 1.38 | 1.14–1.40 | candidate | candidate ✓ |
| QWEN35_M0064 | 0.195 | 0.171 | 1.14 | 1.13–1.24 | candidate | candidate ✓ |
| QWEN35_M0128 | 0.219 | 0.233 | 0.94 | 0.93–1.03 | undecided |  |
| QWEN35_M0256 | 0.252 | 0.342 | 0.73 | 0.73–0.76 | baseline | baseline ✓ |
| QWEN35_M0512 | 0.321 | 0.557 | 0.57 | 0.57–0.58 | baseline | baseline ✓ |
| QWEN35_M1024 | 0.469 | 0.935 | 0.50 | 0.50–0.50 | baseline | baseline ✓ |
| DSV2L_M0001 | 0.205 | 0.111 | 1.88 | 1.78–1.91 | candidate | candidate ✓ |
| DSV2L_M0002 | 0.212 | 0.114 | 1.80 | 1.77–1.85 | candidate | candidate ✓ |
| DSV2L_M0004 | 0.189 | 0.121 | 1.60 | 1.51–1.61 | candidate | candidate ✓ |
| DSV2L_M0008 | 0.193 | 0.159 | 1.24 | 1.21–1.44 | candidate | candidate ✓ |
| DSV2L_M0016 | 0.194 | 0.173 | 1.12 | 1.09–1.22 | candidate | candidate ✓ |
| DSV2L_M0032 | 0.197 | 0.205 | 0.97 | 0.96–1.04 | undecided |  |
| DSV2L_M0064 | 0.208 | 0.276 | 0.76 | 0.70–0.80 | baseline | baseline ✓ |
| DSV2L_M0128 | 0.244 | 0.430 | 0.57 | 0.56–0.59 | baseline | baseline ✓ |
| DSV2L_M0256 | 0.319 | 0.692 | 0.46 | 0.46–0.48 | baseline | baseline ✓ |
| DSV2L_M0512 | 0.470 | 1.192 | 0.38 | 0.38–0.39 | baseline | baseline ✓ |
| DSV2L_M1024 | 0.726 | 2.313 | 0.31 | 0.31–0.32 | baseline | baseline ✓ |

## D_MRG16_Q8_G2_K64_W against PULLW_SLT

| unit | PULLW_SLT ms | D_MRG16_Q8_G2_K64_W ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_M0001 | 0.145 | 0.112 | 1.25 | 1.24–1.30 | candidate | candidate ✓ |
| GLM_M0002 | 0.155 | 0.115 | 1.32 | 1.27–1.36 | candidate | candidate ✓ |
| GLM_M0004 | 0.134 | 0.124 | 1.09 | 0.95–1.12 | undecided |  |
| GLM_M0008 | 0.134 | 0.137 | 0.98 | 0.96–0.99 | undecided |  |
| GLM_M0016 | 0.135 | 0.161 | 0.85 | 0.84–0.91 | baseline | baseline ✓ |
| GLM_M0032 | 0.139 | 0.244 | 0.57 | 0.56–0.60 | baseline | baseline ✓ |
| GLM_M0064 | 0.148 | 0.297 | 0.50 | 0.50–0.54 | baseline | baseline ✓ |
| GLM_M0128 | 0.178 | 0.453 | 0.39 | 0.37–0.40 | baseline | baseline ✓ |
| GLM_M0256 | 0.255 | 0.736 | 0.35 | 0.34–0.35 | baseline | baseline ✓ |
| GLM_M0512 | 0.443 | 1.293 | 0.34 | 0.33–0.35 | baseline | baseline ✓ |
| GLM_M1024 | 0.746 | 2.507 | 0.30 | 0.30–0.31 | baseline | baseline ✓ |
| QWEN35_M0001 | 0.175 | 0.140 | 1.25 | 1.23–1.46 | candidate | candidate ✓ |
| QWEN35_M0002 | 0.145 | 0.111 | 1.30 | 1.29–1.34 | candidate |  |
| QWEN35_M0004 | 0.135 | 0.112 | 1.21 | 1.21–1.23 | candidate | candidate ✓ |
| QWEN35_M0008 | 0.135 | 0.120 | 1.11 | 1.01–1.13 | undecided | candidate ✗ |
| QWEN35_M0016 | 0.136 | 0.128 | 1.06 | 1.03–1.07 | candidate |  |
| QWEN35_M0032 | 0.134 | 0.157 | 0.96 | 0.80–1.01 | undecided |  |
| QWEN35_M0064 | 0.137 | 0.171 | 0.80 | 0.78–0.91 | baseline | baseline ✓ |
| QWEN35_M0128 | 0.155 | 0.233 | 0.67 | 0.66–0.76 | baseline | baseline ✓ |
| QWEN35_M0256 | 0.189 | 0.342 | 0.55 | 0.55–0.56 | baseline | baseline ✓ |
| QWEN35_M0512 | 0.272 | 0.557 | 0.49 | 0.48–0.49 | baseline | baseline ✓ |
| QWEN35_M1024 | 0.486 | 0.935 | 0.52 | 0.51–0.52 | baseline | baseline ✓ |
| DSV2L_M0001 | 0.143 | 0.111 | 1.29 | 1.25–1.33 | candidate | candidate ✓ |
| DSV2L_M0002 | 0.143 | 0.114 | 1.23 | 1.22–1.28 | candidate | candidate ✓ |
| DSV2L_M0004 | 0.132 | 0.121 | 1.11 | 1.08–1.15 | candidate | candidate ✓ |
| DSV2L_M0008 | 0.131 | 0.159 | 0.92 | 0.81–0.99 | undecided |  |
| DSV2L_M0016 | 0.144 | 0.173 | 0.86 | 0.83–1.06 | undecided | baseline ✗ |
| DSV2L_M0032 | 0.137 | 0.205 | 0.68 | 0.67–0.93 | baseline | baseline ✓ |
| DSV2L_M0064 | 0.182 | 0.276 | 0.61 | 0.52–0.69 | baseline | baseline ✓ |
| DSV2L_M0128 | 0.178 | 0.430 | 0.41 | 0.41–0.45 | baseline | baseline ✓ |
| DSV2L_M0256 | 0.261 | 0.692 | 0.38 | 0.38–0.38 | baseline | baseline ✓ |
| DSV2L_M0512 | 0.419 | 1.192 | 0.34 | 0.34–0.35 | baseline | baseline ✓ |
| DSV2L_M1024 | 0.722 | 2.313 | 0.31 | 0.31–0.32 | baseline | baseline ✓ |

## PULL_BLAS_R against NATIVE

| unit | NATIVE ms | PULL_BLAS_R ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_DECODE | 0.192 | 0.138 | 1.38 | 1.38–1.46 | candidate | candidate ✓ |
| GLM_PREFILL | 1.601 | 0.982 | 1.68 | 1.60–1.71 | candidate | candidate ✓ |
| QWEN35_DECODE | 0.179 | 0.148 | 1.22 | 1.21–1.22 | candidate | candidate ✓ |
| QWEN35_PREFILL | 1.443 | 1.288 | 1.12 | 1.11–1.13 | candidate | candidate ✓ |
| DSV2L_DECODE | 0.209 | 0.174 | 1.20 | 1.17–1.22 | candidate | candidate ✓ |
| DSV2L_PREFILL | 1.843 | 1.211 | 1.52 | 1.51–1.52 | candidate | candidate ✓ |

## PULL_BLAS_R against BNEW_BLAS_R

| unit | BNEW_BLAS_R ms | PULL_BLAS_R ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_DECODE | 0.252 | 0.138 | 1.87 | 1.74–1.98 | candidate | candidate ✓ |
| GLM_PREFILL | 1.159 | 0.982 | 1.18 | 1.17–1.24 | candidate | candidate ✓ |
| QWEN35_DECODE | 0.276 | 0.148 | 1.80 | 1.80–1.92 | candidate | candidate ✓ |
| QWEN35_PREFILL | 1.605 | 1.288 | 1.25 | 1.19–1.25 | candidate | candidate ✓ |
| DSV2L_DECODE | 0.305 | 0.174 | 1.73 | 1.73–1.78 | candidate | candidate ✓ |
| DSV2L_PREFILL | 1.468 | 1.211 | 1.21 | 1.19–1.24 | candidate | candidate ✓ |

## D_GATHER16_Q8_G2_K64_W_R against NATIVE

| unit | NATIVE ms | D_GATHER16_Q8_G2_K64_W_R ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_DECODE | 0.192 | 0.129 | 1.49 | 1.46–1.59 | candidate | candidate ✓ |
| GLM_PREFILL | 1.601 | 3.422 | 0.47 | 0.46–0.48 | baseline | baseline ✓ |
| QWEN35_DECODE | 0.179 | 0.115 | 1.57 | 1.55–1.62 | candidate | candidate ✓ |
| QWEN35_PREFILL | 1.443 | 2.928 | 0.49 | 0.49–0.49 | baseline | baseline ✓ |
| DSV2L_DECODE | 0.209 | 0.142 | 1.46 | 1.46–1.50 | candidate | candidate ✓ |
| DSV2L_PREFILL | 1.843 | 4.092 | 0.45 | 0.45–0.45 | baseline | baseline ✓ |

## D_GATHER64_Q1_G4_K64_W_R against NATIVE

| unit | NATIVE ms | D_GATHER64_Q1_G4_K64_W_R ms | ratio | range | outcome | predicted |
|---|---|---|---|---|---|---|
| GLM_DECODE | 0.192 | 0.171 | 1.14 | 1.02–1.15 | undecided | candidate ✗ |
| GLM_PREFILL | 1.601 | 1.108 | 1.44 | 1.42–1.49 | candidate | candidate ✓ |
| QWEN35_DECODE | 0.179 | 0.174 | 1.03 | 1.02–1.10 | undecided | candidate ✗ |
| QWEN35_PREFILL | 1.443 | 0.936 | 1.54 | 1.54–1.54 | candidate | candidate ✓ |
| DSV2L_DECODE | 0.209 | 0.170 | 1.23 | 1.18–1.25 | candidate | candidate ✓ |
| DSV2L_PREFILL | 1.843 | 1.447 | 1.27 | 1.26–1.28 | candidate | candidate ✓ |
