# Error analysis (auto-generated tables)

Pooled over 15 runs; every patient is a test case once per repeat (3 times). Predictions come from the selective models (trained on 75% of each training fold), matched scenario. Positive class = heart disease.

## 1. What kind of mistakes, and are they the ones that get skipped?

| Model | Missing | Missed disease (FN rate) | False alarm (FP rate) | FN share of errors | Skipped (wrong) | Skipped (correct) | Error (answered) | Error (skipped) |
|---|---|---|---|---|---|---|---|---|
| LR + knn | 0% | 0.199 | 0.122 | 0.580 | 0.476 | 0.188 | 0.108 | 0.321 |
| LR + knn | 30% | 0.242 | 0.163 | 0.558 | 0.392 | 0.152 | 0.151 | 0.390 |
| LR + knn | 50% | 0.261 | 0.183 | 0.548 | 0.432 | 0.155 | 0.158 | 0.439 |
| LR + median | 0% | 0.204 | 0.120 | 0.590 | 0.444 | 0.186 | 0.114 | 0.311 |
| LR + median | 30% | 0.276 | 0.148 | 0.612 | 0.399 | 0.176 | 0.160 | 0.371 |
| LR + median | 50% | 0.295 | 0.177 | 0.586 | 0.400 | 0.152 | 0.175 | 0.442 |
| RF + knn | 0% | 0.213 | 0.124 | 0.593 | 0.467 | 0.169 | 0.113 | 0.354 |
| RF + knn | 30% | 0.257 | 0.175 | 0.554 | 0.440 | 0.172 | 0.154 | 0.409 |
| RF + knn | 50% | 0.281 | 0.213 | 0.527 | 0.423 | 0.166 | 0.183 | 0.452 |
| RF + median | 0% | 0.211 | 0.128 | 0.583 | 0.457 | 0.164 | 0.115 | 0.358 |
| RF + median | 30% | 0.285 | 0.197 | 0.551 | 0.454 | 0.146 | 0.166 | 0.492 |
| RF + median | 50% | 0.293 | 0.232 | 0.517 | 0.377 | 0.152 | 0.205 | 0.466 |

## 2. By patient group (median imputation; 0% and 50% missing)

Groups are small, so differences between groups are descriptive only.

| Model | Missing | Group | Value | Patients | Error rate | Skipped rate | Error (answered) |
|---|---|---|---|---|---|---|---|
| LR | 0% | AgeBand | 50-59 | 125 | 0.165 | 0.243 | 0.137 |
| LR | 0% | AgeBand | <50 | 87 | 0.123 | 0.203 | 0.072 |
| LR | 0% | AgeBand | >=60 | 91 | 0.183 | 0.227 | 0.123 |
| LR | 0% | AnyVesselsColoured | 0 | 176 | 0.172 | 0.241 | 0.120 |
| LR | 0% | AnyVesselsColoured | >=1 | 123 | 0.141 | 0.206 | 0.109 |
| LR | 0% | AnyVesselsColoured | unknown | 4 | 0.083 | 0.250 | 0.000 |
| LR | 0% | ChestPainType | ASY | 144 | 0.155 | 0.241 | 0.104 |
| LR | 0% | ChestPainType | ATA | 50 | 0.140 | 0.107 | 0.134 |
| LR | 0% | ChestPainType | NAP | 86 | 0.120 | 0.209 | 0.064 |
| LR | 0% | ChestPainType | TA | 23 | 0.362 | 0.464 | 0.405 |
| LR | 0% | ExerciseAngina | N | 204 | 0.181 | 0.235 | 0.135 |
| LR | 0% | ExerciseAngina | Y | 99 | 0.111 | 0.209 | 0.072 |
| LR | 0% | Sex | F | 97 | 0.100 | 0.162 | 0.082 |
| LR | 0% | Sex | M | 206 | 0.186 | 0.257 | 0.131 |
| LR | 0% | Thal | fixed | 18 | 0.222 | 0.370 | 0.118 |
| LR | 0% | Thal | normal | 166 | 0.153 | 0.181 | 0.123 |
| LR | 0% | Thal | reversible | 117 | 0.154 | 0.265 | 0.101 |
| LR | 50% | AgeBand | 50-59 | 125 | 0.256 | 0.256 | 0.183 |
| LR | 50% | AgeBand | <50 | 87 | 0.172 | 0.153 | 0.136 |
| LR | 50% | AgeBand | >=60 | 91 | 0.253 | 0.198 | 0.205 |
| LR | 50% | AnyVesselsColoured | 0 | 176 | 0.218 | 0.199 | 0.161 |
| LR | 50% | AnyVesselsColoured | >=1 | 123 | 0.247 | 0.228 | 0.193 |
| LR | 50% | AnyVesselsColoured | unknown | 4 | 0.333 | 0.083 | 0.273 |
| LR | 50% | ChestPainType | ASY | 144 | 0.255 | 0.241 | 0.192 |
| LR | 50% | ChestPainType | ATA | 50 | 0.153 | 0.120 | 0.129 |
| LR | 50% | ChestPainType | NAP | 86 | 0.209 | 0.190 | 0.134 |
| LR | 50% | ChestPainType | TA | 23 | 0.333 | 0.275 | 0.360 |
| LR | 50% | ExerciseAngina | N | 204 | 0.232 | 0.201 | 0.178 |
| LR | 50% | ExerciseAngina | Y | 99 | 0.229 | 0.226 | 0.170 |
| LR | 50% | Sex | F | 97 | 0.172 | 0.189 | 0.127 |
| LR | 50% | Sex | M | 206 | 0.259 | 0.218 | 0.199 |
| LR | 50% | Thal | fixed | 18 | 0.296 | 0.333 | 0.222 |
| LR | 50% | Thal | normal | 166 | 0.203 | 0.169 | 0.162 |
| LR | 50% | Thal | reversible | 117 | 0.265 | 0.248 | 0.193 |
| RF | 0% | AgeBand | 50-59 | 125 | 0.168 | 0.200 | 0.137 |
| RF | 0% | AgeBand | <50 | 87 | 0.107 | 0.153 | 0.063 |
| RF | 0% | AgeBand | >=60 | 91 | 0.220 | 0.286 | 0.138 |
| RF | 0% | AnyVesselsColoured | 0 | 176 | 0.180 | 0.214 | 0.120 |
| RF | 0% | AnyVesselsColoured | >=1 | 123 | 0.152 | 0.211 | 0.110 |
| RF | 0% | AnyVesselsColoured | unknown | 4 | 0.000 | 0.167 | 0.000 |
| RF | 0% | ChestPainType | ASY | 144 | 0.164 | 0.218 | 0.107 |
| RF | 0% | ChestPainType | ATA | 50 | 0.133 | 0.107 | 0.119 |
| RF | 0% | ChestPainType | NAP | 86 | 0.128 | 0.190 | 0.091 |
| RF | 0% | ChestPainType | TA | 23 | 0.391 | 0.493 | 0.314 |
| RF | 0% | ExerciseAngina | N | 204 | 0.194 | 0.208 | 0.136 |
| RF | 0% | ExerciseAngina | Y | 99 | 0.108 | 0.222 | 0.069 |
| RF | 0% | Sex | F | 97 | 0.096 | 0.182 | 0.067 |
| RF | 0% | Sex | M | 206 | 0.199 | 0.227 | 0.138 |
| RF | 0% | Thal | fixed | 18 | 0.167 | 0.278 | 0.128 |
| RF | 0% | Thal | normal | 166 | 0.159 | 0.169 | 0.114 |
| RF | 0% | Thal | reversible | 117 | 0.171 | 0.265 | 0.109 |
| RF | 50% | AgeBand | 50-59 | 125 | 0.261 | 0.256 | 0.197 |
| RF | 50% | AgeBand | <50 | 87 | 0.199 | 0.153 | 0.158 |
| RF | 50% | AgeBand | >=60 | 91 | 0.315 | 0.201 | 0.261 |
| RF | 50% | AnyVesselsColoured | 0 | 176 | 0.256 | 0.191 | 0.197 |
| RF | 50% | AnyVesselsColoured | >=1 | 123 | 0.266 | 0.244 | 0.215 |
| RF | 50% | AnyVesselsColoured | unknown | 4 | 0.250 | 0.000 | 0.250 |
| RF | 50% | ChestPainType | ASY | 144 | 0.282 | 0.229 | 0.216 |
| RF | 50% | ChestPainType | ATA | 50 | 0.167 | 0.160 | 0.111 |
| RF | 50% | ChestPainType | NAP | 86 | 0.225 | 0.190 | 0.196 |
| RF | 50% | ChestPainType | TA | 23 | 0.449 | 0.275 | 0.400 |
| RF | 50% | ExerciseAngina | N | 204 | 0.271 | 0.219 | 0.228 |
| RF | 50% | ExerciseAngina | Y | 99 | 0.236 | 0.192 | 0.158 |
| RF | 50% | Sex | F | 97 | 0.216 | 0.186 | 0.152 |
| RF | 50% | Sex | M | 206 | 0.280 | 0.222 | 0.231 |
| RF | 50% | Thal | fixed | 18 | 0.296 | 0.278 | 0.205 |
| RF | 50% | Thal | normal | 166 | 0.247 | 0.191 | 0.199 |
| RF | 50% | Thal | reversible | 117 | 0.271 | 0.225 | 0.217 |

## 3. Patients misclassified in every one of their 3 test appearances

| Model | Missing | Always wrong | Share of 303 | Never wrong | Skipped (always wrong) | Skipped (never wrong) | Share truly diseased |
|---|---|---|---|---|---|---|---|
| LR + knn | 0% | 33 | 0.109 | 239 | 0.283 | 0.142 | 0.636 |
| LR + knn | 30% | 30 | 0.099 | 204 | 0.244 | 0.100 | 0.667 |
| LR + knn | 50% | 18 | 0.059 | 181 | 0.315 | 0.103 | 0.722 |
| LR + median | 0% | 33 | 0.109 | 238 | 0.242 | 0.139 | 0.636 |
| LR + median | 30% | 26 | 0.086 | 195 | 0.218 | 0.115 | 0.731 |
| LR + median | 50% | 23 | 0.076 | 174 | 0.275 | 0.119 | 0.696 |
| RF + knn | 0% | 36 | 0.119 | 239 | 0.324 | 0.145 | 0.667 |
| RF + knn | 30% | 30 | 0.099 | 197 | 0.322 | 0.108 | 0.633 |
| RF + knn | 50% | 20 | 0.066 | 165 | 0.317 | 0.129 | 0.750 |
| RF + median | 0% | 37 | 0.122 | 239 | 0.306 | 0.130 | 0.649 |
| RF + median | 30% | 31 | 0.102 | 181 | 0.269 | 0.090 | 0.548 |
| RF + median | 50% | 21 | 0.069 | 151 | 0.286 | 0.124 | 0.667 |

### Most confidently wrong patients at 0% missing (full list: `error_analysis_persistent_errors.csv`)

**LR + median**

| Patient (row) | True | Mean P(disease) | Mean confidence | Times skipped (of 3) | Age | Sex | ChestPainType | Cholesterol | MaxHR | ExerciseAngina | Oldpeak | NumVessels | Thal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 145 | 1 | 0.040 | 0.960 | 0 | 47 | M | NAP | 243 | 152 | N | 0.000 | 0 | normal |
| 252 | 0 | 0.941 | 0.941 | 0 | 64 | M | ASY | 263 | 105 | Y | 0.200 | 1 | reversible |
| 32 | 1 | 0.064 | 0.936 | 0 | 64 | M | NAP | 335 | 158 | N | 0.000 | 0 | normal |
| 92 | 0 | 0.904 | 0.904 | 0 | 62 | M | NAP | 231 | 146 | N | 1.800 | 3 | reversible |
| 176 | 0 | 0.896 | 0.896 | 0 | 52 | M | ASY | 233 | 147 | N | 0.100 | 3 | reversible |
| 199 | 1 | 0.136 | 0.864 | 0 | 59 | M | TA | 273 | 125 | N | 0.000 | 0 | normal |
| 44 | 1 | 0.139 | 0.861 | 0 | 61 | F | ASY | 330 | 169 | N | 0.000 | 0 | normal |
| 184 | 1 | 0.144 | 0.856 | 0 | 60 | F | ASY | 305 | 161 | N | 0.000 | 0 | normal |

**RF + median**

| Patient (row) | True | Mean P(disease) | Mean confidence | Times skipped (of 3) | Age | Sex | ChestPainType | Cholesterol | MaxHR | ExerciseAngina | Oldpeak | NumVessels | Thal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 145 | 1 | 0.056 | 0.944 | 0 | 47 | M | NAP | 243 | 152 | N | 0.000 | 0 | normal |
| 252 | 0 | 0.883 | 0.883 | 0 | 64 | M | ASY | 263 | 105 | Y | 0.200 | 1 | reversible |
| 32 | 1 | 0.128 | 0.872 | 0 | 64 | M | NAP | 335 | 158 | N | 0.000 | 0 | normal |
| 92 | 0 | 0.859 | 0.859 | 0 | 62 | M | NAP | 231 | 146 | N | 1.800 | 3 | reversible |
| 184 | 1 | 0.154 | 0.846 | 0 | 60 | F | ASY | 305 | 161 | N | 0.000 | 0 | normal |
| 261 | 1 | 0.214 | 0.786 | 0 | 58 | F | ATA | 319 | 152 | N | 0.000 | 2 | normal |
| 22 | 1 | 0.225 | 0.775 | 0 | 58 | M | ATA | 284 | 160 | N | 1.800 | 0 | normal |
| 44 | 1 | 0.229 | 0.771 | 0 | 61 | F | ASY | 330 | 169 | N | 0.000 | 0 | normal |

