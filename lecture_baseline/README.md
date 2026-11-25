# Lecture baseline: predict prices by MSE, then optimize battery actions

This implementation follows the first, two-step baseline marked in the supplied
lecture notes. It does not implement the later robust-optimization,
decision-focused-training, or direct-policy sections. This file documents code;
it is not prose for the course report, which the team must write itself.

## Map the lecture notation to this project

| Notes | Battery project | Implementation |
| --- | --- | --- |
| z_i, observed covariates | Calendar, current net-load forecast, preceding seven days | feature_vector(), a length-130 daily vector |
| y_i, unknown quantity | Current day's 24 realized DA electricity prices | y, length 24; used as training labels and for scoring only |
| g_theta(z_i) | Predicted 24-price daily vector | predict_prices() |
| x, decision | 24 grid exchanges p_t (positive charge, negative discharge) | optimize_schedule(), output shape (24,1) |
| f(x,y) | Daily purchase cost y^T x, the negative of revenue | LP/MILP objective |
| X | Feasible schedules conditional on the actual initial energy | Battery dynamics, hourly limits, energy bounds |

## Exact lecture method

1. Construct daily pairs (z_i,y_i) using only data available when scheduling day i.
2. Fit an affine price model g(z)=Az+b by minimizing average squared price error.
3. At inference, predict y_hat=g(z) without observing y.
4. Solve min_x y_hat^T x over the appropriate physical feasible set.
5. After committing to x, score revenue as -y^T x using realized prices.

The default model is ordinary least squares (OLS), alpha=0. Training standardizes
features using training rows only and fits an intercept. This is still an affine
function of the original z. NumPy's SVD least-squares solver handles rank
deficiency and non-unique coefficients. Internal weights have shape (130,24),
the transpose orientation of A in the lecture's column-vector convention.

The default optimizer is myopic: it has no terminal energy value and no terminal
energy equality constraint. It carries the actual end-of-day energy into the
next day. days_remaining is accepted to match the required interface but is
intentionally unused by this particular daily baseline. This is a simplification
of the full multiday project objective; adding future energy value is a later
modeling extension, not part of this exact implementation of the lecture baseline.

The lossless case is an LP directly in p. If L is the cumulative-sum matrix,
the energy constraints are -s0 <= Lp <= 5-s0 and -2.5 <= p <= 2.5.

The lossy case is a MILP in charging c, discharging d, energy s, and binary mode u:
s[t+1]=s[t]+0.95*c[t]-d[t]/0.95,
0<=c[t]<=2.5*u[t], 0<=d[t]<=2.5*(1-u[t]), 0<=s[t]<=5.
The returned net exchange is p=c-d. Binary modes prevent simultaneous charging
and discharging, including under negative prices.

No final state is forced: charging at a negative final price may earn money even
if the stored energy is never subsequently sold. The solver limit is 5 seconds
per call. Available incumbents are physically clipped if needed; no incumbent
produces a feasible idle schedule. A time-limited incumbent is not a certificate
of optimality. No solver early stops occurred in the recorded experiments.

## Feature construction

Each day contributes one z vector and one y vector, not 24 unrelated days.

| Feature block | Entries |
| --- | ---: |
| Current day's 24 hourly net-load forecasts | 24 |
| Yesterday's 24 realized prices | 24 |
| Same weekday last week's 24 prices | 24 |
| Mean price at each hour over the preceding week | 24 |
| Mean realized-minus-forecast net-load error at each hour over that week | 24 |
| Seasonal sine/cosine and normalized day-of-month | 3 |
| Weekday one-hot indicators (1=Sunday,...,7=Saturday) | 7 |
| Total | 130 |

Net-load values are divided by 10,000 before the training-fitted standardization.
The seasonal angle uses month and day directly; no observation year is guessed.
The feature design is an implementation choice; the notes do not prescribe it.
Initial energy enters the optimizer, not the market-price model.

Historical matrices are (24,7), columns yesterday through seven days ago.
The original 168 initial-history rows are prepended before forming training
examples. The current day's realized load and prices never enter its features.

## Run

Extract the archive, open this directory in a terminal, and use Python 3.

```bash
python -m pip install -r requirements.txt
python train.py --data-dir data
python demo.py --data-dir data
python checks.py --data-dir data
python evaluate.py --data-dir data --output-dir results
```

To limit CPU threading consistently across runs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

train.py defaults to OLS and fits all released training days for deployment.
It saves price_model.npz next to the scheduling code. Inference loads only that
local NPZ with allow_pickle=False; no CSVs or network resources are read.
The saved model is cached within a Python process. Restart Python after changing
the weight file (or call baseline_core.load_model.cache_clear()).

demo.py calls both actual required schedulers using the first day's information,
then computes revenue afterwards. With full-year fitted weights, this is an
execution demonstration, not an out-of-sample performance estimate.

## Optional Ridge comparison

The notes' unregularized MSE model can overfit. A clearly labeled optional extension
adds alpha*||W||_F^2 to the sum of squared standardized-feature residuals.
The intercept is unpenalized. The sum, rather than mean, convention determines
the numerical meaning of alpha.

```bash
python train.py --data-dir data --alpha 100
python checks.py --data-dir data
python demo.py --data-dir data
```

This overwrites the deployment price_model.npz with Ridge weights. To restore
the exact unregularized baseline, retrain with --alpha 0 (the default).
The archive also includes price_model_ridge.npz, fitted with alpha=100 on all
released days, as an explicitly separate optional artifact. Neither full-data
model was used to generate held-out results.

## Chronological evaluation and actual findings

evaluate.py fits on January-June (182 days), validates on July-September (92),
refits on January-September (274), and evaluates on October-December (92).
OLS has no tuned hyperparameter. For the optional Ridge extension only,
alpha is selected from [0.1,1,10,100] using July-September prediction MSE.
It selected 100. The holdout is not used to select alpha or change the method.
Price models are trained for prediction error, not differentiated through the
scheduler or tuned against scheduling revenue.

Each simulation starts at 5 MWh and carries energy across days. Historical
observations from earlier held-out days are legitimate inputs for later days.
No model is retrained during the held-out run.

| Forecast method | Holdout price RMSE ($/MWh) | Lossless revenue ($) | Lossy revenue ($) |
| --- | ---: | ---: | ---: |
| Lecture OLS, alpha=0 | 26.8966 | 17,571.18 | 13,530.27 |
| Optional Ridge, alpha=100 | 8.5002 | 21,187.97 | 17,345.49 |
| Seven-day same-hour mean | 9.2657 | 21,556.83 | 17,997.34 |

All schedules in these experiments were feasible. Unregularized OLS performed
particularly poorly on July-September validation (price RMSE 299.08; negative
revenue for both batteries). Do not hide this result: fitting the lecture's MSE
objective correctly does not ensure generalization. A small daily sample size,
correlated features, and changing market conditions can make OLS unstable.
These experiments show improvement from regularization; they do not establish
that Ridge is the best possible model, and the historical-average benchmark
still earned more holdout revenue. Full metrics and daily traces are included.

Runtime columns measure scheduling optimization and physical correction only,
not price-model training/prediction. Timings are hardware dependent.

## Correctness verification

checks.py tests exact recovery of a synthetic affine map, the Ridge normal
equations, historical matrix ordering, and resistance to current-day label/load
leakage by modifying those observations in temporary CSVs. It compares 32 small
optimization problems with an independent exhaustive mode-enumeration oracle.
It also checks full/empty/fractional initial energies and negative prices.

Both required entry points were called chronologically across all 366 released
days (732 calls), with 12 extra boundary/fractional-state cases. They imported
independently and returned finite (24,1) arrays satisfying physical bounds.
The full-year checks use a model fitted on all released days, so they establish
execution/feasibility only and are not generalization experiments.

## Files and submission

- baseline_core.py: feature vector, affine regression, safe NPZ loading, LP/MILP.
- dataset.py: supplied CSV processing and historical windows.
- train.py: offline fitting; default literal OLS baseline, optional Ridge flag.
- evaluate.py: chronological prediction and revenue experiments.
- checks.py: independent numerical, leakage, and physical tests.
- demo.py: a visible one-day example.
- schedule_lossless.py and schedule_lossy.py: the required independent entry points.
- price_model.npz: default OLS deployment model; full released year only.
- price_model_ridge.npz: optional full-year Ridge model.
- data/: unmodified released CSVs.
- results/: metrics, daily traces, Ridge selection, environment, verification summary.

Keep schedule_lossless.py, schedule_lossy.py, baseline_core.py, and the selected
price_model.npz together at the root of a submission. Include all data-processing,
training, and evaluation code and this README as required by the specification.
The scheduler uses only NumPy/SciPy plus Python's standard library. pandas is
used offline for CSVs. No unapproved solver package or pretrained external model
is needed. Confirm the evaluator has SciPy >=1.9, which provides scipy.optimize.milp.
The recorded environment is Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3.

## Attribution and AI assistance

Method: the uploaded course lecture notes' two-step, certainty-equivalent
predict-then-optimize baseline, applied to the released battery specification.
Optimization formulations and least-squares fitting use standard mathematical
methods. No outside datasets or externally trained weights were used.

ChatGPT assisted with translating the notes into code, designing features,
implementing optimization and regression, and running/debugging the recorded
checks and experiments. The team should review this disclosure, record its own
contributions, and understand and verify the code before submission. The team
must write its own submitted report and develop its own analysis.
