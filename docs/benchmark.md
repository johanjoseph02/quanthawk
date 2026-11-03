# Benchmark

    uv run qh evaluate --seeds 0 1 2     # about 8 minutes on an M1 Max

For each seed the simulator (`quanthawk.simulate`) builds a bank of 3,000 customers over
120 days. Every transaction is replayed through the feature engine in time order. Models
train on the first 60 days and are scored on the last 60. Fraud in the training window
is left in, because an unsupervised model in a bank never gets a clean history. Results
are mean ± standard deviation over the three seeds. All of them are in
`results/benchmark.json`.

## Data

<!-- DATA -->
| Seed | Train transactions | Test transactions | Fraud in train | Fraud in test | Test fraud rate | Fraud episodes in test |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 591,853 | 604,683 | 499 | 640 | 0.106% | 104 |
| 1 | 603,126 | 614,390 | 474 | 702 | 0.114% | 108 |
| 2 | 600,534 | 615,789 | 454 | 662 | 0.108% | 111 |
<!-- /DATA -->

About one transaction in a thousand is fraud: realistic for card and transfer fraud,
and the reason the metrics below are taken at an **alert budget**. A fraud team works a
fixed number of alerts a day, so what matters is how much fraud is in the top 0.5 % or 1 %
of scores. Average precision summarises the whole ranking; ROC AUC is shown because it
is common, but at this class balance it flatters every model.

An **episode** is one defrauded account. It counts as caught if any of its fraudulent
transactions is alerted.

## Models

| Model | What it is |
|---|---|
| rules | Four rules of the kind a bank starts with: large and to a new payee or device; five or more transactions in an hour; a large ATM withdrawal in a new city; most of a new account's inflow passed straight on. Scored by how many fire. |
| isolation forest, global | One Isolation Forest (200 trees, 256 samples each) on every customer's transactions. |
| **k-means + isolation forest (Quant Hawk)** | K-means on account profiles, k chosen by silhouette, one forest per segment, a cold-start forest for new accounts (see [design.md](design.md)). |
| isolation forest per true archetype (oracle) | One forest per *true* customer archetype, which the simulator knows and a bank does not. A reference for how much better clustering could help. |
| gradient boosting, supervised (reference) | Histogram gradient boosting on the same features, trained on the training window's *labels*. Not unsupervised; shown for what labels would buy. |

## Results

<!-- MAIN -->
| Model | Average precision | ROC AUC | Recall @ 0.5% | Precision @ 0.5% | Recall @ 1% | Precision @ 1% | Episodes caught @ 1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| rules | 0.024 ± 0.002 | 0.743 ± 0.010 | 14.0 ± 0.7 | 3.1 ± 0.3 | 15.9 ± 0.8 | 1.7 ± 0.1 | 58.3 ± 4.0 |
| isolation forest, global | 0.093 ± 0.023 | 0.919 ± 0.005 | 33.5 ± 3.3 | 7.3 ± 0.6 | 43.5 ± 3.7 | 4.7 ± 0.4 | 64.8 ± 4.6 |
| **k-means + isolation forest (Quant Hawk)** | 0.367 ± 0.054 | 0.950 ± 0.005 | 57.1 ± 3.6 | 12.5 ± 0.9 | 63.1 ± 3.3 | 6.9 ± 0.4 | 82.1 ± 1.9 |
| isolation forest per true archetype (oracle) | 0.332 ± 0.062 | 0.937 ± 0.004 | 58.7 ± 3.3 | 12.8 ± 0.7 | 64.8 ± 3.0 | 7.1 ± 0.4 | 81.4 ± 0.9 |
| gradient boosting, supervised (reference) | 0.662 ± 0.068 | 0.992 ± 0.005 | 84.7 ± 2.6 | 18.5 ± 1.1 | 89.8 ± 1.5 | 9.8 ± 0.5 | 97.5 ± 1.0 |
<!-- /MAIN -->

### Which fraud gets caught (recall at a 1 % budget)

<!-- TYPES -->
| Model | account takeover | card testing | atm cashout | mule | low and slow |
|---|---:|---:|---:|---:|---:|
| rules | 4% | 17% | 24% | 33% | 3% |
| isolation forest, global | 45% | 88% | 14% | 49% | 5% |
| k-means + isolation forest (Quant Hawk) | 84% | 95% | 49% | 88% | 10% |
| isolation forest per true archetype (oracle) | 90% | 97% | 56% | 88% | 7% |
| gradient boosting, supervised (reference) | 97% | 100% | 100% | 93% | 71% |
<!-- /TYPES -->

### Who gets wrongly alerted (false-alert rate on legitimate transactions, 1 % budget)

<!-- ARCHETYPES -->
| Model | gig worker | high net worth | retiree | salaried | small business | student |
|---|---:|---:|---:|---:|---:|---:|
| rules | 1.66% | 0.19% | 0.05% | 0.15% | 3.20% | 0.19% |
| isolation forest, global | 0.25% | 0.16% | 0.06% | 0.08% | 4.68% | 0.07% |
| k-means + isolation forest (Quant Hawk) | 0.93% | 1.12% | 1.10% | 0.93% | 0.71% | 1.07% |
| isolation forest per true archetype (oracle) | 1.09% | 0.83% | 1.30% | 0.88% | 0.69% | 1.03% |
| gradient boosting, supervised (reference) | 0.91% | 1.24% | 1.12% | 0.79% | 1.05% | 0.82% |
<!-- /ARCHETYPES -->

### What K-means finds

The silhouette score picks the number of segments. The segments are then compared with
the archetypes the simulator used, which the model never saw.

<!-- SEGMENTS -->
| Seed | k chosen | Adjusted Rand index vs archetypes | Segments (majority archetype, share) |
|---:|---:|---:|---|
| 0 | 5 | 0.71 | 0: high net worth 87%; 1: salaried 71%; 2: small business 100%; 3: gig worker 100%; 4: retiree 100% |
| 1 | 5 | 0.69 | 0: salaried 69%; 1: small business 100%; 2: gig worker 100%; 3: high net worth 67%; 4: retiree 100% |
| 2 | 5 | 0.69 | 0: high net worth 71%; 1: small business 100%; 2: gig worker 100%; 3: retiree 100%; 4: salaried 68% |
<!-- /SEGMENTS -->

### Does the number of segments matter? (seed 0)

<!-- ABLATION -->
| k | Average precision | Recall @ 1% | Precision @ 1% |
|---:|---:|---:|---:|
| 1 (a single segment) | 0.164 | 48.4% | 5.1% |
| 3 | 0.368 | 64.8% | 6.9% |
| 6 | 0.407 | 65.5% | 6.9% |
| 10 | 0.404 | 67.0% | 7.1% |
<!-- /ABLATION -->

### A bank with one kind of customer (seed 0)

Every customer salaried. If segmenting only helped because the simulator has archetypes,
it should stop helping here.

<!-- HOMOGENEOUS -->
| Model | Average precision |
|---|---:|
| isolation forest, global | 0.248 |
| k-means + isolation forest (k = 4) | 0.293 |
<!-- /HOMOGENEOUS -->

## Findings

1. **Segmenting first roughly quadruples average precision.** 0.37 ± 0.05 against
   0.09 ± 0.02 for a single forest. At a 1 % budget it finds 63 % of fraudulent
   transactions instead of 44 %, and catches 82 % of defrauded accounts instead of 65 %.
   The hand-written rules are far behind both.

2. **The global forest spends its budget on small businesses.** It alerts on 4.7 % of
   small businesses' legitimate transactions and under 0.3 % of anyone else's: for a
   single model of normal, a business paying suppliers is the most unusual thing in the
   bank. With segments, every kind of customer sits between 0.7 % and 1.1 %.

3. **The gains are where customers differ most.** Account takeover goes from 45 % to 84 %
   caught, mule activity from 49 % to 88 %, ATM cash-outs from 14 % to 49 %. Card testing
   was already easy (a burst of tiny online payments looks odd for anyone).

4. **K-means finds the customer types on its own.** It chose five segments on every seed.
   Three of them are made up entirely of one archetype (small businesses, gig workers,
   retirees), and the adjusted Rand index against the hidden archetypes is about 0.7.
   Students and salaried customers end up together; their habits overlap. The discovered
   segments do as well as the oracle that knows the true archetypes: higher average
   precision (0.37 against 0.33), slightly lower recall at 1 % (63 % against 65 %). They
   gain by also separating customers *within* an archetype by how much they spend.

5. **Segmenting helps even when everyone is the same kind of customer** (0.29 against
   0.25), for the same reason: customers of one type still differ by scale. It did not
   hurt in any configuration tried. Between 5 and 10 segments the results are flat; one
   segment is clearly worse.

6. **Unsupervised detection has a ceiling.** Gradient boosting trained on labels reaches
   0.66 average precision and catches 71 % of "low and slow" fraud, which no anomaly
   detector here catches more than 10 % of: small, plausible payments spread over days
   are not anomalies. The practical design is both: Quant Hawk to find fraud nobody has
   labelled yet, and analyst verdicts (stored by the service) to build the labels a
   supervised model needs.

## A bug the live service found

The first version of the features measured money received in the last 24 hours as an
absolute amount. The offline benchmark looked fine. Running the service on a live stream
(`qh demo`) did not: the first day of the stream was the simulated payday, and the service
alerted on 8 % of transactions against a 1 % budget. With two paydays in a 60-day training
window, the forest had learned that a salary arriving is rare, and so anomalous.

The drift monitor pointed at the feature (population stability index 0.48 on `in_24h`).
The fix was to count only money from payers the account has never received from, and to
compare it with the largest payment the account received *before* the last day. Payday
then alerted on 0.95 % of transactions. The change also lifted average precision from
0.31 to 0.37, because the new feature describes mule activity more exactly than the old
one did. `test_payday_is_ordinary_for_a_salaried_customer` keeps it fixed.

## Limits

- The data is synthetic. The simulator is designed to be hard in realistic ways, and
  every assumption is in one file, but a real bank will behave differently, and its
  fraud will be shaped by its own controls.
- One feature engine, one set of fraud types. A model tuned to this simulator could
  overfit to it; the homogeneous-bank and number-of-segments checks are there to catch the
  most obvious ways that could happen.
- Results vary by seed (see the standard deviations): with roughly 100 fraud episodes in
  each test window, a difference of a few points between two models is not significant.
