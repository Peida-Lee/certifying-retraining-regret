# arXiv companion files

Unpack the complete source archive. Run reproduction commands below from `anc/`; the manuscript is at the archive root. Public record files and row-containing receipts are regenerated locally.

# Certifying Retraining Regret through Residual Structure

The manuscript and supplementary experiments use one full-feature ridge-regret
objective. The source tree has three parts:

- `../main.tex`, `../sections/`, `../images/`: the attributed preprint, proofs, settings and figures.
- `code/`: numerical algorithms, experiment runners and independent auditors.
- `results/`: the aggregate results reported in the manuscript.

Raw benchmark records and completed-table certificates remain local. Public
download scripts regenerate the required records; audit programs regenerate
and check the table-containing certificates after a full run.

## Offline check

For the refinement runs, use Python 3.13 and install
`requirements-refinement-windows.txt`. `requirements.txt` records the earlier
main-study environment and optional SCIP dependency. The complete general-coarsening
grid uses Python 3.10 and `requirements-general-linux.txt`; its exact environment
is recorded in `results/general_adaptive_v33/environment.json`. Then run:

```text
cd code
python -B verify_offline.py
python -B verify_refinement_offline.py
```

The core check imports thirty-one modules, evaluates twenty-four exact rational
energy cases with nonzero metric cross blocks and retained translations,
checks 48 contraction-refinement cases and the positive-ridge interior-regret example, exercises an eight-unknown
planar adapter, and checks the reported aggregate counts. It opens no raw
records and does not rerun timed search queues.

The refinement check regenerates ten synthetic degree-four certificates in
the original-cell and lifted-row-residual representations. Whole-box and
complementary-halfspace cases check the exact metric, polynomial coefficients,
residual links, row and halfspace localizers, nonnegative multipliers and
PSD-factor dual bounds. It also checks the declared precision labels and
rejects an altered halfspace slack. It requires CVXPY, SCS and Clarabel. The new Windows
refinement runs used Python 3.13.5, NumPy 2.4.2, SciPy 1.16.0, CVXPY 1.7.5,
SCS 3.2.9 and Clarabel 0.11.1. Timed first-stage records retain their original
environment and clocks. Runtime-specific versions are recorded alongside
additional grid results.

## Joint residual refinement

The final method retains the structural first stage and refines unresolved
instances using a joint objective/residual polynomial with an independently
validated conic dual. The original candidate, cell intervals and full-model
comparator remain fixed. `results/refinement_v92/` contains all aggregate
endpoints, complete-time traces and audit summaries from this extension.

The original twenty independent masks reach width `1e-5` in 20/20 instances.
A further twenty masks, seeds 92411--92415, use the frozen
`FROZEN_VALIDATION_V2.json` rules and also reach 20/20 within 76.21 seconds.
Forty primary/control instances reach the target, including all twenty-four
conditions in the matched-control table. The same fifteen decision instances
all certify STOP at their original two-percent threshold within 91.68 seconds.
At the common 120-second checkpoint, the refined method certifies fifteen
STOPs and the centered-moment baseline five; the fixed policy avoids 330
additional acquisitions. The ten-second first-stage comparisons remain
separately reported.

After regenerating the original run directories using the commands below,
the stored-table refinement commands are:

```text
python run_saved_refinement.py --family controls
python run_saved_refinement.py --family independent
python run_saved_refinement.py --family decisions
python run_frozen_refinement.py --seed 92411
python run_frozen_refinement.py --seed 92412
python run_frozen_refinement.py --seed 92413
python run_frozen_refinement.py --seed 92414
python run_frozen_refinement.py --seed 92415
```

The validation runner accepts `--cpu` or `REGRET_CPU`. Its default, and the
extended-baseline runner's default below, selects one allowed logical CPU
outside CPUs 0--3, preserving the two performance cores reserved on the
reference desktop. Numerical libraries use one thread.

The original centered-moment method `M` has a complete ten-second budget.
Its outputs and the separately timed extension are:

| Comparison | Original ten-second M | Extended 600-second M |
| --- | --- | --- |
| Primary datasets | `results/contraction_real/aggregate.json` and `summary.csv` | primary rows in `results/refinement_v92/controls_baseline_extended.json` |
| Matched-control table | `results/matched_controls/aggregate.json` and `summary.csv` | `matched_` rows plus the first-mask `primary_` rows in `results/refinement_v92/controls_baseline_extended.json` |
| New validation masks | `moment_width` and `moment_seconds` in `results/refinement_v92/frozen_validation.json` | `results/refinement_v92/validation_baseline_extended.json` |
| Acquisition decisions | `results/energy_decisions/aggregate.json` and `summary.csv` | `results/refinement_v92/decisions_baseline_extended.json` |

`run_contraction_real.py`, `run_matched_controls.py`, `run_frozen_refinement.py`
and `run_energy_decisions.py` generate the respective ten-second records.
After those public-data runs and the stored-table refinement commands above,
run the extended rank-three moment baseline from `code/`:

```text
python extend_validation_baselines.py --help
python extend_validation_baselines.py --family controls
python extend_validation_baselines.py --family validation
python extend_validation_baselines.py --family decisions
```

These commands read the locally regenerated `frozen_controls`,
`frozen_validation` and `frozen_decisions` directories. Each instance includes
its frozen candidate, row boxes and locally generated table-containing
`*_adaptive_energy.json` certificate. The extension remeasures anchor,
common-cap/witness, proposal, moment constants and search costs, retaining
the input-construction charge recorded in the frozen metadata. It uses the
same `1e-10` search tolerance, rank-three reference and complete 600-second
cap as the saved extension. The solver may finish at its tail/arithmetic
floor before that cap. Errors are recorded per instance and the queue
continues through the remaining instances and families.

New results go to `code/extended_reproduction/`: table-containing
`{family}_baseline_extended/*_moment.json` evidence stays local, while
`{family}_baseline_extended.json` contains only aggregate endpoints and
traces. Use a fresh `--output-root` to repeat a timed experiment. Relative
paths supplied to this runner are resolved from `code/`. `--frozen-root`
selects the parent of regenerated frozen directories, and `--cpu` or
`REGRET_CPU` selects an available logical CPU.

The following command recomputes width counts at 10/60/120/600 seconds and
the 120-second decision comparison directly from the supplied aggregate
records, using Python's standard library:

```text
python extend_validation_baselines.py --aggregate-only --dry-run
```

For the decision comparison it reads
`results/energy_decisions/conditions.json`,
`results/refinement_v92/frozen_decisions.json` and
`results/refinement_v92/decisions_baseline_extended.json`. Each bound is
credited only when its complete elapsed time is within 120 seconds;
moment search traces include their `setup_seconds` offset. STOP is
recomputed with exact rational arithmetic as
`upper <= (51/50) * midpoint_lower`, with condition-key coverage checked
against `decision_settings.json`. This aggregate command verifies saved
evidence; it performs no timed search. `--dry-run` writes no files.

After fresh timed runs, aggregate the new moment and refinement records
with the newly generated midpoint intervals:

```text
python extend_validation_baselines.py --aggregate-only --aggregate-root extended_reproduction --refined-decisions frozen_decisions --decision-conditions energy_decisions/results.json --report extended_reproduction/checkpoint_comparison.json
```

For each output directory (`frozen_controls`, `frozen_independent`,
`frozen_decisions`, `frozen_validation`), run:

```text
python audit_first_stages.py frozen_validation
python audit_sdp_exact.py frozen_validation
python audit_refinement_cover.py frozen_validation
```

Substitute the directory name in all three commands. The audits reconstruct
original-table witnesses, all saved polynomial certificates and complete
subdivision coverage. Public records and table-containing receipts are
generated locally by these commands.

The eighteen multidimensional synthetic cases also have a verified sparse
star-block reference. `python run_multirank_exact.py` regenerates all cases;
`python audit_arrowhead_exact.py` checks their exact scalar oracles and full
trees. The block routine applies after checking its stated sparse row support.

The general 128-cell protocol is `PROTOCOL_ADAPTIVE_V33.json`. For a single case:

```text
python run_adaptive_general_v33.py --condition 0 --cpu 6 --output general_adaptive
python audit_adaptive_general_v33.py general_adaptive/condition_00 --cpu 6
```

Indices 0--19 are Diabetes, 20--39 Concrete, and 40--59 Superconductivity.
Repeat these commands for each condition, choosing an available logical CPU
outside reserved CPUs 0--3. The common target is `3e-6`, with a 9000-second
complete algorithm cap per instance. The algorithm chooses the conic representation
with fewer symmetric matrix entries from the feature dimension and uncertain-cell
counts. Both branches use a one-second structural stage. The moment branch uses
the validated joint polynomial; the compressed branch uses algebraic row
coordinates and exact elimination of fully observed columns when at most three
columns are uncertain. Broader compressed cases use residual slabs. Complete
costs include this choice, input construction, reference and witness fitting,
numerical proposals, exact evaluations and partial child work. Independent
rational replay checks the representation choice, each stage and the complete cover.

The complete sixty-condition grid reaches the original `3e-6` width in 60/60
conditions, with 60/60 independent rational replays passing. The maximum complete
cost is 156.76 seconds. `results/general_adaptive_v33/`
contains every outcome, timing trace, representation choice and audit status.

The additional synthetic verification commands are:

```text
python verify_compressed_v29.py
python verify_profile_v32.py
```

They check original-polynomial coordinate identities, uniform curvature,
Schur elimination against the original full comparator and rejection of
invalid certificates. Original row data are regenerated locally.

## Dense-row edge certificates

The four original dense-row conditions use one common algorithm and the
original `1e-8` precision target. All four attain it and pass independent
rational replay. Complete algorithm times are 37.11 seconds for Diabetes and
61.05, 96.12 and 812.99 seconds for the synthetic 4-, 8- and 16-column cases.
Every model retains all original features. The complete 1800-second cap covers
input construction, anchor and witness/reference fitting, row-edge proposals,
exact screening and every committed bound evaluation. Independent replay has
a separate measured cost.

From `code/`, with an available logical CPU outside reserved CPUs 0--3:

```text
python verify_row_edge_offline.py
python run_row_edge_refinement.py --condition 0 --cpu 4
python audit_row_edge_refinement.py --folder dense_edge/condition_00 --cpu 4
```

Repeat the run and audit for conditions 1, 2 and 3. These denote the synthetic
4-, 8- and 16-column cases. New runs preserve existing directories; use a new
`--output` name to repeat an experiment. The runner's `CONFIG` records the
common numerical and branching rules. `results/refinement_v92/dense_edge_summary.json`
contains the reported aggregate results, with no original row records.

The offline check includes six analytic quartic maxima, 238 exact screened
edge points, four sparse/full/rational oracle comparisons and twenty exact
identities against the original objective majorant. The independent run
auditor reconstructs the original metric and feasible witness, the coupling
PSD certificate, every exclusion and the complete residual partition.

## Real-world experiments

The primary study uses Superconductivity, Parkinsons Telemonitoring, Wine
Quality (white) and Yacht Hydrodynamics. It keeps every documented predictor
and the penalized intercept. Parkinsons excludes the record identifier and
the alternative response; binary sex remains observed. The source's total
UPDRS response is linearly interpolated from clinical assessments.

From `code/`, in a fresh extraction:

```text
python download_public.py
python prepare_public_data.py
python download_contraction_data.py
python run_contraction_real.py
python audit_contraction_real.py
python run_matched_controls.py
python audit_matched_controls.py
python run_scip_matched.py
python run_zorro_matched.py
python audit_zorro_matched.py
python run_zorro_refined.py
python audit_zorro_refined.py
python run_independent_masks.py
python fixed_point_contract.py --source-results contraction_run/results.json --expected-count 20 --output contraction_run/fixed_point_contract.json
python run_energy_decisions.py
python fixed_point_contract.py --source-results energy_decisions/results.json --expected-count 15 --output energy_decisions/fresh_fixed_point_contract.json
python audit_energy_decisions.py
```

The two fixed-point checks create independent reports for the twenty primary
and fifteen decision conditions. They check unique condition keys against
the source's frozen grid as well as the requested count, continue after
individual failures, and return a nonzero exit status for missing, duplicate
or failed cases. `REGRET_SOURCE_RESULTS` and `REGRET_FIXED_POINT_OUTPUT`
remain available as alternatives to the two path options.

The fixed settings are in `contraction_settings.json`. The driver produces
60 runs: four datasets, five masks and three methods. The same rank-three
span, candidate, row boxes, common upper caps and feasible midpoint witness
are used in each paired comparison. Complete budgets include input, anchor,
proposal, verification, search and original-table witness fitting. Imports
and certificate serialization lie outside those clocks. The algorithm
searches down to a `1e-10` tolerance or its tail/arithmetic floor, subject to
the complete ten-second budget; success is assessed at `1e-5` width.

The adaptive contraction-energy method reaches that width in 20/20 cases. Median paired
width reductions against independently centered moments are 89.7%, 38.4%,
20.3% and 25.3%, in the order listed above. Every seed-level endpoint and
timing is retained in `results/contraction_real/summary.csv`; the independent
Fraction audit reconstructs 20 primary receipts and four complete search trees.
The combined adaptive audit reconstructs forty primary/control receipts and
their original-table witnesses. It independently reconstructs both the
contraction constants and the selected projected-gradient/moment constants.
`verify_contraction_offline.py` checks 48 exact normal-equation examples
without reading raw records. The core refinement is `contraction_energy.py`. The default
`AdaptiveEnergySearch` uses the smaller contraction-energy and moment tails
on one common verified containing domain. It pays for both verifications.
Set `REGRET_PURE_ENERGY=1` and fresh `REGRET_OUTPUT` /
`REGRET_PRECISION_OUTPUT` directories to reproduce the pure-energy component;
set `REGRET_PRIMARY_RUN` to that primary directory for its matched controls.
The final primary comparison has nineteen strictly narrower intervals and
one equal interval versus centered moments, with a narrower median on each
of four datasets. Pure-energy records remain alongside adaptive records.

The same-instance external controls use SCIP 10.0.2 / PySCIPOpt 6.3.0 and
ZORRO (NeurIPS 2024). Install `pyscipopt==6.3.0` for optional SCIP runs.
`run_scip_matched.py` defaults to the same `1e-10` search tolerance and full
ten-second cap as the primary solvers. `REGRET_SOURCE_RESULTS` can point to
an existing primary results file. Its unchanged input/candidate cost is
reused; all external-specific anchor, cap, construction and solving costs
are newly measured. The `1e-5` target queue is saved separately.

`zorro_regret.py` is a scoped paper-equation reimplementation of ZORRO's
ridge fixed-point envelope, with the original lambda, shared data symbols,
outward coefficient enclosures and box splitting. A joint-symbol
`g_beta dot (beta-theta)` polynomial converts the optimal-model envelope
to the same regret. Its independent Fraction program reconstructs the
three completed fixed-point systems and adapter uppers exactly. An
incomplete cover returns the existing common upper cap. The full local
receipts remain beside the timed records; the portable release contains
their aggregate audit results and the regeneration/audit programs.
`REGRET_ZORRO_OUTPUT` can select its results file. All twenty midpoint
regret and uncertainty-increment intervals are retained in
`results/matched_external/midpoint_increments.json`.

 Matched precision and
uniform-nuisance-row controls use the same four primary datasets and first
mask (88211). They contain 48 method runs, including eight reused primary
runs and forty new runs. Twenty fresh energy receipts/witnesses are audited
alongside four reused primary audits. The control driver reads the existing
`contraction_run` certificates, keeping every box fixed across precision
levels. The controls charge box/candidate construction and each method's
full certification stages; reading the previously fixed primary tables and
certificate metadata precedes these control clocks.

Optional `REGRET_CORE_DATA`, `REGRET_EXTRA_DATA` and `REGRET_NEW_DATA` variables
can reuse existing public inputs. Their portable defaults are `code/data`.
`REGRET_OUTPUT` can select a fresh run directory; its default is
`code/contraction_run`. The benchmark refuses to overwrite existing results.
Queues log errors and continue with remaining conditions. Numerical libraries
use one thread; timed runs pin CPU10 and independent audits CPU12, reserving
logical CPUs 0--3 on the desktop host. Adapt those affinities to another host
and record the change.

The additional Concrete/Airfoil/Power Plant and earlier Superconductivity
studies remain reproducible with `run_real_world.py`, `audit_real_world.py`,
`run_precision_controls.py`, `run_scip.py` and `audit_scip_objective.py` after
running `download_extra.py`. Their original settings and complete aggregate
records accompany the corresponding supplementary controls. The precision
controls contain 48 runs; the general nonlinear-solver controls contain 12.

## Other manuscript experiments

- Structural separation: `run_structure.py`, `audit_structure.py`.
- Nonzero residuals and leakage: `epsilon_run.py`, `epsilon_audit.py`,
  `leakage_run.py`, `leakage_audit.py`.
- Automatic energy and nuisance scaling: `auto_energy_run.py`,
  `auto_energy_audit.py`, `nuisance_scale.py`.
- Enhanced scalar curvature and the actual dyadic preconditioned moment
  comparison: `curvature_controls.py`, `audit_curvature_controls.py`.
- Multiple directions: `multirank_run.py`, `multirank_audit.py`.
- Intrinsic dimension: `intrinsic_dimension.py`.
- General coarsening and components: `run_confirmation.py`,
  `run_component_base.py`, `run_additional_components.py`. General coarsening
  writes `general_confirmation_results.json`; the real-world energy study
  writes `confirmation_results.json`.
- Metric-matched proposals: `metric_run.py`, `metric_audit.py`.
- Generic tail rank curves and row/penalty controls: `run_tail_diagnostics.py`,
  followed by `python audit_tail_diagnostics.py tail_diagnostics`.
  The full grid contains sixty rank points and forty-eight sensitivity points.
  Every retained Gram is checked exactly; a positive Gershgorin bound or the
  reciprocal exact inverse infinity norm certifies its smallest eigenvalue.
  Aggregate constants and the Gram audit are in `results/tail_diagnostics/`.
- Independently calibrated acquisition: `calibration_run.py`,
  `calibration_analyze.py`, `calibration_sensitivity.py`.
- Dense row geometry: `planar_run.py`, `planar_audit.py`,
  `planar_density_run.py`, `planar_refine.py`.

The joint refinement additionally uses `compressed_polynomial.py` and
`localized_moment.py` to retain each row's affine residual relation and
positive-semidefinite box localizers. `run_localized_refinement.py` runs
the unchanged dense-row or general-coarsening tables. Its output records
the reference rule, moment basis, complete elapsed time, target status,
and every completed covering split. `audit_localized_refinement.py`
reconstructs each polynomial, metric, localizer, dual residual and feasible
original-table witness with rational arithmetic, then verifies the full
cover. A program's `PASS` and its `target_met_in_budget` are separate fields.

The offline refinement checks described above require no public record files.
Local run folders contain row-level receipts and remain outside the portable release.

Experiment-specific protocols and index manifests are supplied beside the
runners. The general-coarsening final-width target is `3e-6`; it is a separate
study from the primary real-world experiment's `1e-5` target. The earlier calibrated
acquisition study uses a regularized-training-objective tolerance of `0.25 K^2`.
The fixed-point predictor decision study below uses the separately specified
two-percent tolerance and its complete fifteen-instance grid.

## Compile the manuscript

Regenerate all five manuscript figures from the supplied aggregate records.
This arXiv companion directory is `anc/`; the manuscript images are at `../images/`.
Run the following commands from this directory:

```text
python code/plot_paper_figures.py --output-dir ../images
python code/plot_final_scope.py --output ../images/scope_final.pdf
```

The first command rebuilds `structure_main_v82.pdf`, `public_energy_v84.pdf`,
`calibration_threshold_v83.pdf` and `zorro_refinement_v91.pdf`; the second
rebuilds `scope_final.pdf`. Both use saved aggregate values without rerunning
solvers or reading row-level data. PNG previews and plotted-value records are
written alongside the PDFs. The four-figure runner also writes editable SVGs
and panel-alignment reports using `audit_panel_alignment.py`.

Compile `../main.tex` using Tectonic or a compatible LaTeX installation.
The ICML style is included. Numerical results and the mathematics were checked
locally; external model feedback is recorded separately from this source tree.

## Structural first-stage records and decision protocol

The paper's main theorem, sharp constant and leakage extension are the core
contribution. Fixed-rank search supplies valid intervals at its tail floor;
the end-to-end precision proposition permits rank enrichment to the full
space and sufficiently accurate queries.

`independent_mask_settings.json` freezes the original five response-independent row
masks on each primary dataset. All twenty instances have one uncertain
signal column and use rank three. Adaptive widths improve in fourteen pairs
and are equal in six, with a smaller marginal median in each dataset. The
independent rational audit covers all twenty receipts/witnesses and four
complete trees. Use `REGRET_INDEPENDENT_OUTPUT` for that runner and set
`REGRET_OUTPUT` to the same directory when calling `audit_contraction_real.py`.

`decision_settings.json` and `results/energy_decisions/FROZEN_DESIGN.json`
record the decision experiment before its runs. Three new mutually disjoint
1024-row Superconductivity pools exclude the named development/scaling
pools. The predictor is fixed from development mask 88211 across all fifteen
new tables. The predefined rule is `H_star <= 1.02 * midpoint_regret`, with
verified midpoint intervals and exact rational decision labels. Every method
has ten complete seconds. A fixed acquisition policy reveals thirty-three
cells unless STOP is certified, and charges each audit instance separately.

At the original ten-second first-stage boundary, pure contraction energy and adaptive energy each give thirteen STOPs,
versus five for moments and zero for the common cap, SCIP and refined ZORRO.
The policy saves 429 cells for energy and 165 for moments across the fixed
grid: 264 additional avoided acquisitions. All ninety records and the fixed
one/two/five-percent sensitivities are included. The independent audit checks
thirty energy component receipts/witnesses, fifteen midpoint intervals, all
exact labels, disjointness and three complete adaptive trees; the three
component trees have the same checked mathematics. Its runner uses
`REGRET_DECISION_OUTPUT` (default `code/energy_decisions`). Full tables and
row-containing receipts stay in the local run directory. The release contains
aggregate audit records and the public-dataset row-index manifests required
to reproduce the specified record pools. The manifests contain indices and
seeds; feature values and responses are loaded locally from the source data.

The coefficient representation is a concrete executable contract: signed
eight-bit coefficients with four fractional bits, signed sixteen-bit inputs
with ten fractional bits and a thirty-two-bit multiply/accumulate. The
checker validates all twenty primary and fifteen new decision tables,
including exact decoded predictions and bounds for every partial sum.

`run_scip_matched.py` now defaults to every primary mask (twenty runs).
`run_zorro_refined.py` uses the same four first-mask instances and spends
the full ten-second budget refining valid envelopes. Pending child work
inherits its parent's legal upper. The independent audit verifies eighty
model-envelope/adapter bounds and complete binary partition coverage.
The original single-envelope results remain available. Set
`REGRET_ZORRO_REFINED_OUTPUT` to select its result file. The raw completed
envelope receipts are regenerated locally and are excluded from the release.
