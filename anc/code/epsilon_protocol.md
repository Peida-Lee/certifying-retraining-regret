# Frozen nonzero-residual mechanism experiment

Frozen on 2026-10-28 (Asia/Shanghai), before any v76 epsilon search. This controlled construction extends the v75 original-data residual-energy mechanism. The v75 files remain unchanged.

## Conditions and estimand

Fix m=64, n=128, p=65, lambda=1/2, candidate beta=e0, retained original direction Q=e0, alpha=1/32, and all independent feature intervals exactly as in v75. Signal rows remain x_i=u_i e0, u_i in [3/4,5/4], y_i=2. Nuisance rows remain x_(64+j)=alpha e0+t_j e_j, t_j in [8,16]. Change only their responses to y_(64+j)=alpha+sigma_j delta, where sigma_j=+1 for j=1,...,32 and -1 for j=33,...,64.

The five predeclared conditions, in order, are delta in {0, 2^-15, 2^-14, 2^-13, 2^-12}. They are fixed from the theoretical tail budget, without pilot certificate outputs. All five are reported, including failure to meet a tail or final-width budget. No mask, candidate, coupling, response-sign pattern, or feature interval is selected from the results.

The fixed candidate is the midpoint ridge fit only in the delta=0 control. For delta>0 its nuisance midpoint-gradient entries equal -sigma_j*(3 delta/32), so it is not the perturbed midpoint fit. Q=e0 is explicitly retained from the mechanism design; the original automatic gradient proposal is not claimed to recover it after perturbation. The theorem permits a fixed candidate and retained direction.

## Exact theorem quantities

With w_i=1/128, nuisance candidate residual e_i=-sigma_j delta/sqrt(128),

- epsilon^2=delta^2/2 exactly, epsilon=delta/sqrt(2).
- a^2=alpha^2/2 exactly, a=alpha/sqrt(2).
- G=1/32 and kappa=25/32 are unchanged, computed from the signal rows.
- a epsilon=alpha delta/2, so R=(G+a epsilon)/kappa=(1+16 alpha delta)/25 is rational.
- eta=(epsilon+aR)^2=(delta+alpha R)^2/2 is rational.

For alpha=1/32, R=(1+delta/2)/25 and eta=[1/800+(1601/1600)delta]^2/2. The control and first three nonzero levels have eta<10^-6, while delta=2^-12 has eta>10^-6. This is a tail-only budget classification; a full-table feasible witness can make the final interval narrower than eta. The theorem adds eta to the retained upper and does not force a final width of eta.

In original coordinates, the full optimizer's projection lies in [-R,R]. The implementation uses the actual scalar dyadic ridge-anchor transform T=cI and fixed numerical basis R_num=e0, and rounds R/c upward to the 2^-20 coordinate grid. The theorem tail is evaluated at the full optimizer projection of size <=R, rather than at the slightly enlarged rounded root.

For an additional independent check, the balanced signs leave V_Q=1/(800+512 alpha^2) unchanged. The original optimum is H*=5/(4000+512 alpha^2)+2 delta^2/5. To verify this, eliminate each nuisance coefficient and write f_j=t_j^2/(64+t_j^2). The derivative of the resulting optimized original regret with respect to each f_j is (r_j-alpha s_*)^2/128>=0, where r_j=-sigma_j delta. Thus t_j=16 is maximizing. Balanced signs then cancel the retained linear cross term, and the signal optimum remains all u_i=3/4. This analytic optimum is an extra check and is never used to construct tree bounds or stop the search.

## Search, timing, and retained evidence

Use the v74 original-metric retained point oracle, adaptive node upper bounds (minimum of parent cap, center-gradient, and corner interpolation), original-table feasible ridge witnesses, and common binary-tree driver. A v76 subclass supplies the fixed direction and independently verifies this exact family, radius, and nonzero residual-energy tail. Each condition gets an absolute original-regret width target 10^-6, relative target 0, and search cap 10 seconds. Every point, node, original input, and final witness is saved for all five conditions.

Measure input construction, common anchor setup, method setup, search, setup total, and end-to-end total using a monotonic wall clock. Total includes construction+anchor+method+search and excludes process/import startup and JSON serialization. Timings are single observations. No timing or condition is rerun to improve the table.

The entire queue runs sequentially on logical CPU4 with numerical libraries restricted to one thread; logical CPUs0-3 are reserved. A condition error is retained and the remaining conditions continue. Expected duration is below15minutes, so no long-job monitor is scheduled.

## Independent validation

A separate standard-library Fraction audit imports no search implementation. It reconstructs every input cell and response, candidate, fixed direction, scalar transform, exact midpoint gradient, full-rank minor, epsilon^2, root, and tail. It independently computes each original retained point objective, checks every node bound and exact binary coverage, validates the full-table comparator and original ridge-loss lower bound, and establishes the final upper from tree coverage plus eta and the original gradient cap. It checks the analytic H* only after the certificate proof. Report numerical errors, incomplete budgets, and audit failures explicitly.
