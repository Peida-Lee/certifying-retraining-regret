# Fixed confirmation and component study

Frozen after full-feature development pilot, before these confirmation outcomes,2026-10-28 Asia/Shanghai.

## Primary confirmation

Full-feature Diabetes(n442,p11), Concrete(n1030,p9) and Superconductivity(n1024,p82 including intercept). Two uncertainty patterns2-column/8-column, two widths1/32 and1/8,128 uncertain cells and2 cells per selected row. Five confirmation mask seeds74111,74123,74137,74141,74153. Diabetes and Concrete reuse the public records with new masks; seeds are operating-condition repetitions, not independent participant cohorts. The Superconductivity confirmation pool uses seed74101 sampled after excluding all1024 development records; no development-record overlap.

Candidate midpoint ridge,lambda1/32,original weights1/n. Observed-row anchor and a common midpoint preconditioner. Rank proposal: numerical PCA of the uncertainty-gradient generators; certify each rank1,2,3 in order. Select the first rank with residual-moment tail<=2.5e-6. If none qualifies, choose the smallest tail among these three and mark the requested tail budget unmet. Full rank is separately compared with the same anchor and preconditioner. Total target absolute certificate width3e-6. Each primary full/reduced search gets1second, including its own lower/upper evidence; setup is separately reported and included in total. All60 conditions and both methods are retained. Shared-best-lower comparisons are additional fields only.

## Development ablations and curves

For each of the three development datasets and both uncertainty patterns, run fixed ranks1,2,3,full, ridge-only/observed anchors and input-space baseline with the same candidate andlambda. Use1/10second caps for the selected primary comparison;60seconds is reserved for the high-dimensional concentrated development example. Exact retained objective uses the actual verified dense metric; the original sharp spectral theorem and itsB0/rho curves are also reported separately.

Timing repeats: three fresh repeated executions for the Superconductivity concentrated development rank selected by the input-only rule; report median and range of setup/search/total times. Precision targets1e-4,1e-5,3e-6,1e-6 are evaluated from valid time trajectories, with unattained targets recorded. Final three-way acquisition decision uses U<=tau / L>tau / unresolved.

## Numerical and scientific controls

Independent rational audit of new row query controls, dense anchor geometry, projected-gradient tail and representative complete search trees. The public full-feature construction never drops original input features. For every dataset compare number of uncertain cells and uncertain columns explicitly. Initial spectral-only and residual-moment development outcomes remain in their own files. No universal-win requirement or post-outcome condition deletion.
