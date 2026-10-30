from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Matched public original-objective SCIP controls with explicit scaling.'
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, time, json, traceback, csv
import numpy as np, psutil
from fractions import Fraction as F
B = Path(__file__).resolve().parent
ROOT = B.parent
OLD = MODULE_DIR
from centered_signal_energy import ObservedAnchor, dot
from run_real_world import coarse
from observed_anchor import sqrt_up, I, matvec, dot as idot
from pyscipopt import Model, quicksum, Eventhdlr, SCIP_EVENTTYPE

def feasible(a, x):
    theta = np.linalg.solve(x.T @ x / a.n + a.lam * np.eye(a.p), x.T @ a.y / a.n)

    def loss(b):
        return (matvec(I(x), I(b)) - I(a.y)).square().sum() / a.n + a.lam * idot(I(b), I(b))
    return float((loss(a.beta) - loss(theta)).lo)

class Trace(Eventhdlr):

    def __init__(self, a, variables, start, common, budget, scale, lower):
        self.a = a
        self.variables = variables
        self.start = start
        self.common = common
        self.budget = budget
        self.scale = scale
        self.lower = lower
        self.primal = -float('inf')
        self.events = []

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED | SCIP_EVENTTYPE.BESTSOLFOUND | SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODEFOCUSED | SCIP_EVENTTYPE.BESTSOLFOUND | SCIP_EVENTTYPE.LPSOLVED, self)

    def eventexec(self, event):
        now = self.common + time.monotonic() - self.start
        if now >= self.budget:
            self.model.interruptSolve()
            return
        primal = float(self.model.getPrimalbound()) / self.scale
        if self.model.getNSols() and primal > self.primal + 1e-12:
            self.primal = primal
            sol = self.model.getBestSol()
            x = self.a.c.copy()
            for (i, j), var in self.variables.items():
                x[i, j] += self.a.rad[i, j] * np.clip(self.model.getSolVal(sol, var), -1, 1)
            self.lower = max(self.lower, feasible(self.a, x))
        self.events.append(dict(seconds=self.common + time.monotonic() - self.start, lower=self.lower, numeric_upper=min(self.a.gradient_upper, float(self.model.getDualbound()) / self.scale), nodes=int(self.model.getNNodes())))

def run(a, common, budget, scale, witness, absolute_tolerance=1e-05):
    start = time.monotonic()
    radius = sqrt_up(a.gradient_upper / a.mu)
    model = Model('matched_regret_displacement')
    model.hideOutput()
    model.setRealParam('limits/absgap', absolute_tolerance * scale)
    model.setRealParam('numerics/feastol', 1e-09)
    model.setRealParam('numerics/dualfeastol', 1e-09)
    model.setIntParam('parallel/maxnthreads', 1)
    z = [model.addVar(f'z{j}', lb=-radius, ub=radius) for j in range(a.p)]
    variables = {}
    evars = {}
    K = a.t.T @ a.G @ a.t
    qb = list(map(F, a.beta))
    g0 = [dot(row, qb) - v for row, v in zip(a.qG, a.qr)]
    hh = [float(dot(list(map(F, col)), g0)) for col in a.t.T]
    objective = 2 * quicksum((hh[j] * z[j] for j in range(a.p))) - quicksum((float(K[j, k]) * z[j] * z[k] for j in range(a.p) for k in range(a.p)))
    for ii, i in enumerate(a.remaining):
        ec = float(a.c[i] @ a.beta - a.y[i])
        ae = ec
        be = ec - quicksum((float(a.ct.lo[ii, j] + a.ct.hi[ii, j]) / 2 * z[j] for j in range(a.p)))
        for j in np.flatnonzero(a.rad[i]):
            s = model.addVar(f's{i}_{j}', lb=-1, ub=1)
            variables[int(i), int(j)] = s
            ae += float(a.rad[i, j] * a.beta[j]) * s
            be += float(a.rad[i, j] * a.beta[j]) * s - float(a.rad[i, j]) * s * quicksum((float(a.t[j, k]) * z[k] for k in range(a.p)))
        bound = abs(ec) + sum((abs(float(a.rad[i, j] * a.beta[j])) for j in np.flatnonzero(a.rad[i]))) + radius * sum(abs(a.c[i] @ a.t)) + radius * sum((a.rad[i, j] * sum(abs(a.t[j])) for j in np.flatnonzero(a.rad[i])))
        ev = model.addVar(f'e{i}', lb=-float(bound), ub=float(bound))
        evars[int(i)] = ev
        model.addCons(ev == be)
        objective += (ae * ae - ev * ev) / a.n
    t = model.addVar('scaled_regret', lb=0, ub=float(a.gradient_upper * scale))
    model.addCons(t <= scale * objective)
    model.setObjective(t, 'maximize')
    seed = model.createSol()
    th = np.array(witness['theta'])
    zm = np.linalg.solve(a.t, a.beta - th)
    for var, value in zip(z, zm):
        model.setSolVal(seed, var, float(value))
    for var in variables.values():
        model.setSolVal(seed, var, 0.0)
    for i, var in evars.items():
        model.setSolVal(seed, var, float(a.c[i] @ th - a.y[i]))
    model.setSolVal(seed, t, max(0.0, (witness['regret_lower'] - 1e-12) * scale))
    accepted = model.addSol(seed, free=True)
    logger = Trace(a, variables, start, common, budget, scale, witness['regret_lower'])
    model.includeEventhdlr(logger, 'trace', 'completed original-objective evidence')
    setup = time.monotonic() - start
    remaining = max(0.001, budget - common - setup)
    model.setRealParam('limits/time', remaining)
    model.optimize()
    x = a.c.copy()
    if model.getNSols():
        for (i, j), var in variables.items():
            x[i, j] += a.rad[i, j] * np.clip(model.getVal(var), -1, 1)
    lower = max(logger.lower, feasible(a, x))
    upper = min(a.gradient_upper, float(model.getDualbound()) / scale)
    total = common + time.monotonic() - start
    return dict(status='PASS', budget_seconds=budget, objective_scale=scale, lower=lower, numeric_upper=upper, numeric_width=upper - lower, target_met=upper - lower <= 1e-05, total_seconds=total, setup_seconds=setup, remaining_internal_limit=remaining, solver_seconds=model.getSolvingTime(), solver_status=str(model.getStatus()), nodes=int(model.getNNodes()), initial_solution_accepted=bool(accepted), upper_kind='SCIP tolerance-based dual, capped by common outward bound', events=logger.events)

def main():
    psutil.Process().cpu_affinity([10])
    source = json.loads((OLD / 'confirmation_results.json').read_text())
    out = B / 'scip_followup_results.json'
    assert not out.exists()
    records = []
    for name in ('concrete', 'superconductivity', 'airfoil', 'powerplant'):
        condition = next((c for c in source['conditions'] if c['dataset'] == name and c['seed'] == 84111))
        original = next((r for r in source['results'] if r['dataset'] == name and r['seed'] == 84111 and (r['method'] == 'energy')))
        cert = json.loads((OLD / original['certificate_file']).read_text())['certificate']
        args = [np.array(cert[k], float) for k in ('lo', 'hi', 'y', 'beta')] + [cert['lambda_value']]
        a = ObservedAnchor(*args, anchor='observed', preconditioner='anchor')
        caps, witness = coarse(a)
        common = sum((condition[k] for k in ('input_seconds', 'anchor_seconds', 'coarse_seconds')))
        for budget, scale in ((10.0, 1000000.0), (30.0, 1000000.0), (10.0, 1.0)):
            row = dict(dataset=name, seed=84111, common_seconds=common, p=a.p, n=a.n)
            try:
                row.update(run(a, common, budget, scale, witness))
                print(name, budget, scale, row['solver_status'], row['numeric_width'], flush=True)
            except Exception as e:
                row.update(status='ERROR', budget_seconds=budget, objective_scale=scale, error=str(e), traceback=traceback.format_exc())
                print(name, budget, scale, 'ERROR', str(e), flush=True)
            records.append(row)
            out.write_text(json.dumps(dict(status='RUNNING', results=records), indent=2))
    out.write_text(json.dumps(dict(status='PASS' if all((r['status'] == 'PASS' for r in records)) else 'COMPLETED_WITH_ERRORS', results=records), indent=2))
    flat = [{k: v for k, v in r.items() if k not in ('events', 'traceback')} for r in records]
    keys = sorted(set().union(*(r.keys() for r in flat)))
    with (B / 'scip_followup_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
if __name__ == '__main__':
    main()
