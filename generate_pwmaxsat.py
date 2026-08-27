import os
import numpy as np
import cnfgen
import random
from PyMiniSolvers import minisolvers

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

def gen_iclause_pair(min_n, max_n, p_k_2=0.3, p_geo=0.4):
    n = random.randint(min_n, max_n)

    solver = minisolvers.MinisatSolver()
    for i in range(n): solver.new_var(dvar=True)

    iclauses = []

    while True:
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        iclause = generate_k_iclause(n, k)

        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)
        else:
            break

    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat

def write_weighted_dimacs_cnf(clauses, weights, num_var, out_path):
    with open(out_path, "w") as f:
        f.write(f"p wcnf {num_var} {len(clauses)}\n")
        for c, w in zip(clauses, weights):
            f.write(f"{int(w)} " + " ".join(str(int(l)) for l in c) + " 0\n")

def generate_split(out_dir, num_instances, seed,
                    min_n=40, max_n=100, min_k=3, max_k=3, min_alpha=3.0, max_alpha=4.0,
                    min_soft_pct=0.3, max_soft_pct=0.7, soft_weight_min=1, soft_weight_max=100,
                    hard_weight=None):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    for i in range(num_instances):
        #k = rng.randint(min_k, max_k + 1)
        n = rng.randint(min_n, max_n + 1)
        #alpha = rng.uniform(min_alpha, max_alpha)
        #m = max(int(np.ceil(n * alpha)), 1)
        #cnf = cnfgen.RandomKCNF(k, n, m) # to change
        #clauses = [list(cls) for cls in cnf.clauses()] # to change
        _, clauses, _, _ = gen_iclause_pair(min_n=n, max_n=n)

        soft_pct = rng.uniform(min_soft_pct, max_soft_pct)
        num_soft = int(np.ceil(len(clauses) * soft_pct))
        num_hard = len(clauses) - num_soft

        weights = np.zeros(len(clauses), dtype=np.int64)
        hw = hard_weight if hard_weight is not None else max(num_soft, 1) * soft_weight_max + 1
        weights[:num_hard] = hw
        weights[num_hard:] = rng.randint(soft_weight_min, soft_weight_max + 1, size=num_soft)

        num_var = max(abs(l) for c in clauses for l in c)
        write_weighted_dimacs_cnf(clauses, weights, num_var,
                                   os.path.join(out_dir, f"instance_{i:04d}.wcnf"))
        if (i + 1) % 5000 == 0:
            print(f"  {out_dir}: {i + 1}/{num_instances} done")

gen_kwargs = dict(
    min_n=40, max_n=100, min_k=3, max_k=3,
    min_alpha=3.0, max_alpha=4.0,
    min_soft_pct=0.3, max_soft_pct=0.7,
    soft_weight_min=1, soft_weight_max=100,
)

generate_split("data/test/PWMAXSAT", num_instances=30000, seed=1, **gen_kwargs)
generate_split("data/validation/PWMAXSAT", num_instances=10000, seed=2, **gen_kwargs)
print("done")