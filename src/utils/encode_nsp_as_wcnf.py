import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pathlib import Path

from data_utils import load_nsp, write_dimacs_wcnf

#####
# Nurse Scheduling Problem (NSP) specific functions

def at_most_one_shift(d, m, num_shifts, var_of, f):
    "Nurse m can work at most one shift on day d"
    for s1 in range(num_shifts):
        for s2 in range(s1 + 1, num_shifts):
            f.append([-var_of(d, s1, m), -var_of(d, s2, m)])

def at_least_one_shift(d, m, num_shifts, var_of, f):
    "Nurse m must work at least one shift on day d"
    clause = [var_of(d, s, m) for s in range(num_shifts)]
    f.append(clause)

def off_implicates_no_working(d, m, num_shifts, var_of, f):
    "If nurse m is off on day d, then she cannot work other shifts"
    for s in range(num_shifts-1):
        clause = [-var_of(d, num_shifts-1, m), -var_of(d, s, m)]
        f.append(clause)

def at_least_k_offshifts(m, k, num_shifts, num_days, vpool, var_of, f):
    "Nurse m must have at least k off shifts"
    off_shift = num_shifts - 1
    lits = [var_of(d, off_shift, m) for d in range(num_days)]
    cnf = CardEnc.atleast(lits=lits, bound=k, vpool=vpool, encoding=EncType.seqcounter)
    f.extend(cnf.clauses)

def coverage_constraint(d, s, k, num_nurses, vpool, var_of, f):
    "At least k nurses must work shift s on day d"
    lits = [var_of(d, s, m) for m in range(num_nurses)]
    cnf = CardEnc.atleast(lits=lits, bound=k, vpool=vpool, encoding=EncType.seqcounter)
    f.extend(cnf.clauses)

def get_hard_constraints(num_nurses, num_days, num_shifts, var_of, coverage, one_shift_per_day=True):
    clauses = []
    vpool = IDPool(start_from=num_days * num_shifts * num_nurses + 1) # track next free variable id
    for d in range(num_days):
        for m in range(num_nurses):
            at_least_one_shift(d, m, num_shifts, var_of, clauses)
            if one_shift_per_day:
                at_most_one_shift(d, m, num_shifts, var_of, clauses)
            else:
                off_implicates_no_working(d, m, num_shifts, var_of, clauses)
        for s in range(num_shifts):
            k = coverage[d, s]
            if k > 0:
                coverage_constraint(d, s, k, num_nurses, vpool, var_of, clauses)
    for m in range(num_nurses):
        at_least_k_offshifts(m, 2, num_shifts, num_days, vpool, var_of, clauses)
    weights = [7000] * len(clauses)
    return clauses, weights

def get_soft_constraints(num_nurses, num_days, num_shifts, var_of, preference):
    clauses = []
    weights = []
    for d in range(num_days):
        for s in range(num_shifts):
            for m in range(num_nurses):
                clauses.append([var_of(d, s, m)])
                weights.append(preference[m, d * num_shifts + s])
    return clauses, weights

def write_nsp_to_wcnf(nsp_path, wcnf_path, one_shift_per_day=True):
    num_nurses, num_days, num_shifts, coverage, preference = load_nsp(nsp_path)

    coverage = np.array(coverage)
    preference = np.array(preference)

    def var_of(d, s, m):
        return (d * num_shifts + s) * num_nurses + m + 1
    
    hard_clauses, hard_weights = get_hard_constraints(num_nurses, num_days, num_shifts, var_of, coverage, one_shift_per_day=one_shift_per_day)
    soft_clauses, soft_weights = get_soft_constraints(num_nurses, num_days, num_shifts, var_of, preference)

    clauses = hard_clauses + soft_clauses
    weights = hard_weights + soft_weights

    write_dimacs_wcnf(clauses, weights, wcnf_path)

def convert_all_nsp_to_wcnf(input_dir, output_dir, one_shift_per_day=True):
    """
    Converts every .nsp file in input_dir into a .wcnf file in output_dir.
    :param input_dir: directory containing .nsp files (e.g. 'data/test/NSP')
    :param output_dir: directory to save .wcnf files (e.g. 'data/test/NSP_wcnf')
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nsp_files = sorted(input_dir.glob('*.nsp'))
    for nsp_path in nsp_files:
        wcnf_path = output_dir / f'{nsp_path.stem}.wcnf'
        print(f'Converting {nsp_path} -> {wcnf_path}')
        write_nsp_to_wcnf(str(nsp_path), str(wcnf_path), one_shift_per_day=one_shift_per_day)

if __name__ == "__main__":
    input_dir = "input_dir"  # Replace with the path to your .nsp files
    output_dir = "output_dir"  # Replace with the desired output directory for .wcnf files
    convert_all_nsp_to_wcnf(input_dir, output_dir)