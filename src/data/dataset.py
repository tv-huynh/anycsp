from src.csp.csp_data import CSP_Data
from src.utils.data_utils import load_dimacs_cnf, load_mtx, load_mc, load_nsp
from src.data.xparser import XParser

from glob import glob
import numpy as np
from tqdm import tqdm
import itertools

import torch
from torch.utils.data import Dataset


def load_cnf_file(path):
    """
    Converts a .dimacs cnf (conjunctive normal form) file into a CSP_Data object
    :param path: the path to a .dimacs cnf file
    :return: A CSP_Data object representing the cnf formula
    """
    cnf = load_dimacs_cnf(path)
    cnf = [np.int64(c) for c in cnf]

    num_var = np.max([np.abs(c).max() for c in cnf]) # as variables are 1-indexed in dimacs format, we can use the max variable index as the number of variables
    num_const = len(cnf) # as each clause is a constraint, the number of constraints is equal to the number of clauses

    arity = np.int64([c.size for c in cnf]) # length of each clause
    const_idx = np.arange(0, num_const, dtype=np.int64) # indices of each constraint
    tuple_idx = np.repeat(const_idx, arity) # each literal in a clause gets a tuple index equal to the index of the clause it belongs to

    cat = np.concatenate(cnf, axis=0)
    var_idx = np.abs(cat) - 1 # 0-indexed variable indices
    val_idx = np.int64(cat > 0).reshape(-1) # 1 if literal is positive, 0 if literal is negative

    data = CSP_Data(num_var=num_var, domain_size=2, path=path) # domain -> {0, 1} for false/true assignment
    data.add_constraint_data(
        True,
        torch.tensor(const_idx),
        torch.tensor(tuple_idx),
        torch.tensor(var_idx),
        torch.tensor(val_idx)
    )
    return data

def load_wcnf_file(path):
    """
    Converts a weighted .wcnf file (DIMACS-style, weight as first token per clause line)
    into a CSP_Data object representing the weighted partial MaxSAT formula.
    """
    cnf, weights = load_dimacs_cnf(path, weighted=True)
    cnf = [np.int64(c) for c in cnf]

    num_var = np.max([np.abs(c).max() for c in cnf])
    num_const = len(cnf)

    arity = np.int64([c.size for c in cnf])
    const_idx = np.arange(0, num_const, dtype=np.int64)
    tuple_idx = np.repeat(const_idx, arity)

    cat = np.concatenate(cnf, axis=0)
    var_idx = np.abs(cat) - 1
    val_idx = np.int64(cat > 0).reshape(-1)

    data = CSP_Data(num_var=num_var, domain_size=2, path=path)
    data.add_constraint_data(
        True,
        torch.tensor(const_idx),
        torch.tensor(tuple_idx),
        torch.tensor(var_idx),
        torch.tensor(val_idx),
        weight=torch.tensor(weights, dtype=torch.float32),
    )
    return data


def nx_to_col(nx_graph, num_colors):
    """
    Converts a NetworkX graph into a CSP_Data object representing the graph coloring problem
    :param nx_graph: The graph as a NetworkX graph object
    :param num_colors: The number of colors to use for coloring the graph
    :return: A CSP_Data object representing the graph coloring problem
    """ 
    num_vert = nx_graph.order()
    num_edges = nx_graph.number_of_edges()

    idx_map = {v: i for i, v in enumerate(nx_graph.nodes())}

    const_idx = np.repeat(np.arange(0, num_edges, dtype=np.int64), num_colors)
    tuple_idx = np.repeat(np.arange(0, num_edges * num_colors, dtype=np.int64), 2)

    vertex_idx = np.int64([[idx_map[u], idx_map[v]] for u, v in nx_graph.edges()])
    vertex_idx = np.tile(vertex_idx, (1, num_colors))
    vertex_idx = vertex_idx.reshape(-1)

    val_idx = np.repeat(np.arange(0, num_colors, dtype=np.int64), 2)
    val_idx = np.tile(val_idx, (num_edges,))

    data = CSP_Data(num_var=num_vert, domain_size=num_colors) # domain -> {0, 1, ..., num_colors - 1} for coloring variables
    data.add_constraint_data(
        True,
        torch.tensor(const_idx),
        torch.tensor(tuple_idx),
        torch.tensor(vertex_idx),
        torch.tensor(val_idx)
    )
    return data


def nx_to_maxcut(nx_graph, edge_weights=None, path=None):
    """
    Converts a NetworkX graph into a CSP_Data object representing the maxcut problem
    :param nx_graph: The graph as a NetworkX graph object
    :param edge_weights: The weights of the edges in the graph, if any
    :param path: The path to the file from which the graph was loaded, if any
    :return: A CSP_Data object representing the maxcut problem
    """ 
    num_vert = nx_graph.order()
    num_edges = nx_graph.number_of_edges()

    idx_map = {v: i for i, v in enumerate(nx_graph.nodes())}

    const_idx = np.repeat(np.arange(0, num_edges, dtype=np.int64), 2)
    tuple_idx = np.repeat(np.arange(0, num_edges * 2, dtype=np.int64), 2)

    vertex_idx = np.int64([[idx_map[u], idx_map[v]] for u, v in nx_graph.edges()])
    vertex_idx = np.tile(vertex_idx, (1, 2))
    vertex_idx = vertex_idx.reshape(-1)

    val_idx = np.repeat(np.arange(0, 2, dtype=np.int64), 2)
    val_idx = np.tile(val_idx, (num_edges,))

    if edge_weights is None:
        edge_weights = np.ones((num_edges,), dtype=np.int64)
    cst_type = (edge_weights + 1) // 2

    data = CSP_Data(num_var=num_vert, domain_size=2, path=path) # domain -> {0, 1} for which side of the cut
    data.add_constraint_data(
        True,
        torch.tensor(const_idx),
        torch.tensor(tuple_idx),
        torch.tensor(vertex_idx),
        torch.tensor(val_idx),
        cst_type=torch.tensor(cst_type)
    )
    return data


def load_xcsp3(path):
    parser = XParser(path)
    data = parser.to_CSP_data()
    return data


def load_mc_file(path):
    g, edge_weights = load_mc(path)
    data = nx_to_maxcut(g, edge_weights, path=path)
    return data

# TODO: debugging, results are weird
def load_nsp_file(path, min_days_off=2):

    num_nurses, num_days, num_shifts, coverage, preference = load_nsp(path)

    coverage = np.int64(coverage)                          # shape (num_days, num_shifts)
    preference = np.int64(preference).reshape(              # shape (num_nurses, num_days, num_shifts)
        num_nurses, num_days, num_shifts
    )

    num_var = num_days * num_shifts * num_nurses
    weight_hard_constraints = 10 * num_var

    def var_of(d, s, m):
        return (d * num_shifts + s) * num_nurses + m

    data = CSP_Data(num_var=num_var, domain_size=2, path=path)

    limit1 = True  # set to False to allow multiple shifts per day per nurse

    if limit1:
        # --- Hard constraint 1: exactly one shift-slot per (nurse, day) ---
        num_const_shift = num_days * num_nurses # 175
        num_tuples_per_const = num_shifts # 4
        total_tuples_shift = num_const_shift * num_tuples_per_const # 700

        d_for_const = np.repeat(np.arange(num_days), num_nurses) # days: [0,...,0,1,...,1,2.....6] each day num_nurses times
        m_for_const = np.tile(np.arange(num_nurses), num_days) # nurses: [0,1,2,...,24,0,...,24,0,....] each group of nurses num_days times

        base_vars = np.stack(
            [var_of(d_for_const, s, m_for_const) for s in range(num_shifts)], axis=1
        )  # shape (num_const_shift, num_shifts) -> first row encodes nurse 0 working first, second, third, and fourth shift on the first day, second row nurse 1, ..., 26th row encodes nurse 0 working on the second day...

        const_idx_1 = np.repeat(np.arange(0, num_const_shift, dtype=np.int64), num_tuples_per_const) # which constraint the entry belongs to (each constraint has 4 tuples)
        tuple_idx_1 = np.repeat(np.arange(0, total_tuples_shift, dtype=np.int64), num_shifts) # 
        var_idx_1 = np.tile(base_vars, (1, num_shifts)).reshape(-1) # which variable each row refers to

        identity = np.eye(num_shifts, dtype=np.int64)
        val_idx_1 = np.tile(identity.reshape(-1), num_const_shift) # what value that variable takes in that specific tuple, the four options: 1000, 0100, 0010, 0001 (repeated 700 times)
        #print(const_idx_1.shape, tuple_idx_1.shape, var_idx_1.shape, val_idx_1.shape)
        data.add_constraint_data(
            False,                                                # negate=False: listed tuples are ALLOWED
            torch.tensor(const_idx_1),
            torch.tensor(tuple_idx_1),
            torch.tensor(var_idx_1),
            torch.tensor(val_idx_1),
            weight = torch.full((num_const_shift,), weight_hard_constraints, dtype=torch.float32),
        )
    else:
        # ----------------------------------------------------------------------
        # HARD CONSTRAINT 1 – multiple real shifts allowed, off‑only exclusive,
        #                     empty set NOT allowed
        # ----------------------------------------------------------------------
        real_shifts = num_shifts - 1               # number of *real* shift types (e.g. 3)

        # Build the allowed binary patterns

        # all 0/1 combos of the real shifts
        all_combos = np.array(
            list(itertools.product([0, 1], repeat=real_shifts)),
            dtype=np.int64
        )                                            # shape (2**real_shifts, real_shifts)

        # keep only the combos that contain at least one ‘1’
        nonempty_combos = all_combos[np.any(all_combos == 1, axis=1)]

        # attach a column of zeros for the off‑bit (still 0)
        real_part = np.hstack([nonempty_combos,
                            np.zeros((nonempty_combos.shape[0], 1), dtype=np.int64)])

        # off‑only tuple (all real bits = 0, off = 1)
        off_only = np.zeros((1, real_shifts + 1), dtype=np.int64)
        off_only[:, -1] = 1

        # final pattern matrix – each row = one allowed tuple
        allowed_patterns = np.concatenate([real_part, off_only], axis=0)   # (num_allowed, num_shifts)
        num_allowed = allowed_patterns.shape[0]                         # = 2**real_shifts
        
        # Variable IDs for each (day,nurse)
        num_const_shift = num_days * num_nurses               # one constraint per (day,nurse)

        d_for_const = np.repeat(np.arange(num_days),  num_nurses)   # (num_const_shift,)
        m_for_const = np.tile(np.arange(num_nurses),   num_days)  # (num_const_shift,)

        # matrix of variable IDs, shape (num_const_shift, num_shifts)
        base_vars = np.stack(
            [var_of(d_for_const, s, m_for_const) for s in range(num_shifts)],
            axis=1
        )

        # a) constraint id per *tuple* (one entry per allowed pattern)
        cst_idx_1 = np.repeat(np.arange(num_const_shift, dtype=np.int64),
                            num_allowed)                       # length = num_const_shift * num_allowed

        # b) global tuple index – repeat each tuple id once per variable in the tuple
        tup_idx_1 = np.repeat(np.arange(num_const_shift * num_allowed, dtype=np.int64),
                            num_shifts)                        # length = num_const_shift * num_allowed * num_shifts

        # c) variable ids – repeat base_vars for every allowed pattern and flatten
        var_idx_1 = np.tile(base_vars, (num_allowed, 1)).reshape(-1)   # same length as tup_idx_1

        # d) the value (0/1) that each variable must take in the corresponding tuple
        val_idx_1 = np.tile(allowed_patterns, (num_const_shift, 1)).reshape(-1)  # same length as var_idx_1

        # sanity check (feel free to comment out later)
        assert cst_idx_1.shape[0] * num_shifts == tup_idx_1.shape[0] == var_idx_1.shape[0] == val_idx_1.shape[0]

        data.add_constraint_data(
            False,                                            # listed tuples are ALLOWED
            torch.tensor(cst_idx_1,  dtype=torch.int64),
            torch.tensor(tup_idx_1,  dtype=torch.int64),
            torch.tensor(var_idx_1,  dtype=torch.int64),
            torch.tensor(val_idx_1, dtype=torch.int64),
            weight=torch.full((num_const_shift,),
                            weight_hard_constraints,
                            dtype=torch.float32),
        )


    # --- Hard constraint 2: max_shifts atMost(k) per nurse
    num_real_shifts = num_shifts - 1  # exclude "off"
    max_shifts = num_days - min_days_off
    m_for_max = np.arange(num_nurses)  # one constraint per nurse

    max_var_idx = np.stack(
        [
            var_of(d, s, m_for_max)
            for d in range(num_days)
            for s in range(num_real_shifts)
        ],
        axis=1
    )  # shape (num_nurses, num_days * num_real_shifts) -> for each nurse, gather every variable id across [day,real-shift] combination, ie, 21 variables per nurse
 
    max_coeffs = np.ones_like(max_var_idx) # coefficients all 1
    max_b = np.full((num_nurses,), max_shifts, dtype=np.int64) # threshold 5
    max_comp = ['le'] * num_nurses # less-or-equal

    data.add_linear_constraint_data(
        torch.tensor(max_var_idx, dtype=torch.int64),
        torch.tensor(max_coeffs, dtype=torch.int64),
        torch.tensor(max_b, dtype=torch.int64),
        max_comp,
        weight=torch.full((num_nurses,), weight_hard_constraints, dtype=torch.float32),
    ) # for each nurse: sum_{d=0}^6 sum_{s=0}^2 x_{d,s,m} \leq 5

    # --- Hard constraint 3: coverage atLeast(k) per (day, shift-slot) ---
    d_for_cov, s_for_cov = np.meshgrid(
        np.arange(num_days), np.arange(num_real_shifts), indexing='ij'
    )
    d_for_cov, s_for_cov = d_for_cov.reshape(-1), s_for_cov.reshape(-1)
    num_cov_cst = d_for_cov.shape[0]  # num_days * num_real_shifts

    cov_var_idx = np.stack(
        [var_of(d_for_cov, s_for_cov, m) for m in range(num_nurses)], axis=1
    )  # shape (num_cov_cst, num_nurses) -- for each (day, shift), gather every nurse's variable

    cov_coeffs = -np.ones_like(cov_var_idx)     # negate for >= via le: -sum(x) <= -coverage
    cov_b = -coverage[d_for_cov, s_for_cov].astype(np.int64)
    cov_comp = ['le'] * num_cov_cst

    data.add_linear_constraint_data(
        torch.tensor(cov_var_idx, dtype=torch.int64),
        torch.tensor(cov_coeffs, dtype=torch.int64),
        torch.tensor(cov_b, dtype=torch.int64),
        cov_comp,
        weight=torch.full((num_cov_cst,), weight_hard_constraints, dtype=torch.float32),
    )

    # --- Soft constraints: preference cost per (nurse, day, shift-slot), including "off" ---
    n_idx, d_idx, s_idx = np.meshgrid(
        np.arange(num_nurses), np.arange(num_days), np.arange(num_shifts), indexing='ij'
    )
    n_idx, d_idx, s_idx = n_idx.reshape(-1), d_idx.reshape(-1), s_idx.reshape(-1)

    num_pref_cst = n_idx.shape[0]
    const_idx_pref = np.arange(0, num_pref_cst, dtype=np.int64)
    tuple_idx_pref = const_idx_pref.copy()                    # arity 1: tuple index == constraint index
    pref_var_idx = var_of(d_idx, s_idx, n_idx)
    val_idx_pref = np.zeros(num_pref_cst, dtype=np.int64)      # cost incurred when x_{d,s,m} == 1
    weights = preference[n_idx, d_idx, s_idx]

    data.add_constraint_data(
        True,                                                  # negate=True: listed tuple is disallowed
        torch.tensor(const_idx_pref),
        torch.tensor(tuple_idx_pref),
        torch.tensor(pref_var_idx),
        torch.tensor(val_idx_pref),
        weight=torch.tensor(weights, dtype=torch.float32)
    )

    return data
#data=load_nsp_file('data/N25/1.nsp', min_days_off=2)

def load_mtx_file(path):
    g, edge_weights = load_mtx(path)
    data = nx_to_maxcut(g, edge_weights, path=path)
    return data


class File_Dataset(Dataset):

    def __init__(self, path, preload=True):
        super(File_Dataset, self).__init__()
        self.path = path
        self.files = glob(path)

        self.preload = preload
        if self.preload:
            print('Loading Data:')
            self.data = [self.load_file(file_path) for file_path in tqdm(self.files)]

    def load_file(self, file_path):
        postfix = file_path.split('.')[-1]
        if postfix == 'xml':
            data = load_xcsp3(file_path)
        elif postfix == 'cnf':
            data = load_cnf_file(file_path)
        elif postfix == 'wcnf':
            data = load_wcnf_file(file_path)
        elif postfix == 'mc':
            data = load_mc_file(file_path)
        elif postfix == 'mtx':
            data = load_mtx_file(file_path)
        elif postfix == 'nsp':
            data = load_nsp_file(file_path)
        else:
            raise ValueError(f'File type {postfix} is not supported.')
        return data

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if self.preload:
            data = self.data[item]
        else:
            file_path = self.files[item]
            data = self.load_file(file_path)
        return data


class Generator_Dataset(Dataset):

    def __init__(self, generators, epoch_samples=1000):
        super(Generator_Dataset, self).__init__()
        self.generators = generators
        self.num_gen = len(self.generators)
        self.epoch_samples = epoch_samples

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, item):
        i = np.random.randint(self.num_gen)
        return self.generators[i].create_random_instance()
