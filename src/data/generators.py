from src.csp.csp_data import CSP_Data
from src.data.dataset import nx_to_col, nx_to_maxcut
from src.utils.rb_utils import get_random_RB

import numpy as np
import networkx as nx
import torch
import cnfgen


class KSAT_Generator:

    def __init__(self, min_n=100, max_n=100, min_k=3, max_k=3, min_alpha=4.0, max_alpha=5.0):
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def create_random_instance(self):
        k = np.random.randint(self.min_k, self.max_k + 1)
        n = np.random.randint(self.min_n, self.max_n + 1)
        alpha = np.random.uniform(self.min_alpha, self.max_alpha)
        m = max(int(np.ceil(n * alpha)), 1)
        cnf = cnfgen.RandomKCNF(k, n, m)
        
        cnf = [cls for cls in cnf.clauses()]
        cnf = [np.int64(c) for c in cnf]

        num_var = np.max([np.abs(c).max() for c in cnf])
        num_const = len(cnf)

        arity = np.int64([c.size for c in cnf])
        const_idx = np.arange(0, num_const, dtype=np.int64)
        tuple_idx = np.repeat(const_idx, arity)

        cat = np.concatenate(cnf, axis=0)
        var_idx = np.abs(cat) - 1
        val_idx = np.int64(cat > 0).reshape(-1)

        data = CSP_Data(num_var=num_var, domain_size=2)
        data.add_constraint_data(
            True,
            torch.tensor(const_idx),
            torch.tensor(tuple_idx),
            torch.tensor(var_idx),
            torch.tensor(val_idx)
        )
        return data

class PWKSAT_Generator:

    def __init__(self, min_n=100, max_n=100, min_k=3, max_k=3, min_alpha=4.0, max_alpha=5.0, min_soft_pct=0.3, max_soft_pct=0.7, soft_weight_min=1, soft_weight_max=100, hard_weight=None):
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_soft_pct = min_soft_pct
        self.max_soft_pct = max_soft_pct
        self.soft_weight_min = soft_weight_min
        self.soft_weight_max = soft_weight_max
        self.hard_weight = hard_weight # if None, derived below

    def create_random_instance(self):
        k = np.random.randint(self.min_k, self.max_k + 1)
        n = np.random.randint(self.min_n, self.max_n + 1)
        alpha = np.random.uniform(self.min_alpha, self.max_alpha)
        m = max(int(np.ceil(n * alpha)), 1)
        cnf = cnfgen.RandomKCNF(k, n, m)
        
        cnf = [cls for cls in cnf.clauses()]
        cnf = [np.int64(c) for c in cnf]

        # Split total number of clauses m into hard and soft clauses
        soft_pct = np.random.uniform(self.min_soft_pct, self.max_soft_pct)
        num_soft = int(np.ceil(m * soft_pct))
        num_hard = m - num_soft 

        # Assign weights to hard and soft clauses
        weights = np.zeros(len(cnf), dtype=np.int64)
        if self.hard_weight is not None:
            hard_weight = self.hard_weight
        else:
            hard_weight = max(num_soft, 1) * self.soft_weight_max + 1
        weights[:num_hard] = hard_weight
        weights[num_hard:] = np.random.uniform(self.soft_weight_min, self.soft_weight_max + 1, size=num_soft)

        # Same as KSAT_Generator, but with weights
        num_var = np.max([np.abs(c).max() for c in cnf])
        num_const = len(cnf)

        arity = np.int64([c.size for c in cnf])
        const_idx = np.arange(0, num_const, dtype=np.int64)
        tuple_idx = np.repeat(const_idx, arity)

        cat = np.concatenate(cnf, axis=0)
        var_idx = np.abs(cat) - 1
        val_idx = np.int64(cat > 0).reshape(-1)

        data = CSP_Data(num_var=num_var, domain_size=2)
        data.add_constraint_data(
            True,
            torch.tensor(const_idx),
            torch.tensor(tuple_idx),
            torch.tensor(var_idx),
            torch.tensor(val_idx),
            weight=torch.tensor(weights, dtype=torch.float32),
        )
        return data

# Test if the data pipeline is correctly modified to accept, store, move-to-device, and re-concatenate the weight tensor
"""data = PWKSAT_Generator(min_n=10, max_n=10, min_k=3, max_k=3, min_alpha=4.0, max_alpha=4.0, min_soft_pct=0.3, max_soft_pct=0.7, soft_weight_min=1, soft_weight_max=100).create_random_instance()
print(data.constraints['ext'].weight)
print(data.constraints['ext'].weight.shape)
data.init_adj()
assignment = data.hard_assign_sample(torch.ones(data.num_val))[0]
print(data.count_unsat(assignment)) 

data = KSAT_Generator(min_n=20, max_n=20).create_random_instance()
data.init_adj()
print(data.weight)          # should be all 1.0
print(data.batch_total_weight)  # should exactly equal data.batch_num_cst
print(data.batch_num_cst)
data.init_adj()
assignment = data.hard_assign_sample(torch.ones(data.num_val))[0]
print(data.count_unsat(assignment)) """

class COL_Generator_Base:

    def __init__(self, min_col, max_col):
        self.min_col = min_col
        self.max_col = max_col

    def sample_nx_graph_(self):
        raise NotImplementedError

    def create_random_instance(self):
        G = self.sample_nx_graph_()
        coloring = nx.greedy_color(G)
        greedy_num_col = max([c for v, c in coloring.items()]) + 1
        min_col = max(self.min_col, min(self.max_col, greedy_num_col - 1))
        max_col = max(self.min_col, min(self.max_col, greedy_num_col - 1))
        k = np.random.randint(min_col, max_col + 1)
        data = nx_to_col(G, k)
        return data


class COL_GNM_Generator(COL_Generator_Base):

    def __init__(self, min_n, max_n, min_deg, max_deg, min_col, max_col):
        self.min_n = min_n
        self.max_n = max_n
        self.min_deg = min_deg
        self.max_deg = max_deg
        super(COL_GNM_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        deg = np.random.uniform(self.min_deg, self.max_deg)
        num_edge = int(np.ceil(n * deg / 2.0))
        G = nx.gnm_random_graph(n, num_edge)
        return G


class COL_ER_Generator(COL_Generator_Base):

    def __init__(self, min_n=20, max_n=70, min_p=0.2, max_p=0.3, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_p = min_p
        self.max_p = max_p
        super(COL_ER_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        num_vert = np.random.randint(self.min_n, self.max_n + 1)
        p = np.random.uniform(self.min_p, self.max_p)
        G = nx.erdos_renyi_graph(num_vert, p)
        while G.number_of_edges() <= 0:
            G = nx.erdos_renyi_graph(num_vert, p)
        return G


class COL_BA_Generator(COL_Generator_Base):

    def __init__(self, min_n=20, max_n=50, min_m=2, max_m=10, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        super(COL_BA_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        m = np.random.randint(self.min_m, self.max_m + 1)
        G = nx.barabasi_albert_graph(n, m)
        return G


class COL_REG_Generator(COL_Generator_Base):

    def __init__(self, min_n=50, max_n=50, min_d=3, max_d=20, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_d = min_d
        self.max_d = max_d
        super(COL_REG_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        d = np.random.randint(self.min_d, self.max_d + 1)
        G = nx.random_regular_graph(d, n)
        return G


class COL_GEO_Generator(COL_Generator_Base):

    def __init__(self, min_n=100, max_n=100, min_r=0.1, max_r=0.2, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_r = min_r
        self.max_r = max_r
        super(COL_GEO_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        r = np.random.uniform(self.min_r, self.max_r)
        G = nx.random_geometric_graph(n, r)
        while G.number_of_edges() <= 0:
            G = nx.random_geometric_graph(n, r)
        return G


class RB_Generator:

    def __init__(self, min_k=2, max_k=4, min_n=5, max_n=40):
        self.min_k = min_k
        self.max_k = max_k
        self.min_n = min_n
        self.max_n = max_n

    def create_random_instance(self):
        k = np.random.randint(self.min_k, self.max_k + 1)
        n = np.random.randint(self.min_n, self.max_n + 1)
        data = get_random_RB(k, n)
        return data


class MC_ER_Generator:

    def __init__(self, min_n, max_n, min_p, max_p, weighted_prob=0.5):
        self.min_n = min_n
        self.max_n = max_n
        self.min_p = min_p
        self.max_p = max_p
        self.weighted_prob = weighted_prob

    def create_random_instance(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        p = np.random.uniform(self.min_p, self.max_p)
        G = nx.erdos_renyi_graph(n, p)
        num_edge = G.number_of_edges()

        if np.random.random() < self.weighted_prob:
            edge_weights = np.random.choice([-1, 1], (num_edge,))
        else:
            edge_weights = np.ones((num_edge,), dtype=np.int64)

        data = nx_to_maxcut(G, edge_weights)
        return data


generator_dict = {
    'KSAT': KSAT_Generator,
    'PWKSAT': PWKSAT_Generator,
    'COL': COL_GNM_Generator,
    'COL_GNM': COL_GNM_Generator,
    'COL_BA': COL_BA_Generator,
    'COL_ER': COL_ER_Generator,
    'COL_GEO': COL_GEO_Generator,
    'COL_REG': COL_REG_Generator,
    'RB': RB_Generator,
    'MC_ER': MC_ER_Generator,
}
