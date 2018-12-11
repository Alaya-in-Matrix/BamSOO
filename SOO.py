import numpy as np

class TreeNode:
    def __init__(self):
        self.lb        = None
        self.ub        = None
        self.x         = None
        self.y         = None
        self.chileren  = None
        self.num_split = None
        pass

    def expand(self):
        pass

    def traverse(self):
        pass

    def kth_layer(self, k):
        pass

    def get_data(self):
        pass

    def print_tree(self):
        pass

    def depth(self):
        pass

class SOO:
    def __init__(self, f, conf):
        self.root = None
        pass
    
    def optimize(self):
        pass

    def _optimize_oneiter(self):
        vmax      = -1 * np.inf;
        depth     = self.root.depth()
        node_list = [self.root]
        for i in range(1, depth):
            node_list = self._next_layer_nodes(node_list)
            best_node = self._select_from_one_layer(node_list)
            to_eval   = self._decide_eval(best_node, vmax)
            if to_eval:
                node_list.expand()
        pass

    def _decide_eval(self, node, vmax):
        pass

    def _next_layer_nodes(self, node_list):
        pass

    def _select_from_one_layer(self, node_list):
        pass

    def _comparator_init(self, x, y):
        pass

    def _compare(self, x1, x2):
        pass
