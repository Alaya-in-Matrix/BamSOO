import numpy as np

class TreeNode:
    def __init__(self):
        self.lb        = None
        self.ub        = None
        self.x         = None
        self.y         = None
        self.children  = None
        self.num_split = None
        pass

    def expand(self):
        pass

    def traverse(self):
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

    def eval_f(self, x):
        pass
    
    def optimize(self):
        while self.eval_counter < self.max_eval:
            self._optimize_oneiter()

    def _optimize_oneiter(self):
        vmax      = self._init_vmax();
        depth     = self.root.depth()
        node_list = [self.root]
        for i in range(1, depth):
            node_list = self._next_layer_nodes(node_list)
            best_node = self._select_from_one_layer(node_list)
            to_expand = self._decide_expand(best_node, vmax)
            if to_expand:
                best_node.expand()
                for c in best_node.children:
                    self._set_node_value(c)
        pass

    def _next_layer_nodes(self, node_list):
        xs = []
        for node in node_list:
            if node is not None:
                xs += node.children
        return xs

    def _init_vmax(self):
        pass

    def _decide_expand(self, node, vmax):
        pass
    
    def _set_node_value(self, node):
        pass

    def _select_from_one_layer(self, node_list):
        pass

    def _comparator_init(self, x, y):
        pass

    def _compare(self, x1, x2):
        pass
