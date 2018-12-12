import sys, os
import numpy as np

# TODO: Use anytree to represent the tree
class TreeNode:
    def __init__(self, lb, ub, num_split):
        self.lb        = lb
        self.ub        = ub
        self.x         = 0.5 * (lb + ub)
        self.y         = np.nan
        self.children  = []
        self.num_split = num_split
        if not np.all(self.lb < self.ub):
            print("Error in the bound of node");
            sys.exit(1)
        pass

    def is_leaf(self):
        return (not self.children)

    def expand(self):
        if not self.is_leaf():
            print("Can not expand node, the node is not a leaf")
            sys.exit(1)
        id  = np.argmax(self.ub - self.lb)
        len = self.ub[id] - self.lb[id]
        for i in range(self.num_split):
            lb     = self.lb.copy()
            ub     = self.ub.copy()
            lb[id] = self.lb[id] + len * i;
            ub[id] = self.lb[id] + len * (i+1);
            self.children.append(TreeNode(lb, ub, num_split))

    def depth(self):
        if self.is_leaf():
            return 1;
        else:
            max_sub_depth = 0;
            for c in self.children:
                max_sub_depth = max(max_sub_depth, c.depth())
            return 1 + max_sub_depth

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

    #XXX: The methods below should be modified by classes inheriting SOO
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
