import sys, os
import numpy as np
import math

# TODO: Use anytree to represent the tree
class TreeNode:
    def __init__(self, lb, ub, num_split, rand = False):
        self.lb = lb
        self.ub = ub
        self.x         = 0.5 * (lb + ub) if not rand else np.random.uniform(lb, ub)
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
        len = (self.ub[id] - self.lb[id]) / self.num_split
        for i in range(self.num_split):
            lb     = self.lb.copy()
            ub     = self.ub.copy()
            lb[id] = self.lb[id] + len * i;
            ub[id] = self.lb[id] + len * (i+1);
            self.children.append(TreeNode(lb, ub, self.num_split))
    
    def children_leaves(self):
        return list(filter(lambda c : c.is_leaf(), self.children))

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
        self.func = f
        self.conf = conf
        try:
            self.lb = np.array(conf['lb'])
            self.ub = np.array(conf['ub'])
            for i in range(len(self.lb)):
                if(self.lb[i] >= self.ub[i]):
                    print("LB[%d] >= UB[%d]" % (i, i))
                    sys.exit(1)
        except:
            print("Please provide bound in configure file");
            sys.exit(1)

        self.debug        = conf.get('debug', False)
        self.dim          = len(self.lb)
        self.num_spec     = 1
        self.max_eval     = conf.get('max_eval', self.dim * 20)
        self.eval_counter = 0;
        self.best_x       = np.ones(self.dim) * np.nan
        self.best_y       = np.inf
        self.dbx          = np.zeros((0, self.dim))
        self.dby          = np.array([])
        self.b            = self.lb; # transform from [0, 50] to [lb, ub]
        self.a            = (self.ub - self.lb) / 50;
        self.num_split    = conf.get('num_split', 2)
        self.root         = TreeNode(np.zeros(self.dim), 50 * np.ones(self.dim), self.num_split)
        self.root.y       = self._eval_f(np.random.uniform(0, 50, self.dim))
        pass
    
    def _scale_x(self, x):
        """ 
        Scale from [0, 50] to [lb, ub]
        """
        return self.a * x + self.b

    
    def optimize(self):
        while self.eval_counter < self.max_eval:
            self._optimize_oneiter()
        return self.best_x, self.best_y

    def _optimize_oneiter(self):
        vmax      = self._init_vmax();
        depth     = self.root.depth()
        node_list = [self.root]
        for i in range(min(self._h(), depth)):
            best_node = self._select_from_one_layer(node_list)
            to_expand = self._decide_expand(best_node, vmax)
            if to_expand:
                vmax = best_node.y
                best_node.expand()
                for c in best_node.children:
                    self._set_node_value(c)
            node_list = self._next_layer_nodes(node_list)
    
    def _h(self):
        return 1 + self.eval_counter

    def _next_layer_nodes(self, node_list):
        xs = []
        for node in node_list:
            if node is not None:
                xs += node.children
        return xs

    #XXX: The methods below should be modified by classes inheriting SOO

    def _eval_f(self, x):
        """
        Input: 1D tensor,  length = dim
        Output: Scalar value
        """
        if self.debug:
            assert(x.ndim == 1)
            assert(x.size == self.dim)
        evaled   = self.func(self._scale_x(x))
        evaled   = evaled.reshape(evaled.size)
        assert evaled.size == self.num_spec
        self.dbx = np.concatenate((self.dbx, np.array([x])))
        self.dby = np.append(self.dby, evaled)
        self._comparator_init(self.dbx, self.dby)
        if self._compare(evaled, self.best_y):
            self.best_x = self._scale_x(x)
            self.best_y = evaled
        print("Best: %g" % self.best_y)
        self.eval_counter += 1
        return evaled

    def _init_vmax(self):
        return np.inf

    def _decide_expand(self, node, vmax):
        return (node is not None) and (node.y < vmax)
    
    def _set_node_value(self, node):
        node.y = self._eval_f(node.x)

    def _select_from_one_layer(self, node_list):
        leaves = list(filter(lambda n : n.is_leaf(), node_list))
        if not leaves:
            return None
        else:
            best_node = leaves[0]
            for n in leaves:
                if(n.y < best_node.y):
                    best_node = n
            return best_node

    def _comparator_init(self, x, y):
        pass

    def _compare(self, y1, y2):
        return np.all(y1 < y2)
