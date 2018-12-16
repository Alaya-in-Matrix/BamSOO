import sys, os
import numpy as np
import math
from TreeNode import TreeNode

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

        self.debug          = conf.get('debug', False)
        self.dim            = len(self.lb)
        self.num_spec       = 1
        self.max_eval       = conf.get('max_eval', self.dim * 20)
        self.fmin           = conf.get('fmin', -1 * np.inf)
        self.best_x         = np.ones(self.dim) * np.nan
        self.best_y         = np.inf
        self.dbx            = np.zeros((0, self.dim))
        self.dby            = np.zeros((0, 1))
        self.b              = self.lb; # transform from [0, 10] to [lb, ub]
        self.a              = (self.ub - self.lb) / 10;
        self.num_split      = conf.get('num_split', 2)
        self.iter_counter   = 0 
        self.node_expansion = 0 # n in BamSOO paper
        self.eval_counter   = 0 # t in BaMSOO paper
        self.root           = TreeNode(np.zeros(self.dim), 10 * np.ones(self.dim), self.num_split)
        self.root.y         = self._eval_f(self.root.x)

    
    def _scale_x(self, x):
        """ 
        Scale from [0, 10] to [lb, ub]
        """
        return self.a * x + self.b

    
    def optimize(self):
        while self.eval_counter < self.max_eval and not self._compare(self.best_y, self.fmin):
            self._optimize_oneiter()
        return self.best_x, self.best_y

    def _optimize_oneiter(self):
        self.iter_counter += 1
        vmax      = self._init_vmax();
        depth     = self.root.depth()
        node_list = [self.root]
        for i in range(min(self._h(), depth)):
            best_node = self._select_from_one_layer(node_list)
            to_expand = self._decide_expand(best_node, vmax)
            if to_expand:
                best_node.expand()
                for child_id in range(best_node.num_split):
                    if best_node.num_split % 2 == 1 and i == math.ceil(best_node.num_split / 2):
                        best_node.children[child_id].y = best_node.y
                    else:
                        self._set_node_value(best_node.children[child_id])
                vmax = best_node.y
                self.node_expansion += 1
            node_list = self._next_layer_nodes(node_list)
        print("After %d iter, evaluated: %d, best: %g" % (self.iter_counter, self.eval_counter, self.best_y))
    
    def _h(self):
        return math.ceil(math.sqrt(self.max_eval))
        # return 1 + math.ceil(math.sqrt(self.node_expansion))

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
        evaled   = evaled.reshape(1, evaled.size)
        assert evaled.size == self.num_spec
        self.dbx = np.concatenate((self.dbx, np.array([self._scale_x(x)])))
        self.dby = np.concatenate((self.dby, evaled))
        self._comparator_init(self.dbx, self.dby)
        if self._compare(evaled, self.best_y):
            self.best_x = self._scale_x(x)
            self.best_y = evaled
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

