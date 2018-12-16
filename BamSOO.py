from SOO import TreeNode, SOO
from GP import GP
from TreeNode import TreeNode
import numpy as np
import math
import sys

class BamSOO(SOO):
    def __init__(self, f, conf):
        # SOO.__init__(self, f, conf)
        super(BamSOO, self).__init__(f, conf)
        self.new_eval        = 0
        self.rand_init       = conf.get('rand_init', 2)
        self.train_gp        = conf.get('train_gp', False)
        self.eval_gap        = conf.get('eval_gap', np.inf)
        self.debug           = conf.get('debug', False)
        self._eta            = conf.get('eta', 0.5)
        self.node_evaluation = 0 # N in BamSOO paper
        self._init_GP()
        if self.eval_gap == 0: # reduce to SOO
            self.train_gp = False

    def _init_GP(self):
        init_x = np.random.uniform(0, 50, (self.rand_init, self.dim))
        for x in init_x:
            self._eval_f(x)
            self.new_eval += 1
        self.gp = GP(self.dbx, self.dby)
    
    def _train_GP(self):
        if self.new_eval > 0 and self.eval_gap > 0:
            assert(self.gp.train_x.shape[0] == self.dbx.shape[0])
            if self.train_gp:
                self.gp.train()
                if self.debug:
                    print(self.gp.m)
        self.new_eval = 0

    def _beta(self):
        """
        Beta for LCB/UCB calculation
        """
        return np.sqrt(2 * np.log(np.pi**2 * self.node_evaluation**2 / (6 * self._eta)))

    def _cb(self, py, ps2):
        beta = self._beta()
        lcb  = py - beta * np.sqrt(ps2)
        ucb  = py + beta * np.sqrt(ps2)
        return lcb, ucb
    
    def _optimize_oneiter(self):
        self.iter_counter += 1
        vmax               = self._init_vmax();
        node_list          = [self.root]
        depth              = self.root.depth()
        search_depth       = min(self._h(), depth)
        self._train_GP() # XXX: new eval 问题
        for i in range(search_depth):
            best_node = self._select_from_one_layer(node_list)
            to_expand = self._decide_expand(best_node, vmax)
            if to_expand:
                best_node.expand()
                for child_id in range(best_node.num_split):
                    if best_node.num_split % 2 == 1 and i == math.ceil(best_node.num_split / 2):
                        best_node.children[child_id].y = best_node.y
                    else:
                        self.node_evaluation += 1
                        self._set_node_value(best_node.children[child_id], i)
                vmax = best_node.y
                self.node_expansion += 1
            node_list = self._next_layer_nodes(node_list)
        print("After %d iter, beta = %g, depth: %d, search depth: %d, evaluated: %d, node_eval: %d, num_expand: %d, best: %g" % (
            self.iter_counter,
            self._beta(),
            depth,
            search_depth,
            self.eval_counter,
            self.node_evaluation, 
            self.node_expansion, 
            self.best_y))

    def _set_node_value(self, node, depth = -1):
        if self.eval_gap > 0:
            py, ps2       = self.gp.predict(node.x)
            lcb, ucb      = self._cb(py, ps2)
            if self._compare(lcb, self.best_y) or (self.node_evaluation - self.eval_counter >= self.eval_gap):
                node.y        = self._eval_f(node.x)
                self.gp.update_db(self.dbx, self.dby)
                self.new_eval += 1
                if self.debug:
                    print('Depth = %d, py = %g, ps = %g, lcb = %g, y = %g' % (depth, py, np.sqrt(ps2), lcb, node.y))
            else:
                node.y = ucb
        else: # reduce to SOO, no GP predictions
            node.y        = self._eval_f(node.x)
            self.new_eval += 1
