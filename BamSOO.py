from SOO import TreeNode, SOO
from GP import GP
import numpy as np
import sys

class BamSOO(SOO):
    def __init__(self, f, conf):
        # SOO.__init__(self, f, conf)
        super(BamSOO, self).__init__(f, conf)
        self.new_eval        = 0
        self.rand_init       = conf.get('rand_init', 2)
        self._eta            = conf.get('eta', 0.5)
        self.node_evaluation = 0 # N in BamSOO paper
        self._init_GP()

    def _init_GP(self):
        init_x = np.random.uniform(0, 50, (self.rand_init, self.dim))
        for x in init_x:
            self._eval_f(x)
            self.new_eval += 1
        self.gp        = GP(self.dbx, self.dby)
    
    def _train_GP(self):
        if self.new_eval > 0:
            self.gp.update_db(self.dbx, self.dby)
            self.gp.train()
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
                for c in best_node.children:
                    self.node_evaluation += 1
                    self._set_node_value(c)
                vmax = best_node.y
                self.node_expansion += 1
            node_list = self._next_layer_nodes(node_list)
        print("After %d iter, beta = %g, depth: %d, search depth: %d, evaluated: %d, best: %g" % (self.iter_counter, self._beta(), depth, search_depth, self.eval_counter, self.best_y))

    def _set_node_value(self, node):
        py, ps2       = self.gp.predict(node.x)
        lcb, ucb      = self._cb(py, ps2)
        if(self._compare(lcb, self.best_y)):
            node.y        = self._eval_f(node.x)
            self.new_eval += 1
        else:
            node.y = ucb
