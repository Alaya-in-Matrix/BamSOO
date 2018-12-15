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

    def expand(self, rand = False):
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
            self.children.append(TreeNode(lb, ub, self.num_split, rand))
    
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

