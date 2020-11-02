import pandas as pd
import numpy as np

def compute_info_gain(samples, attr, target):

    split_ent = 0
    # Compute the attribute's value-based Entropy
    for key, value in samples[attr].value_counts(normalize=True).iteritems():
        index = samples[attr]== key
        sub_ent = compute_entropy(target[index])
        split_ent +=  sub_ent * value
   
    # Compute Overall entropy of the target variable in the entire set
    overall_ent = compute_entropy(target)

    # Compute the information gain of interested attibute with regards to the target
    return overall_ent - split_ent

def compute_entropy(y):
    """
    :param y: The data samples array of the target variable
    """
    if len(y) < 2: #  If the data sample is less than 2 rows, no reason to compute entropy
        return 0
    # Get the count of each value in the target array passed to this function
    # calculate the probability / distribition of each target value 0 , 1
    prob = np.array( y.value_counts(normalize=True) )
  
    return -(prob * np.log2(prob + 1e-6)).sum() 


class TreeNode:
      def __init__(self, node_name="", min_sample_num=10, default_decision=None):
        self.children = {}
        self.decision = None
        self.split_root_name = None
        self.name = node_name
        self.default_decision = default_decision
        self.min_sample_num = min_sample_num
     
      def pretty_print(self, prefix=''):
        if self.split_root_name is not None:
            for k, v in self.children.items():
                v.pretty_print(f"{prefix}:When {self.split_root_name} is {k}")
                #v.pretty_print(f"{prefix}:{k}:")
        else:
            print(f"{prefix}:{self.decision}")

      def predict(self, sample):
         if self.decision is not None:
           #print("Decision:", self.decision)
           return self.decision
         else:
            #print("self.split_root_name", self.split_root_name)
            attr_val = sample[self.split_root_name]
            child = self.children[attr_val]
            #print("Testing", self.split_root_name, "->", attr_val)
            
            return child.predict(sample)
     
      def fit(self, X, y):
          """
          The function accepts a training dataset, from which it builds the tree 
          structure to make decisions or to make children nodes (tree branches) 
          to do further inquiries
          :param X: [n * p] n observed data samples of p attributes
          :param y: [n] n samples of the target's values
          """
          if self.default_decision is None:
              self.default_decision = y.mode()[0]

          print(self.name, "received", len(X), "samples") 
          if len(X) < self.min_sample_num:   
                if len(X) == 0:
                    # The tree node is end, decision is reached
                    self.decision = self.default_decision
                    print("Decision", self.decision)
                else:
                    self.decision = y.mode()[0]
                    print("Decision", self.decision)
                return
          else:         
                unique_values = y.unique()
                # if the unique value is only 1 , no need to calculate Entropy
                if len(unique_values) == 1:
                    self.decision = unique_values[0]
                    print("Decision", self.decision)
                    return
                else:
                    info_gain_max = 0
                    for attribute in X.keys():
                          # Compute the information gain of each attribute with respect to the target
                          aig = compute_info_gain(X, attribute, y)
                          if aig > info_gain_max:
                              info_gain_max = aig
                              self.split_root_name = attribute
                    print(f"Split by {self.split_root_name}, IG: {info_gain_max:.2f}")
                   
                    self.children = {}
                    # Construct the sub tree of the attribute with highest IG
                    for v in X[self.split_root_name].unique():
                          # row by row judgement of the attribute if equal to q1 / q2 / q3 / q4 in the loop 
                          # and construct each bracnch of the attibute
                          index = X[self.split_root_name] == v                                  
                          self.children[v] = TreeNode(
                              node_name=self.name + ":" + self.split_root_name + "==" + str(v),
                                    min_sample_num=self.min_sample_num,
                                    default_decision=self.default_decision)
                           # Assign the rows of those equal to q1 to q4 into 4 branches of the sub-tree
                          self.children[v].fit(X[index], y[index])