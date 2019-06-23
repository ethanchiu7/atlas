#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import math

class FeatureEngineering:
    def __init__(self):
        pass
    
    def fixed_len_value(self, tensor_dict, key):
        return tensor_dict[key][0][0]
    
    def map_fn(self, tensor_dict):
        pass
    
    
    def filter_item_type(self, tensor_dict):
        item_type = self.fixed_len_value(tensor_dict, 'item_type')
        item_type = int(item_type)
        if item_type != 0 and item_type != 201 and item_type != 208:
            return False
        return True
    
    def filter_fn(self, tensor_dict):
        fns = [
            self.filter_item_type
        ]
        for fn in fns:
            if not fn(tensor_dict):
                return False
        return True
    
    def get_tensor_dict(self, reader):
        print(reader)
        return reader.filter(self.filter_fn).map(self.map_fn)

def test():
    fe = FeatureEngineering()
    tensor_dict = {
        'app': np.array([['browser_iflow']]),
        'user_lt_top_cate': np.array([['-1']]),
        'item_type': np.array([['502']]),
    }
    print (fe.fixed_len_value(tensor_dict, 'app'))

if __name__ == '__main__':
    test()
