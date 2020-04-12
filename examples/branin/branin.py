# Copyright (c) 2016, the MyGPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from MyGPyOpt.objective_examples.experiments2d import branin as branin_creator
import numpy as np

f = branin_creator()

def branin(x,y):
    return f.f(np.hstack([x,y]))
