# Copyright (c) 2016, the MyGPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from .anchor_points_generator import ObjectiveAnchorPointsGenerator, ThompsonSamplingAnchorPointsGenerator
from .optimizer import apply_optimizer, choose_optimizer

max_objective_anchor_points_logic = "max_objective"
thompson_sampling_anchor_points_logic = "thompson_sampling"
sobol_design_type = "sobol"
random_design_type = "random"


class AcquisitionOptimizer(object):
    """
    General class for acquisition optimizers defined in domains with mix of discrete, continuous, bandit variables

    :param space: design space class from MyGPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    """

    def __init__(self, space, optimizer='lbfgs', **kwargs):

        self.space = space
        self.optimizer_name = optimizer
        self.kwargs = kwargs
        self.spec_win_cnt = 0

        ### -- save extra options than can be passed to the optimizer
        if 'model' in self.kwargs:
            self.model = self.kwargs['model']

        ### -- get a logic for anchor points sampler over acquisition space, with max_objective_anchor_points_logic as default
        if 'type_anchor_points_logic' in self.kwargs:
            self.type_anchor_points_logic = self.kwargs['type_anchor_points_logic']
        else:
            self.type_anchor_points_logic = max_objective_anchor_points_logic

        ## -- Context handler: takes
        self.context_manager = ContextManager(space)

    def optimize(self, f=None, df=None, f_df=None, duplicate_manager=None, x_opt=None):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df

        ## --- Update the optimizer, in case context has beee passed.
        self.optimizer = choose_optimizer(self.optimizer_name, self.context_manager.noncontext_bounds)

        ## --- Selecting the anchor points and removing duplicates
        if self.type_anchor_points_logic == max_objective_anchor_points_logic:
            anchor_points_generator = ObjectiveAnchorPointsGenerator(self.space, random_design_type, f)
        elif self.type_anchor_points_logic == thompson_sampling_anchor_points_logic:
            anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, sobol_design_type, self.model)

        # -- Select the anchor points (with context)
        np_prec = 5
        anchor_points = anchor_points_generator.get(duplicate_manager=duplicate_manager,
                                                    context_manager=self.context_manager, x_opt=x_opt)
        # print('anc_pts', np.array2string(anchor_points, precision=np_prec, separator=', ', max_line_width=np.inf), sep='\n')

        #  --- Applying local optimizers at the anchor points and update bounds of the optimizer (according to the
        # context)
        optimized_points = [
            apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager,
                            context_manager=self.context_manager, space=self.space) for a in anchor_points]
        # print('x_opt_pts', np.array2string(np.vstack([o[0] for o in optimized_points]), precision=np_prec, separator=', ',
        #                                    max_line_width=np.inf), sep='\n')
        # print('fx_opt_pts',
        #       np.array2string(np.vstack([o[1] for o in optimized_points]), separator=', ', max_line_width=np.inf),
        #       sep='\n')
        min_idx = np.argmin(np.vstack([o[1] for o in optimized_points]).flatten())
        num_anchors = 5
        if min_idx >= num_anchors:
            self.spec_win_cnt += 1
            print('specified anchors win', self.spec_win_cnt)

        x_min, fx_min = optimized_points[min_idx]

        # x_min, fx_min = min([apply_optimizer(self.optimizer, a, f=f, df=None, f_df=f_df, duplicate_manager=duplicate_manager, context_manager=self.context_manager, space = self.space) for a in anchor_points], key=lambda t:t[1])

        return x_min, fx_min


class ContextManager(object):
    """
    class to handle the context variable in the optimizer
    :param space: design space class from MyGPyOpt.
    :param context: dictionary of variables and their contex values
    """

    def __init__(self, space, context=None):
        self.space = space
        self.all_index = list(range(space.model_dimensionality))
        self.all_index_obj = list(range(len(self.space.config_space_expanded)))
        self.context_index = []
        self.context_value = []
        self.context_index_obj = []
        self.nocontext_index_obj = self.all_index_obj
        self.noncontext_bounds = self.space.get_bounds()[:]
        self.noncontext_index = self.all_index[:]

        if context is not None:

            ## -- Update new context
            for context_variable in context.keys():
                variable = self.space.find_variable(context_variable)
                self.context_index += variable.index_in_model
                self.context_index_obj += variable.index_in_objective
                self.context_value += variable.objective_to_model(context[context_variable])

            ## --- Get bounds and index for non context
            self.noncontext_index = [idx for idx in self.all_index if idx not in self.context_index]
            self.noncontext_bounds = [self.noncontext_bounds[idx] for idx in self.noncontext_index]

            ## update non context index in objective
            self.nocontext_index_obj = [idx for idx in self.all_index_obj if idx not in self.context_index_obj]

    def _expand_vector(self, x):
        '''
        Takes a value x in the subspace of not fixed dimensions and expands it with the values of the fixed ones.
        :param x: input vector to be expanded by adding the context values
        '''
        x = np.atleast_2d(x)
        x_expanded = np.zeros((x.shape[0], self.space.model_dimensionality))
        x_expanded[:, np.array(self.noncontext_index).astype(int)] = x
        x_expanded[:, np.array(self.context_index).astype(int)] = self.context_value
        return x_expanded
