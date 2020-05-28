# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import tensorflow as tf
import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import Mamba
from rl.core.function_approximators.policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP


import os


def load_policy(path, name):
    policy = RobustKerasMLPGassian((1,), (1,), init_lstd=0, name='dummy')
    policy.restore(path, name=name)
    return policy

def load_vfn(path, name):
    vfn = SuperRobustKerasMLP((1,), (1,), name='dummy')
    vfn.restore(path, name=name)
    return vfn


def create_experts(envid, name, path=None):
    def load_expert(path, name):
        expert = load_policy(path, name)
        expert.name = 'expert_policy'
        return expert

    if path is None:
        path = os.path.join('experts',envid)

    for f in os.scandir(path):  # for a single expert
        if f.name.endswith(name):
            experts = [load_expert(path, name)]
            break
    else: # for a set of experts
        experts = []
        for d in os.scandir(path):
            experts.append(load_expert(os.path.join(path,d.name,'saved_policies'), name))
            experts[-1].name = 'expert_'+str(len(experts))
    return experts

def create_learner(envid, seed, policy0, vfn0):
    # Try to load the initial policy and value function
    policy_name = 'learner_policy_'+str(seed)
    vfn_name = 'learner_vfn_'+str(seed)
    policy=vfn=None

    path = os.path.join('init_learner',envid)
    if os.path.exists(path):
        for f in os.scandir(path):
            if f.name.endswith(policy_name):
                policy = load_policy(path, policy_name)
            if f.name.endswith(vfn_name):
                vfn = load_vfn(path, vfn_name)
    if policy is None:
        print("Cannot find existing initial policy. Create a new one.")
        policy = policy0
        policy.save(path, name=policy_name)
    if vfn is None:
        print("Cannot find existing initial vfn. Create a new one.")
        vfn = vfn0
        vfn.save(path, name=vfn_name)

    policy.name = 'learner_policy'
    vfn.name = 'learner_vfn'

    return policy, vfn


def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create mdp and fix randomness
    mdp = ps.setup_mdp(c['mdp'], c['seed'])

    # Create learnable objects
    ob_shape = mdp.ob_shape
    ac_shape = mdp.ac_shape
    if mdp.use_time_info:
        ob_shape = (np.prod(ob_shape)+1,)

    # Define the learner
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='learner_policy',
                                    init_lstd=c['init_lstd'],
                                    units=c['policy_units'])
    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='learner_vfn',
                                units=c['value_units'])
    policy, vfn = create_learner(c['mdp']['envid'], c['seed'], policy, vfn)

    # Define experts
    if c['use_experts']:
        experts = create_experts(c['mdp']['envid'],c['expert_name'], path=c['expert_path'])
        if c['n_experts'] is not None and len(experts)>c['n_experts']:
            experts = experts[:c['n_experts']]
        if len(experts)<1:
            experts = None
    else:
        experts=None

    # Create algorithm
    ro_by_n_samples = c['experimenter']['ro_kwargs'] is not None
    alg = Mamba(policy, vfn,
                experts=experts,
                horizon=mdp.horizon, gamma=mdp.gamma,
                mix_unroll_kwargs={'ro_by_n_samples':ro_by_n_samples},
                **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['ro_kwargs'])
    exp.run(**c['experimenter']['run_kwargs'])


CONFIG = {
    'top_log_dir': 'log_mamba',
    'exp_name': 'cp',
    'seed': 0,
    'mdp': {
        'envid': 'DartCartPole-v1',
        'horizon': 1000,  # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes': 1,
        'min_ro_per_process': 2,  # needs to be at least 2 so the experts will be rollout
        'max_run_calls':25,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'eval_freq': 1,
            'save_freq': None,
        },
        'ro_kwargs': {
            'min_n_samples': None,
            'max_n_rollouts': 8,
        },
    },
    'algorithm': {
        'optimizer':'adam',
        'lr':0.001,
        'max_kl':0.05,
        'delta':None,
        'lambd':0.5,
        'max_n_batches':2,
        'n_warm_up_itrs':None,
        'n_pretrain_itrs':2,
        # new kwargs
        'eps':1.0,
        'strategy':'max',
        'policy_as_expert': True,
        'max_n_batches_experts':100,
    },
    'policy_units': (128,128),
    'value_units': (256,256),
    'init_lstd': -1,
    #
    'use_experts':True,
    'expert_name':'policy_best',
    'expert_path': None, #os.path.join('experts','DartCartPole-v1','600','saved_policies'),
    'n_experts': 2, # None,
}

if __name__ == '__main__':
    main(CONFIG)
