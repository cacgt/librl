import numpy as np
from rl.algorithms.algorithm import Algorithm
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValueBasedPolicyGradient as Oracle
from rl.core.function_approximators.policies import Policy
from rl.core.online_learners import base_algorithms as balg
from rl.core.online_learners import BasicOnlineOptimizer
from rl.core.online_learners.scheduler import PowerScheduler
from rl.core.utils.misc_utils import timed, zipsame
from rl.core.utils import logz
from rl.core.datasets import Dataset
from rl.core.utils.mvavg import PolMvAvg, ExpMvAvg


class GeneralizedPolicyGradient(Algorithm):
    """ Basic policy gradient method. """

    def __init__(self, policy, expert, expert_vfn, horizon,
                 lr=1e-3,
                 gamma=1.0, delta=None, lambd=0.9,
                 max_n_batches=1000,
                 n_warm_up_itrs=None,
                 n_pretrain_itrs=5,
                 sampling_rule='exponential', # define how random switching time is generated
                 cyclic_rate=2): # the rate of forward training, relative to the number of iterations
        assert isinstance(policy, Policy)
        self._policy = policy
        self.expert = expert
        self.expert_vfn = expert_vfn
        # create online learner
        x0 = self.policy.variable
        scheduler = PowerScheduler(lr)
        self.learner = BasicOnlineOptimizer(balg.Adam(x0, scheduler))
        # create oracle
        self.ae = ValueBasedAE(expert, expert_vfn,  # wrt expert
                               gamma=gamma, delta=delta, lambd=lambd,
                               use_is='one', max_n_batches=max_n_batches)
        self.oracle = Oracle(policy, self.ae)

        self._n_pretrain_itrs = n_pretrain_itrs
        if n_warm_up_itrs is None:
            n_warm_up_itrs = float('Inf')
        self._n_warm_up_itrs =n_warm_up_itrs
        self._itr = 0

        # for sampling random switching time
        if horizon is None: horizon=float('Inf')
        assert horizon<float('Inf') or gamma<1.
        self._horizon = horizon
        self._gamma = gamma  # discount factor of the original problem
        assert sampling_rule in ['exponential','cyclic','uniform']
        self._sampling_rule = sampling_rule
        self._cyclic_rate = cyclic_rate
        self._avg_n_steps = PolMvAvg(1,weight=1)  # the number of steps that the policy can survive so far
        self._reset_pi_ro()

    def _reset_pi_ro(self):
        # NOTE Should be called for each iteration
        self._sample=True  #  randomly sample a switching time step
        self._ro_with_policy = True  # in the phase of gradient rollout
        self._t_switch = []  # switching time
        self._scale = []  # extra scaling factor due to switching

    @property
    def policy(self):
        return self._policy

    def pi(self, ob, t, done):
        return self.policy(ob)

    # It alternates between two phases
    #   1) roll-in learner and roll-out expert for updating value function
    #   2) roll fully the learner for computing gradients
    #
    # NOTE The below implementation assume that `logp` is called "ONLY ONCE" at
    # the end of each rollout.

    def pi_ro(self, ob, t, done):
        if self._ro_with_policy:  # just run the learner
            return self.policy(ob)
        else:  # roll-in policy and roll-out expert
            if t==0:
                assert self._sample
                # At the begining of the two-phased rollouts
                # sample t_switch in [1, horizon]
                self._sample_t_switch() # update self._t_switch & self._scale
                self._sample=False

            if t<self._t_switch[-1]:  # roll-in
                return self.policy(ob)
            else:
                return self.expert(ob)

    def logp(self, obs, acs):
        if self._ro_with_policy:
            self._ro_with_policy = False
            return self.policy.logp(obs, acs)
        else: # roll-in policy and roll-out expert
            self._ro_with_policy = True  # do cv rollout the next time
            self._sample = True
            t_switch = self._t_switch[-1]
            logp0 = self.policy.logp(obs[:t_switch], acs[:t_switch])
            logp1 = self.expert.logp(obs[t_switch:], acs[t_switch:])
            return np.concatenate([logp0, logp1])

    def _sample_t_switch(self):
        # Define t_swich and p
        if self._sampling_rule=='cyclic':
            assert self._horizon < float('Inf')
            t_switch = (int(self._cyclic_rate*self._itr)%self._horizon)+1
            p = 1./self._horizon
        elif self._sampling_rule=='exponential':
            beta = self._avg_n_steps.val
            t_switch = int(np.ceil(np.random.exponential(beta)))
            p = 1./beta*np.exp(-t_switch/beta)
            print('exponential', t_switch, beta, p)
        else:
            if self._horizon < float('Inf'):
                p0 = self._gamma**np.arange(self._horizon)
                sump0 = np.sum(p0)
                p0 = p0/sump0
                ind = np.random.multinomial(1,p0)
                t_switch =  np.where(ind==1)[0][0]+1
                p = p0[t_switch-1]
            else:
                t_switch = np.random.geometric(p=1-self._gamma)[0]
                p = self._gamma**t_switch*(1-self._gamma)
        # correct for potential discount factor
        t_switch = min(t_switch, self._horizon)
        if self._horizon < float('Inf'):
            p0 = self._gamma**np.arange(self._horizon)
            sump0 = np.sum(p0)
            p0 = p0/sump0
            pp = p0[t_switch-1]
        else:
            sump0 = 1/(1-self._gamma)
            pp = self._gamma**t_switch*(1-self._gamma)
        scale = (pp/p)*sump0
        self._t_switch.append(t_switch)
        self._scale.append(scale)

    def pretrain(self, gen_ro):
        pi_exp = lambda ob, t, done: self.expert(ob)
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                ro = gen_ro(pi_exp, logp=self.expert.logp)
                self.oracle.update(ro, policy=self.policy, update_nor=False)
                self.policy.update(ro['obs_short'])
        self._reset_pi_ro()

    def update(self, ro):
        # Update input normalizer for whitening
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        # Mirror descent
        with timed('Update oracle'):
            # Split ro into two phases
            rollouts = ro.to_list() # even length
            ro_mix = rollouts[1:][::2]  # ro with random switch
            assert len(ro_mix)==len(self._t_switch) or len(ro_mix)==len(self._t_switch)-1
            # if a rollout too short, it is treated as zero
            ro_exp = []
            for r, t, s in zip(ro_mix, self._t_switch, self._scale):
                if len(r)>=t:
                    r = r[t:]
                    r.scale = s
                    ro_exp.append(r)
            ro_exp = Dataset(ro_exp)
            ro_pol = Dataset(rollouts[0:][::2])
            self.oracle.update(ro_pol, update_vfn=False, policy=self.policy)
            _, ev0, ev1 = self.oracle.update_vfn(ro_exp)

            # # for adaptive sampling
            self._avg_n_steps.update(np.mean([len(r) for r in ro_pol]))

        with timed('Compute policy gradient'):
            g = self.oracle.grad(self.policy)

        with timed('Policy update'):
            self.learner.update(g)
            self.policy.variable = self.learner.x

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        logz.log_tabular('std', np.mean(np.exp(2.*self.policy.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
        logz.log_tabular('ExplainVarianceBefore(AE)', ev0)
        logz.log_tabular('ExplainVarianceAfter(AE)', ev1)
        logz.log_tabular('NumberOfExpertRollouts', len(ro_exp))
        logz.log_tabular('NumberOfLearnerRollouts', len(ro_pol))

        # reset
        self._reset_pi_ro()
        self._itr+=1