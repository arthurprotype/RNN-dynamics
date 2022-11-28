#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example template for contributing new tasks."""

import numpy as np
import random
import neurogym as ngym
from neurogym import spaces

class MyTaskSim(ngym.TrialEnv):
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1, dim_ring=2, cohs=None, theta=0):
        super().__init__(dt=dt)
        # Possible decisions at the end of the trial
        self.choices = [0, 1]  # e.g. [anticlockwise, clockwise]
        self.cues = [1, 0]

        if cohs is None: # coherence
            self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) # difficulty
        else:
            self.cohs = cohs

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        # Optional rewards dictionary
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Optional timing dictionary
        # if provided, self.add_period can infer timing directly
        self.timing = {
            'sample1': 500,
            'delay1': 500,
            'sample2': 500,
            'delay2': 500,
            'cue': 1000,
            # 'probe': 500,
            'decision': 500 # 做长一点，轨迹的趋势，自治系统
        }
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2 * np.pi, dim_ring+1)[:-1] 
        # self.theta = np.linspace(25, 175, dim_ring) * np.pi/ 180 # dim_ring+1==3, 在线性空间中以均匀步长生成数字序列
        # [:-1]取原始数据的除最后一个元素之外的值, [::-1] 顺序相反操作

        # Similar to gym envs, define observations_space and action_space
        # Optional annotation of the observation space
        # name = {'fixation': 0, 'stimulus': range(1, dim_ring+1), 'cue': 3}
        name = {'fixation': 0, 
                'stimulus1': range(1, 1+dim_ring), 
                'stimulus2': range(1+dim_ring, 1+dim_ring*2), 
                'cue': range(1+dim_ring*2, 3+dim_ring*2)
                }
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3+dim_ring*2,), dtype=np.float32, name=name)
        # Optional annotation of the action space
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.Discrete(3, name=name)


    def _new_trial(self, **kwargs):
        """
        self._new_trial() is called internally to generate a next trial.
        Typically, you need to
            set trial: a dictionary of trial information
            run self.add_period():
                will add time periods to the trial
                accesible through dict self.start_t and self.end_t
            run self.add_ob():
                will add observation to np array self.ob
            run self.set_groundtruth():
                will set groundtruth to np array self.gt
        Returns:
            trial: dictionary of trial information
        """

        # Setting trial information
        cue1 = self.rng.choice(self.cues)
        cue2 = 1 - cue1
        # cue1 = self.rng.choice(self.cues)
        # if cue1 == 1:
        #     cue2 = cue3 = cue4 = 0
        # else:
        #     cue2 = self.rng.choice(self.cues)
        #     if cue2 == 1:
        #         cue3 = cue4 = 0
        #     else:
        #         cue3 = self.rng.choice(self.cues)
        #         cue4 = 1 - cue3
        
        # cue2 = 1 - cue1
        gt1 = self.rng.choice(self.choices)
        gt2 = self.rng.choice(self.choices)
        trial = {
            'cue1': cue1,
            'cue2': cue2,
            'coh': self.rng.choice(self.cohs),
        }
        trial.update(kwargs)  # allows wrappers to modify the trial

        coh = trial['coh']
        # stim1_theta = random.choice(self.theta)
        # stim2_theta = random.choice(self.theta) 
        stim1_theta = self.theta[gt1]
        stim2_theta = self.theta[gt2]

        difference_list = [3, 6, 9, 13, 18, 24, 30]

        # if cue1 == 1:
        #     gt = gt1
        #     stim_theta = stim1_theta
        # elif cue2 == 1:
        #     gt = gt2
        #     stim_theta = stim2_theta
        # elif cue3 == 1:
        #     gt = gt3
        #     stim_theta = stim3_theta
        # else:
        #     gt = gt4
        #     stim_theta = stim4_theta
        
        gt = gt1 if cue1 == 1 else gt2
        stim_theta = stim1_theta if cue1 == 1 else stim2_theta

        test_theta = stim_theta if gt == 1 else np.pi - stim_theta   
        # if gt == 1:
        #     test_theta = stim_theta
        # else:
        #     theta_tmp = np.delete(self.theta, int(stim_theta*180/(np.pi*25)-1))
        #     test_theta = random.choice(theta_tmp)

        difference = random.choice(difference_list) if gt == 1 else -random.choice(difference_list)
        difference_theta = difference * np.pi / 180
        # test_theta = stim_theta + difference_theta

        trial['ground_truth'] = gt  
        trial['ground_truth1'] = gt1  
        trial['ground_truth2'] = gt2 
        trial['stim1_theta'] = stim1_theta 
        trial['stim2_theta'] = stim2_theta
        trial['stim_theta'] = stim_theta
        trial['difference'] = difference 
        trial['test_theta'] = test_theta

        stim1 = np.cos(self.theta - stim1_theta) 
        stim2 = np.cos(self.theta - stim2_theta) 
        test = np.cos(self.theta - test_theta) 

        periods = ['sample1', 'delay1', 'sample2', 'delay2', 'cue', 'decision']
        self.add_period(periods)

        self.add_ob(1, period=['sample1', 'delay1', 'sample2', 'delay2', 'cue'], where='fixation') 
        self.add_ob(stim1, 'sample1', where='stimulus1')
        self.add_ob(stim2, 'sample2', where='stimulus2')
        self.add_ob([cue1,cue2], 'cue', where='cue')
        # self.add_ob(test, 'probe', where='stimulus1')
        # self.add_ob(test, 'probe', where='stimulus2')
        # if cue1 == 1:
        #     self.add_ob(test, 'probe', where='stimulus1')
        # else:
        #     self.add_ob(test, 'probe', where='stimulus2')
        self.add_randn(0, self.sigma, 'sample1', where='stimulus1')
        self.add_randn(0, self.sigma, 'sample2', where='stimulus2')
        # self.add_randn(0, self.sigma, 'probe', where='stimulus1') 
        # self.add_randn(0, self.sigma, 'probe', where='stimulus2') 
        self.set_groundtruth(gt, period='decision',where='choice') # 加噪音

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        reward = 0

        ob = self.ob_now
        gt = self.gt_now

        if self.in_period('decision'):
            if action != 0:
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
        else:
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    # Instantiate the task
    env = MyTaskSim()
    trial = env.new_trial()
    print('Trial info', trial)
    print('Trial observation shape', env.ob.shape)
    print('Trial action shape', env.gt.shape)
    env.reset()
    ob, reward, done, info = env.step(env.action_space.sample())
    print('Single time step observation shape', ob.shape)