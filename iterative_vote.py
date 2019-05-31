import gym
import itertools
from gym import spaces
from gym.utils import seeding
from collections import Counter
import random
import numpy as np
import enum
# from ray.rllib.utils.annotations import PublicAPI


def find_majority(votes):


    """votes is 2-d array, (actions submitted) """

    bincnt = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), 0, arr=votes)
    # ("number of zeros = ", bincnt[0])
    # ("number of ones = ", bincnt[1])
    q = bincnt.shape[1]
    # print(bincnt,q)
    return [1 if bincnt[1][i] > bincnt[0][i] else 0 for i in range(q)]



class IterativeVote(gym.Env):


    voters_num = 3  # agents, learners
    quest_num = 3  # multiple proposals



    def create_profiles(self, voters_num, quest_num):


        """ (number of voters, number of questions) """

        # fix seed for repeatable profiles
        np.random.seed(voters_num)
                # create game
        agents = []
        for i in range(voters_num):
            agent = np.reshape(list(itertools.product([0, 1], repeat = quest_num)), (2**quest_num, quest_num))
            np.random.shuffle(agent)
            agents.append((agent))
        return agents

    def calculate_result(self, actions):

        """  (selectedaction) returns majority vote result  """
        result = find_majority(actions)
        print("MV:", result)
        return result


    def result_rank(self, vresult,agent_index):
        agent_prof = self.profiles[agent_index]
        print("agent profile:", agent_prof)
        rank = np.where(np.all(agent_prof ==vresult,axis=1))
        # rank = np.where(profile[agent_index]=result)
        return rank

    def __init__(self, voters_num=3,  quest_num = 3, test=False):
        self.voters_num = voters_num
        self.quest_num = quest_num
        self.action_space = spaces.Discrete(2**quest_num)
        self.observation_space = spaces.Discrete(1)
        # self.reward_range = (0,2**quest_num - 1)
        self.state = None
        self.profiles = self.create_profiles(voters_num, quest_num)

    def _get_obs(self):
        # return
        return 1

    def reset(self):
        self.state = 0
        return self._get_obs()

    def step(self, actions):

        next_obs = calculate_result(actions)
        return next_obs
    
    def get_reward(self, voter_index):
        # return self.result_rank()
        # linear reward w.r.t rank of result in agent[index] profile
        return 1




instance = IterativeVote(3,3)
profile = instance.create_profiles(3,3)
print(profile)
sincere_actions = [profile[i][0] for i in range(len(profile))]
res = instance.calculate_result(sincere_actions)
print("result is ", res)
rank = instance.result_rank(res,1)
print(" Result rank for agent index1 :", rank[0])
