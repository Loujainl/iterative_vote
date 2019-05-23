import gym
import itertools
from gym import spaces
from gym.utils import seeding
from collections import Counter
import random
import numpy as np
import enum


def find_majority(votes):

    """votes is 2-d array, (actions submitted) """

    bincnt =np.apply_along_axis(lambda x: np.bincount(x, minlength=2), 0, arr=votes)
    # ("number of zeros = ", bincnt[0])
    # ("number of ones = ", bincnt[1])
    q = bincnt.shape[1]
    # print(bincnt,q)
    return [1 if bincnt[1][i]>bincnt[0][i] else 0 for i in range(q)]

def create_profiles(voters_num, quest_num):

    """ (number of voters, number of questions) """

    # fix seed for repeatable profiles
    np.random.seed(voters_num)
            # create game
    agents = []
    for i in range(voters_num):
        agent = np.reshape(list(itertools.product([0, 1], repeat = quest_num)), (2**quest_num,quest_num))
        np.random.shuffle(agent)
        agents.append((agent))
    return agents

profiles = create_profiles(3,3)
print(profiles)
sincere_actions = [profiles[i][0] for i in range(len(profiles))]
def calculate_result(actions):

    """  (selectedaction) returns majority vote result  """
    # actions =np.array([profiles[0][0],profiles[1][0], profiles[2][0]])
    # act = np.array([profiles[0][0],profiles[1][0], profiles[2][0]])
    # print(actions, " << Actions")
    result = find_majority(actions)
    print("MV:" ,result)
    return result

calculate_result(sincere_actions)



class IterativeVote(gym.Env):
    voters_num= 3 #agents, learners
    quest_num = 3  #multiple proposals
    profile = create_profiles(voters_num, quest_num)

    def __init__(self, voters_num=3,  quest_num = 3, test=False):
        self.action_space = spaces.Discrete(2**quest_num)
        self.observation_space = spaces.Discrete(1)
        self.reward_range = (0,2**quest_num - 1)


    def _get_obs(self):
        # return
        return 1

    def reset(self):
        return self._get_obs()



    def step(self, actions):
        next_obs = calculate_result(actions)
        return next_obs
