import gym
import itertools
from gym import spaces
from gym.utils import seeding
from collections import Counter
import random
import numpy as np
import enum



def find_majority(votes):

    """votes is 1-d array, per question"""
    bincnt =np.bincount(votes)
    print("number of zeros = ", bincnt[0])
    print("number of ones = ", bincnt[1])
    return 1 if bincnt[1]>bincnt[0] else 0


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
def calculate_result(actions):

    """  (selectedaction) returns majority vote result  """
    actions = [profiles[0][0],profiles[1][0], profiles[2][0]]
    print(actions)
    result = []
    for i in range(len(actions[0])):
        question = [actions[0][i], actions[1][i], actions[2][i] ]
        print("question",str(i+1), question)
        per_question = find_majority(question)
        result.append(per_question)
    print(result)
    return result

calculate_result(profiles)



class IterativeVote(gym.Env):
    voters_num= 3 #agents, learners
    quest_num = 3  #multiple proposals

    def __init__(self, voters_num=3,  quest_num = 3, test=False):
        self.action_space = spaces.Discrete(2**quest_num)
        self.observation_space = spaces.Discrete(1)
        self.reward_range = (0,2**quest_num - 1)


    def _get_obs(self):
        # return
        return 1

    def reset(self):
        return self._get_obs()



