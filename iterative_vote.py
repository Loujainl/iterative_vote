import gym
import itertools
from gym import spaces
from gym.utils import seeding
from collections import Counter
import random
import numpy as np
import enum

def find_majority(votes):
    #votes is 1-d array, per question
    bincnt =np.bincount(votes)
    print("number of zeros = ", bincnt[0])
    print("number of ones = ", bincnt[1])
    return 1 if bincnt[1]>bincnt[0] else 0


def create_profiles(voters_num, quest_num):
            # create agents
    agents = []
    for i in range(voters_num):
        # name = 'agent-' + str(i+1)
        agent = np.reshape(list(itertools.product([0, 1], repeat = quest_num)), (2**quest_num,quest_num))
        np.random.shuffle(agent)
        agents.append((agent))
    return agents




    class IterativeVote(gym.Env):
        voters_num= 3 #agents, learners
        quest_num = 3  #multiple proposals

        def __init__(self, voters_num=3,  quest_num = 3, test=False):
            self.action_space = spaces.Discrete(2**quest_num)
            self.observation_space = spaces.Discrete(1)
            self.reward_range = (0,2**quest_num - 1)

            if test:
              self._seed(voters_num)
            else:
              self._seed()

            # Start the first game
            self.reset()

        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]


        #fix seed to get the same result everytime
        np.random.seed(voters_num)
        # generate the permutations
        profile1 = np.reshape(list(itertools.product([0, 1], repeat = quest_num)), (2**quest_num,quest_num))
        np.random.shuffle(profile1)
        print("Profile one:",profile1)
        # [[1 0 1]
        #  [1 1 1]
        #  [1 0 0]
        #  [1 1 0]
        #  [0 1 1]
        #  [0 0 1]
        #  [0 0 0]
        #  [0 1 0]]
        profile2 = np.reshape(list(itertools.product([0, 1], repeat = quest_num)), (2**quest_num,quest_num))
        np.random.shuffle(profile2)
        print("Profile two:",profile2)

        profile3 = np.reshape(list(itertools.product([0, 1], repeat = quest_num)), (2**quest_num,quest_num))
        np.random.shuffle(profile3)
        print("Profile three:",profile3)


        def _get_obs(self):
            # return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))
            return 1

        def reset(self):
            # self.dealer = draw_hand(self.np_random)
            # self.player = draw_hand(self.np_random)
            return self._get_obs()


        # class Actions(enum.IntEnum):
        #     """ Actions for the player """
        #     SINCERE = 0
        #     LEARN = 1
        #
        # class Action_Mode(enum.IntEnum):
        #     INITIAL = 0
        #     TRAIN = 1



        # collectivote = np.concatenate(profile1[1], profile2[1], profile3[1])
        # print("Collective vote",collectivote)

x = create_profiles(3,3)
print(x[0][0],x[1][0], x[2][0] )
per_question = [x[0][0][0], x[1][0][0], x[2][0][0] ]
print("first question", per_question)
print(find_majority(per_question))
