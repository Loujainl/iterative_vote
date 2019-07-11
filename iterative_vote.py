import gym
import itertools
from gym import spaces
from gym.utils import seeding
from collections import Counter
import random
import numpy as np
import enum
import matplotlib.pyplot as plt
import sys


def find_majority(votes):


    """votes is 2-d array, (actions submitted) """

    bincnt = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), 0, arr=votes)
    q = bincnt.shape[1]
    return [1 if bincnt[1][i] > bincnt[0][i] else 0 for i in range(q)]

# mab is a multi-armed bandit, implementing different exploration strategies

def epsilon_greedy(mab, epsilon):
  rand = np.random.uniform()
  if rand < epsilon:
    return random(mab)
  else:
    return np.argmax(mab.bandit_q_values)

def decaying_epsilon_greedy(mab, epsilon, schedule):
  epsilon = schedule(mab, epsilon)
  return epsilon_greedy(mab, epsilon)

def random(mab):
  return np.random.randint(mab._no_actions)

def ucb1(mab):
  return np.argmax(mab.bandit_q_values + np.sqrt(2*np.log
    (mab.step_counter+1)/(mab.bandit_counters+1)))



class IterativeVote(gym.Env):


    def __init__(self, voters_num=3,  quest_num = 3, test=False):
        self.voters_num = voters_num
        self.quest_num = quest_num
        self.profiles = self.create_profiles()
        # choose specific agent to access her action space
        self.active_agent = None
       # self.action_space = spaces.Discrete(2**quest_num)
       # self.observation_space = spaces.Discrete(1)
        # self.reward_range = (0,2**quest_num - 1)

        self.cached_state = None


    def create_profiles(self):


        """ (number of voters, number of questions)
         fix seed for repeatable profiles """
        np.random.seed(self.voters_num)
                # create game
        agents = []
        for i in range(self.voters_num):
            agent = np.reshape(list(itertools.product([0, 1], repeat = self.quest_num)), (2**self.quest_num, self.quest_num))
            np.random.shuffle(agent)
            agents.append((agent))
        return np.array(agents)


    def select_agent(self, agent_index):
        assert agent_index < self.voters_num
        self.active_agent = self.profiles[agent_index]
        return self.active_agent



    #change mehtods with respect to self.active_agent attribute
    def select_action(self, action_index):
        return self.active_agent[action_index]

    def vote_sincere(self):
        return self.select_action(0)


    def calculate_result(self, actions):

        """  (selectedaction) returns majority vote result  """
        result = find_majority(actions)
        self.cached_state = result
       # print("MV:", result)
        return result

    # def calculate_result(self, ):


    def result_rank(self, agent_index):
        agent_prof = self.profiles[agent_index]
        rank = np.where(np.all(agent_prof ==self.cached_state,axis=1))
        return rank[0]+1


    def get_reward(self ,agent_index):
        # return self.result_rank()
        # linear reward w.r.t rank of result in agent[index] profile
        rank = self.result_rank(agent_index)
        print(rank)
        # in case we want to edit linear to exponential reward
        reward = 1/ 2**rank[0]
        return reward


    def _get_obs(self):
        # return
        return self.cached_state

    def reset(self):
        self.cached_state = 0
        return self._get_obs()

    def step(self, actions):

        next_obs = calculate_result(actions)
        return next_obs


class Bandit:
  def __init__(self,agent_index , action_index, bias, q_value=0, counter=0):
    self.bias = bias
    self.q_value = q_value
    self.counter = counter
    self.action_index = action_index
    self.agent_index = agent_index


  def pull(self):
    iterv = IterativeVote(0,3,3)
    self.counter += 1
    # redifine the reward
    act = iterv.select_action(self.agent_index, self.action_index)
    reward = np.clip(self.bias + np.random.uniform(), 0, 1)
    self.q_value = self.q_value + 1/self.counter * (reward - self.q_value)
    return reward


class MAB:
    # each agent is a MAB
  def __init__(self, agent_index, best_action, *bandits):
    agent = IterativeVote(agent_index)
    #self.bandits = bandits
    self.bandits = agent.select_agent(agent_index)
    self._no_actions = len(bandits)
    self.step_counter = 0
    self.best_action = best_action

  def pull(self, action):
    self.step_counter += 1
    return self.bandits[action].pull(), self.bandits[action].q_value

  def run(self, no_rounds, exploration_strategy, **strategy_parameters):
    regrets = []
    rewards = []
    for i in range(no_rounds):
      if (i + 1) % 100 == 0:
        print("\rEpisode {}/{}".format(i + 1, no_rounds), end="")
        sys.stdout.flush()
      action = exploration_strategy(self, **strategy_parameters)
      reward, q = self.pull(action)
      best_action_value = self.best_action(self)[1]
      regret = best_action_value - q
      regrets.append(regret)
      rewards.append(reward)
    return regrets, rewards

  @property
  def bandit_counters(self):
    return np.array([bandit.counter for bandit in self.bandits])

  @property
  def bandit_q_values(self):
    return np.array([bandit.q_value for bandit in self.bandits])

  @property
  def no_actions(self):
    return self._no_actions


# class Agent(MAB):
#     def __init__(self, best_action, *bandits):
#         super(Agent,self).__init__(self, best_action, *bandits)
#


instance = IterativeVote(3,3)
profile = instance.create_profiles()
print("Profile index 1", instance.select_agent(1), "type of iterativeVote profile is ", type(instance.profiles))
# sincere_actions = [profile[i][0] for i in range(len(profile))]
length = len(instance.profiles)
print(length)
sincere_actions = []
strategic_actions = []
for i in range(length):
    instance.select_agent(i)
    action = instance.vote_sincere()
    sincere_actions.append(action)
    # print("selected 2nd action for all agents:")
    new_action =instance.select_action(1)
    strategic_actions.append(new_action)


#sincere_actions = [instance.vote_sincere(i) for i in range(length)]

res = instance.calculate_result(sincere_actions)
print("result is ", res)
rank = instance.result_rank(1)
print(" Result rank for agents:")
# print(rank[i] for i in range(rank.shape[0]))
reward = instance.get_reward(1)
print("reward for agent index 1 is 1/2^rank", reward)



resu = instance.calculate_result(strategic_actions)
print("choosing second action result", resu)

rewards = instance.get_reward(1)
print("new reward for agent index 1 is",rewards)
