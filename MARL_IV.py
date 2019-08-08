import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import time


start = time.time()

def find_majority(votes):
    """votes is 2-d array, (actions submitted) """
    bincnt = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), 0, arr=votes)
    q = bincnt.shape[1]
    return [1 if bincnt[1][i] > bincnt[0][i] else 0 for i in range(q)]


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
    return np.random.randint(mab._num_actions)


def ucb1(mab):
    return np.argmax(mab.bandit_q_values + np.sqrt(2 * np.log
    (mab.step_counter + 1) / (mab.bandit_counters + 1)))


def plot(dict):
    for strategy, value in dict.items():
        # total_regret = np.cumsum(regret)
        plt.ylabel('ASI')
        plt.xlabel('Iteration')
        plt.plot(np.arange(len(value)), value, label=strategy)
    plt.legend()
    plt.savefig('asi.png')
    plt.show()


class MAV:
    class Bandit:
        def __init__(self, action_index, q_value=0, counter=0):
            self.q_value = q_value
            self.counter = counter
            self.action_index = action_index  # active bandit
            self.step_reward = 0

        # now just update q_value
        def get_q_value(self):
            return self.q_value

        def update_q_value(self):
            self.q_value = self.q_value + 1.0 / self.counter * (self.step_reward - self.q_value)

        def get_counter(self):
            return self.counter

        def set_step_reward(self, step_reward):
            self.step_reward = step_reward

        def get_step_reward(self):
            return self.step_reward

    # each agent is a MAB
    def __init__(self, agent_index, num_quest, best_action):
        self.agent_index = agent_index
        self.num_quest = num_quest
        self.profile = self.create_profile()
        self._num_actions = len(self.profile)
        self.step_counter = 0  # total number of runs
        self.step_action = 0  # action index
        self.step_rank = 0
        self.best_action = best_action
        self.bandits = self.createBandits()


    def get_step_action(self):
        return self.step_action

    def create_profile(self):
        profile = np.reshape(list(itertools.product([0, 1], repeat=self.num_quest)),
                             (2 ** self.num_quest, self.num_quest))
        np.random.shuffle(profile)
       # print ("profile", profile)
        return profile

    def createBandits(self):
        bandits = []
        for i in range(self._num_actions):
            bandit = self.Bandit(i)
            bandits.append(bandit)
        return np.array(bandits)

    def selectBandit(self, bandit_index):
        return self.bandits[bandit_index]

    def run_one_round(self, exploration_strategy, **strategy_parameters):
        self.step_counter += 1
        self.step_action = exploration_strategy(self, **strategy_parameters)
        self.bandits[self.step_action].counter += 1
        return self.profile[self.step_action]

    def result_rank(self, result): # if result is ranked 0, reward is 8-1-0 = 7, named steprank is linear reward
        rank = np.where(np.all(self.profile == result, axis=1))
        #print ("profile is",self.profile, "selected action is ",self.step_action,"result is",result)
        #print ("rank0", rank[0])
        step_rank = self._num_actions - 1 - rank[0]
        self.step_rank = step_rank
        #print("returned result rank", step_rank)
        return self.step_rank

    def get_exp_reward(self, result, *type):
        rank = 2**self.num_quest - 1 - self.result_rank(result)
        # in case we want to edit linear to exponential reward
        reward = 1.0 / 2 ** rank[0]
        self.bandits[self.step_action].set_step_reward(reward)
        self.bandits[self.step_action].update_q_value()
        #print ("reward  exp 1/2^rank", reward, "linear is", 7 - rank)
        return reward

    def get_linear_reward(self, result, *type):
        rank = self.result_rank(result)
        # in case we want to edit linear to exponential reward
        reward = rank[0]
        self.bandits[self.step_action].set_step_reward(reward)
        self.bandits[self.step_action].update_q_value()
        #print ("the linear reward is 7 - rank", self.bandits[self.step_action].get_step_reward())
        return reward

    @property
    def bandit_counters(self):
        return np.array([bandit.get_counter() for bandit in self.bandits])

    @property
    def bandit_q_values(self):
        return np.array([bandit.get_q_value() for bandit in self.bandits])

    @property
    def num_actions(self):
        return self._num_actions


class IterativeVote():

    def __init__(self, mav_num=3, quest_num=3, test=False):
        self.mav_num = mav_num
        self.quest_num = quest_num
        # Changed profiles inton one profile, one Agent
        # choose specific agent to access her action space
        self.selected_actions = []
        self.cached_state = None
        self.mabs = self.createGame()

    def createGame(self):
        #np.random.seed(500)
        # create number of MABS objects
        mavs = []
        for i in range(self.mav_num):
            agent = MAV(i, self.quest_num, 0)
            mavs.append((agent))
        return np.array(mavs)

    def get_selected_actions(self):
        actions = np.empty((self.mav_num, self.quest_num), dtype=int)
        for i in range(self.mav_num):
            actions[i, :] = self.mabs[i].profile[self.mabs[i].step_action]
        return actions

    def calculate_result(self, actions):

        """  (selectedaction) returns majority vote result  """
        result = find_majority(actions)
        self.cached_state = result
        return result

    def distribute_reward(self):
        pass

    def run_all_mavs(self, num_rounds, exploration_strategy, **strategy_parameters):
        #regrets = np.ndarray((self.mav_num, num_rounds))
        rewards = np.ndarray((self.mav_num, num_rounds))

        step_q_values = np.ndarray((self.mav_num, num_rounds))
        rank = np.ndarray((self.mav_num, num_rounds))
        asi = []
        for i in range(num_rounds):
            if (i + 1) % 100 == 0:
                print("\rEpisode {}/{}".format(i + 1, num_rounds))
                sys.stdout.flush()
            for j in range(self.mav_num):
                self.mabs[j].run_one_round(exploration_strategy, **strategy_parameters)
            #print("strategy and parameters", exploration_strategy, strategy_parameters)
            actions = self.get_selected_actions()
            result = self.calculate_result(actions)
            for r in range(self.mav_num):
                rewards[r, i] = self.mabs[r].get_linear_reward(result)
                rank[r, i] = self.mabs[r].step_rank
            step_ranks = rank.sum(axis=0)[i]
            #print (step_ranks)
            step_asi =  step_ranks / self.mav_num# ASI per iteration is the averaged sum of agents rank
            asi.append(step_asi)
            #print("step",i, "step asi",step_asi)
        return np.around(np.array(asi).astype(float),4), rewards


if __name__ == '__main__':
    def schedule(mab, epsilon):
        return epsilon - 1e-2 * mab.step_counter


    epsilon = 0.3

    strategies = {
        epsilon_greedy: {'epsilon': epsilon},
        decaying_epsilon_greedy: {'epsilon': epsilon, 'schedule': schedule},
        random: {},
        ucb1: {}
    }

    average_total_returns = {}
    asi_score = {}
    num_agents = 5
    num_quest = 5

    num_actions = 2 ** num_quest
    num_iterations = 500
    num_profiles = 500
    profiles = np.ndarray(num_profiles)
    asi_all_profiles = np.ndarray((num_profiles,num_iterations))
    swu_all_profiles = np.ndarray((num_agents,num_iterations,num_profiles))

    #a.mean(axis=1)     # to take the mean of each row

    for strategy, parameters in strategies.items():
        for profile in range(num_profiles):
            instance = IterativeVote(num_agents, num_quest)  # creategame on init defines the MABS access each by:  self.mabs[index]
            #print (instance.mabs.profile)
            np.append(profiles,instance)
            print(strategy.__name__)
            asi, average_total_return = instance.run_all_mavs(num_iterations, strategy, **parameters)
            print("\n")
            #print ("asi size", asi.size)
           # print ("asi for profile",profile,"is",asi)
            asi_all_profiles[profile,:] = asi

            #print ("asi for profile",asi_all_profiles[profile,:])
            np.dstack((swu_all_profiles,average_total_return[:,:,None]))
        #print ("type of asi all profiles", asi_all_profiles.dtype, "size", asi_all_profiles.size, "shape",
         #      asi_all_profiles.shape,"values",asi_all_profiles)
        average_total_returns[strategy.__name__] = swu_all_profiles.mean(axis=1)
        asi_score[strategy.__name__] = asi_all_profiles.mean(axis=0)

        #print(profiles)
        #print("asi all profiles axis 0",asi_all_profiles[0, :])
        #print ("mean is ", asi_all_profiles.mean(axis=0),"shape", asi_all_profiles.mean(axis=0).shape, "type",  asi_all_profiles.mean(axis=0).dtype)
        #print (swu_all_profiles)


    for strategy, swu in asi_score.items(): # chqnge between asi_score and average_total_return
         # total_regret = np.cumsum(regret)
         plt.ylabel('ASI')
         plt.xlabel('Iteration')
         plt.plot(np.arange(len(swu)), swu, label=strategy)
         plt.legend()
         plt.savefig('asi.png')
    plt.show()

    end = time.time()
    print("runtime", end - start)


    #
    # for strategy, asi in asi_score.items(): # chqnge between asi_score and average_total_returns
    #     # total_regret = np.cumsum(regret)
    #     plt.ylabel('ASI')
    #     plt.xlabel('Iteration')
    #     plt.plot(np.arange(len(asi)),  np.cumsum(asi), label=strategy)
    #     plt.legend()
    #     plt.savefig('asi.png')
    # plt.show()




