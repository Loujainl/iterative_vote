import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import sys


def find_majority(votes):
    """votes is 2-d array, (actions submitted) """
    # print("votes are", votes)
    bincnt = np.apply_along_axis(lambda x: np.bincount(x, minlength=2), 0, arr=votes)
    q = bincnt.shape[1]
    return [1 if bincnt[1][i] > bincnt[0][i] else 0 for i in range(q)]


# mab is the multi-armed bandit, implementing different exploration strategies

def epsilon_greedy(mab, epsilon):
    rand = np.random.uniform()
    if rand < epsilon:
        return random(mab)
    else:
        print ("current mab bandit_a_values is, we take argmax", mab.bandit_q_value)
        return np.argmax(mab.bandit_q_values)


def decaying_epsilon_greedy(mab, epsilon, schedule):
    epsilon = schedule(mab, epsilon)
    return epsilon_greedy(mab, epsilon)


def random(mab):
    return np.random.randint(mab._num_actions)


def ucb1(mab):
    return np.argmax(mab.bandit_q_values + np.sqrt(2 * np.log
    (mab.step_counter + 1) / (mab.bandit_counters + 1)))


def plot(regrets):
    for strategy, regret in regrets.items():
        # total_regret = np.cumsum(regret)
        plt.ylabel('ASI')
        plt.xlabel('Iteration')
        plt.plot(np.arange(len(regret)), regret, label=strategy)
    plt.legend()
    plt.savefig('asi.png')


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
            self.q_value = self.q_value + 1 / self.counter * (self.step_reward - self.q_value)
            # print("q_value updated to", self.q_value)

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

    def create_profile(self):
        profile = np.reshape(list(itertools.product([0, 1], repeat=self.num_quest)),
                             (2 ** self.num_quest, self.num_quest))
        np.random.shuffle(profile)
        return profile

    def createBandits(self):
        bandits = []
        for i in range(self._num_actions):
            bandit = self.Bandit(i)
            bandits.append(bandit)
        return np.array(bandits)

    def run_one_round(self, exploration_strategy, **strategy_parameters):
        self.step_counter += 1
        self.step_action = exploration_strategy(self, **strategy_parameters)
        self.bandits[self.step_action].counter += 1
        # print("step_action", self.step_action)
        # print("bandit counter", self.bandits[self.step_action].counter)
        return self.profile[self.step_action]

    def result_rank(self, result):
        # print("result is ", result)
        rank = np.where(np.all(self.profile == result, axis=1))
        # rank = np.where(np.all(agent_prof ==self.cached_state,axis=1))
        step_rank = self._num_actions - 1 - rank[0]
        self.step_rank = step_rank
        # print("the rank is ", rank[0])
        return self.step_rank

    def get_reward(self, result, *type):
        rank = self.result_rank(result)
        # in case we want to edit linear to exponential reward
        reward = 1 / 2 ** rank[0]
        # print("rank for agent", rank[0], "reward for agent ", self.agent_index, "is", reward)
        self.bandits[self.step_action].step_reward = reward
        self.bandits[self.step_action].update_q_value()
        return reward

    @property
    def bandit_counters(self):
        return np.array([bandit.counter for bandit in self.bandits])

    @property
    def bandit_q_values(self):
        return np.array([bandit.q_value for bandit in self.bandits])

    @property
    def num_actions(self):
        return self._num_actions


class IterativeVote():

    def __init__(self, mav_num=3, quest_num=3, test=False):
        self.mav_num = mav_num
        self.quest_num = quest_num
        # Changed profiles inton one profile, one Agent
        # choose specific agent to access her action space
        # self.active_agent = None
        self.selected_actions = []
        self.cached_state = None
        self.mabs = self.createGame()

    def createGame(self):
        np.random.seed(self.mav_num)
        # create number of MABS objects
        mavs = []
        for i in range(self.mav_num):
            agent = MAV(i, self.quest_num, 0)
            # np.random.shuffle(agent)
            mavs.append((agent))
        # print(mavs)

        return np.array(mavs)

    def get_selected_actions(self):
        actions = np.empty((self.mav_num, self.quest_num), dtype=int)
        # print("actions shape is ", actions.shape)
        for i in range(self.mav_num):
            actions[i, :] = self.mabs[i].profile[self.mabs[i].step_action]
            # actions[i] = MAV(i,self.quest_num,0).step_action
            # print("mab",i,"step_action is", self.mabs[i].profile[self.mabs[i].step_action])
            # actions.append(action)
        # self.selected_actions = np.array(actions)
        # print("get_selected_actions return", actions)
        return actions

    def calculate_result(self, actions):

        """  (selectedaction) returns majority vote result  """
        result = find_majority(actions)
        self.cached_state = result
        print("actions are", actions, "result is", result)
        # print("MV:", result)
        return result

    def distribute_reward(self):
        pass

    def run_all_mavs(self, num_rounds, exploration_strategy, **strategy_parameters):
        regrets = np.ndarray((self.mav_num, num_rounds))
        rewards = np.ndarray((self.mav_num, num_rounds))

        step_q_values = np.ndarray((self.mav_num, num_rounds))
        rank = np.ndarray((self.mav_num, num_rounds))
        # best_action_value = np.ndarray((self.mav_num,num_rounds))
        asi = []
        for i in range(num_rounds):
            if (i + 1) % 100 == 0:
                print("\rEpisode {}/{}".format(i + 1, num_rounds))
                sys.stdout.flush()
            for j in range(self.mav_num):
                self.mabs[j].run_one_round(exploration_strategy, **strategy_parameters)
            print("strategy and parameters", exploration_strategy, strategy_parameters)
            actions = self.get_selected_actions()
            result = self.calculate_result(actions)
            for r in range(self.mav_num):
                # print("agent index", r, "reward is ", self.mabs[r].get_reward(result))
                rewards[r, i] = self.mabs[r].get_reward(result)
                rank[r, i] = self.mabs[r].step_rank
                print ("rank agent",r, "iteration",i,"is ",rank[r, i])
            #  step_q_values[r , i] = self.mabs[r].bandits.get_q_value
            step_ranks = rank.sum(axis=0)[i]
            print ("sumranks ", step_ranks)
            step_asi =  step_ranks / self.mav_num# ASI per iteration is the averaged sum of agents rank
            print ("step asi", step_asi)
            asi.append(step_asi)
            # best_action_value = self.mabs[r].best_action()
            # regret = best_action_value - step_q_values
            # regrets.append(regret)
           # print("asi step",i ,"is", step_asi)
        # rewards.append(reward)
        return asi, rewards


if __name__ == '__main__':
    def schedule(mab, epsilon):
        return epsilon - 1e-6 * mab.step_counter


    epsilon = 0.6

    strategies = {
        epsilon_greedy: {'epsilon': epsilon},
        decaying_epsilon_greedy: {'epsilon': epsilon, 'schedule': schedule},
        random: {},
        ucb1: {}
    }

    average_total_returns = {}
    asi_score = {}
    num_agents = 3
    num_quest = 3

    instance = IterativeVote(num_agents,
                             num_quest)  # creategame on init defines the MABS access each by:  self.mabs[index]
    # best_action_index = 0
    # best_action_value = 1 # fix this according to the reward defined
    print("profile of agent 0 is ", instance.mabs[0].profile)
    print("profile of agent 1 is ", instance.mabs[1].profile)
    print("profile of agent 2 is ", instance.mabs[2].profile)

    rewards = []
    regrets = []
    bandits = []
    votes = []
    mabs = []

    num_actions = 2 ** num_quest
    # print("number of actions", num_actions)

    # def best_action(mab):
    #  return best_action_index, best_action_value
    num_iterations = 100

    for strategy, parameters in strategies.items():
        print(strategy.__name__)
        if strategy == "epsilon_greedy":
            asi, average_total_return = instance.run_all_mavs(num_iterations, strategy, **parameters)
            print("\n")
            average_total_returns[strategy.__name__] = average_total_return
            asi_score[strategy.__name__] = asi
        else: continue

    for strategy, asi_s in asi_score.items():
        # total_regret = np.cumsum(regret)
        plt.ylabel('ASI')
        plt.xlabel('Iteration')
        plt.plot(np.arange(len(asi_s)), asi_s, label=strategy)
        plt.legend()
