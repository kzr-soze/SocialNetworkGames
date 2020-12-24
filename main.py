import numpy as np
import numpy.random as rand
import sys
import networkx as nx
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.stats import pareto
# import plotly.graph_objects as go

from networkGame import SNGame

num_trials = 1000
num_rounds = 1998
zkc = False
dem2 = True
epoch = 100

is_epoch = True
trial_num= 100
graph_num = 'zkc'

def demo(game_type='collab'):
    n = 7
    k = 2
    neighbors ={0:[1,4,6],
                    1:[2,3,4],
                    2:[1,4,6],
                    3:[1,4,5,6],
                    4:[1,2,3],
                    5:[3,6],
                    6:[0,2,3,5]}
    network = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j in neighbors[i]:
                network[i,j] = 1
                network[j,i] = 1

    neighbors = {}
    # network = np.zeros([n,n])
    fixed_deltas = []

    # All leader and follower payoff distributions are iid exp(4)
    lamb= 4
    dist = lambda: rand.exponential(lamb,[2,2])
    f_array = []
    for i in range(n):
        entry = []
        for j in range(n):
            entry.append(dist)
        f_array.append (entry)

    deltas = np.zeros(n)

    # Epoch updates
    if is_epoch:
        delta_set = np.zeros([int(num_rounds/epoch) +2,n])
        delta_index = 1
    else:
        delta_set = np.zeros([num_rounds+2,n])

    # fixed_deltas = []
    # for i in fixed_deltas:
    #     deltas[i] = 30.0
    known_deltas=True
    delta_set[0] = deltas
    m = 91
    game = SNGame(n,k,network,f_array,f_array,fixed_deltas=fixed_deltas,deltas=deltas,game_type=game_type,known_deltas=known_deltas,m=m)
    print("Successfully initialized")
    print("Playing first round ")
    game.play_round()
    game.print_game()
    game.update_collab()
    print(np.mean(game.total_values))
    if is_epoch:
        delta_index = 1
    else:
        delta_set[1] = game.deltas
        delta_index=2


    for i in range(num_rounds):
        game.play_round()
        game.update_collab()

        if is_epoch and game.rounds % epoch == epoch-1:
            delta_set[delta_index] = game.deltas
            delta_index += 1
            print(delta_set)
            print("Average value per round: ",game.total_values/(i+3))
            print("Average value per round per player: ",np.mean(game.total_values)/(i+2))
            print(delta_set)
            np.save('Outputs/demo_trial{}_{}_epoch.npy'.format(trial_num,graph_num),delta_set)
            np.savetxt('Outputs/demo_trial{}_{}_epoch.txt'.format(trial_num,graph_num), delta_set, delimiter=',')
        if not is_epoch:
            delta_set[delta_index] = game.deltas
            delta_index+=1
            np.save('Outputs/demo_trial{}_{}_prob.npy'.format(trial_num,graph_num),delta_set)
            np.savetxt('Outputs/demo_trial{}_{}_prob.txt'.format(trial_num,graph_num), delta_set, delimiter=',')
    game.print_game()
    print("Average value per round: ",game.total_values/(num_rounds+2))
    print("Average value per round per player: ",np.mean(game.total_values)/(num_rounds+1))
    print(delta_set)

if __name__ == "__main__":
    print("Beginning Demo")
    demo()
