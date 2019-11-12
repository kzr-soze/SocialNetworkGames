import numpy as np
import numpy.random as rand
import sys

from networkGame import SNGame

num_rounds = 1000

def demo(game_type):
    n = 7
    k = 2
    neighbors ={0:[1,4,6],
                    1:[2,3,4],
                    2:[1,4,5],
                    3:[1,4,5,6],
                    4:[1,2,3],
                    5:[3,6],
                    6:[2,3,5]}
    network = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j in neighbors[i]:
                network[i,j] = 1
                network[j,i] = 1
    # All leader and follower payoff distributions are iid exp(4)
    lamb= 2
    dist = lambda: rand.exponential(lamb,[2,2])
    f_array = []
    for i in range(n):
        entry = []
        for j in range(n):
            entry.append(dist)
        f_array.append (entry)

    # deltas = rand.random(n)*3 # Uniform from [0,3)
    deltas = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2])
    delta_set = np.zeros([num_rounds+2,n])
    delta_set[0] = deltas
    print(delta_set[0:5])

    game = SNGame(n,k,network,f_array,f_array,deltas=deltas,game_type=game_type)
    print("Successfully initialized")
    print("Playing first round ")
    game.play_round()
    game.print_game()
    game.update_collab()
    delta_set[1] = game.deltas
    # game.print_game()

    for i in range(num_rounds):
        game.play_round()
        game.update_collab()
        delta_set[i+2] = game.deltas
    game.print_game()
    np.save('demo_trial1.npy')

if __name__ == "__main__":
    demo(sys.argv[1])
    # print(sys.argv[1])
