import numpy as np
import numpy.random as rand
import sys

from networkGame import SNGame

trials = 100
num_rounds = 200

def demo1(game_type):
    print("\n\n\n\n\n\n\n")
    n = 7
    k = 2
    neighbors ={0:[1,4,6],
                    1:[2,3,4],
                    2:[1,4,6],
                    3:[1,4,5,6],
                    4:[0,1,2,3],
                    5:[3,6],
                    6:[0,2,3,5]}
    network = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if j in neighbors[i]:
                network[i,j] = 1
                network[j,i] = 1
    # All leader and follower payoff distributions are iid exp(4)
    lamb= 4.0
    dist = lambda: rand.exponential(lamb,[2,2])
    f_array = []
    for i in range(n):
        entry = []
        for j in range(n):
            entry.append(dist)
        f_array.append (entry)

    # deltas = rand.random(n)*3 # Uniform from [0,3)
    deltas = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2])*5
    delta_set = np.zeros([num_rounds+2,n])
    delta_set[0] = deltas
    print(delta_set[0:5])
    fixed_deltas = [0,1,2,3,4,5,6]
    known_deltas = False
    # fixed_deltas = []
    game = SNGame(n,k,network,f_array,f_array,fixed_deltas=fixed_deltas,deltas=deltas,game_type=game_type,known_deltas=known_deltas)
    print("Successfully initialized")
    print("Playing first round ")
    # game.play_round()
    # game.print_game()
    # game.update_collab()
    delta_set[1] = game.deltas
    # game.print_game()

    low_delta_35 = np.zeros(num_rounds)
    low_delta_53 = np.zeros(num_rounds)
    high_delta_35 = np.zeros(num_rounds)
    high_delta_53 = np.zeros(num_rounds)

    for i in range(num_rounds):
        game.play_round()
        low_delta_35[i] = game.perceived_deltas[3][5]['delta_min']
        low_delta_53[i] = game.perceived_deltas[5][3]['delta_min']
        high_delta_35[i] = game.perceived_deltas[3][5]['delta_max']
        high_delta_53[i] = game.perceived_deltas[5][3]['delta_max']
        # game.update_collab()
        print(i,game.perceived_deltas[5])
        delta_set[i+2] = game.deltas
    game.print_game()
    if not fixed_deltas:
        print(delta_set)
    np.save('demo1_trial1.npy',delta_set)
    np.savetxt('demo1_trial1.txt', delta_set, delimiter=',')
    return low_delta_35,low_delta_53,high_delta_35,high_delta_53

def demo2(game_type):
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
    # All leader and follower payoff distributions are iid exp(4)
    lamb= 2.0
    dist = lambda: rand.geometric(1/lamb,[2,2])
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
    fixed_deltas = [0,1,2,3,4,5,6]
    # fixed_deltas = []
    game = SNGame(n,k,network,f_array,f_array,fixed_deltas=fixed_deltas,deltas=deltas,game_type=game_type,known_deltas=False)
    print("Successfully initialized")
    print("Playing first round ")
    game.play_round()
    game.print_game()
    game.update_collab()
    delta_set[1] = game.deltas
    # game.print_game()

    for i in range(num_rounds):
        game.play_round()
        # game.update_collab()
        delta_set[i+2] = game.deltas
    game.print_game()
    print(delta_set)
    np.save('demo_trial2.npy',delta_set)
    np.savetxt('demo_trial2.txt', delta_set, delimiter=',')

if __name__ == "__main__":
    low35 = np.zeros(num_rounds)
    low53 = np.zeros(num_rounds)
    high35 = np.zeros(num_rounds)
    high53 = np.zeros(num_rounds)
    perc_deltas = np.zeros([num_rounds,4])
    for i in range(trials):
        print('Trial: ',i)
        a,b,c,d = demo1(sys.argv[1])
        # print(a,b,c,d)
        # print(perc_deltas)
        perc_deltas[:,0] += a
        perc_deltas[:,2] += b
        perc_deltas[:,1] +=c
        perc_deltas[:,3] += d
        # print(perc_deltas)
    perc_deltas /= trials
    print(perc_deltas)
    np.savetxt('demo_delta_estimate35.txt',perc_deltas,delimiter=',')
    # print(sys.argv[1])
