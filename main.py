import numpy as np
import numpy.random as rand
import sys
import networkx as nx
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.stats import pareto
# import plotly.graph_objects as go

from networkGame import SNGame

num_trials = 100
num_rounds = 9998
zkc = False
dem2 = True
epoch = 100

is_epoch = True
trial_num=100
graph_num = 'zkc'

def karate_club(deltas,rounds,k=2,lamb=3):
    game_type = "collab"
    print("\n\n\n\n\n\n\n")
    G = nx.karate_club_graph()
    n = 34
    neighbors = {}
    network = np.zeros([n,n])
    fixed_deltas = []
    for i in G:
        neighbors[i] = list(G[i].keys())
        network[i][neighbors[i]] =1
        fixed_deltas.append(i)
    lamb = 4
    dist = lambda: rand.exponential(lamb,[2,2])
    f_array = []
    for i in range(n):
        entry = []
        for j in range(n):
            entry.append(dist)
        f_array.append(entry)

    delta_set = np.zeros([rounds+2,n])
    delta_set[0] = deltas
    games_played = np.zeros([rounds+2,n])
    total_utility= np.zeros([rounds+2,n])

    known_deltas = False
    seed = 0
    rand.seed(seed=seed)
    game = SNGame(n,k,network,f_array,f_array,fixed_deltas=fixed_deltas,deltas=deltas,game_type=game_type,known_deltas=known_deltas)
    print("Successfully initialized")
    print("Playing first round ")
    delta_set[1] = game.deltas
    games_played[1][:] = game.games_played
    total_utility[1][:] = game.total_values
    for i in range(rounds):
        game.play_round()
        delta_set[i+2] = game.deltas
        games_played[i+2] = game.games_played
        total_utility[i+2] = game.total_values
        # print(i)

    game.print_game()
    if not fixed_deltas:
        print(delta_set)
    return games_played,total_utility,delta_set






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
    known_deltas = True
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

def demo2(game_type='collab'):
    # n = 7
    # k = 2
    # neighbors ={0:[1,4,6],
    #                 1:[2,3,4],
    #                 2:[1,4,6],
    #                 3:[1,4,5,6],
    #                 4:[1,2,3],
    #                 5:[3,6],
    #                 6:[0,2,3,5]}
    # network = np.zeros([n,n])
    # for i in range(n):
    #     for j in range(n):
    #         if j in neighbors[i]:
    #             network[i,j] = 1
    #             network[j,i] = 1


    G = nx.karate_club_graph()
    n = 34
    k=10

    # H = nx.read_edgelist("0.edges", nodetype=int)
    # H = nx.DiGraph(H)
    # n = len(H.nodes)
    # k = 3
    # i = 0
    # mapping = {}
    # for j in H.nodes:
    #     mapping[j] = i
    #     i+=1
    # G = nx.relabel_nodes(H,mapping)

    neighbors = {}
    network = np.zeros([n,n])
    fixed_deltas = []
    for i in G:
        neighbors[i] = list(G[i].keys())
        network[i][neighbors[i]] =1
        fixed_deltas.append(i)

    # All leader and follower payoff distributions are iid exp(4)
    lamb= 4
    dist = lambda: rand.exponential(lamb,[2,2])

    # scale = 1.5
    # alpha = 1.5
    # dist = lambda: pareto.rvs(alpha, size = [2,2])

    # mu, sigma = 1,1
    # dist = lambda: rand.normal(mu,sigma,[2,2])
    f_array = []
    for i in range(n):
        entry = []
        for j in range(n):
            entry.append(dist)
        f_array.append (entry)

    # deltas = rand.random(n)*3 # Uniform from [0,3)
    # deltas = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2])*4.0/1.2
    # deltas = np.array([1.2,1.0,0.8,0.6,0.4,0.2,0.0])*4.0/1.2
    # deltas = np.zeros(n)+2
    deltas = np.zeros(n)
    # deltas[n-1] = 0.01

    # Epoch updates
    if is_epoch:
        delta_set = np.zeros([int(num_rounds/epoch) +2,n])
        delta_index = 1
    else:
        delta_set = np.zeros([num_rounds+2,n])

    delta_set[0] = deltas
    # print(delta_set[0:5])
    # fixed_deltas = [0,1,2,3,4,5,6]
    fixed_deltas = []
    known_deltas=False
    game = SNGame(n,k,network,f_array,f_array,fixed_deltas=fixed_deltas,deltas=deltas,game_type=game_type,known_deltas=known_deltas)
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
        # game.print_game()
        delta_index=2


    for i in range(num_rounds):
        game.play_round()
        game.update_collab()
        # delta_set[i+2] = game.deltas

        if is_epoch and game.rounds % epoch == epoch-1:
            delta_set[delta_index] = game.deltas
            delta_index += 1
            print(delta_set)
            print("Average value per round: ",game.total_values/(num_rounds+2))
            print("Average value per round per player: ",np.mean(game.total_values)/(num_rounds+1))
            print(delta_set)
            np.save('Ouptputs/demo_trial{}_{}_epoch.npy'.format(trial_num,graph_num),delta_set)
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
    np.save('Outputs/demo_trial{}_{}_prob.npy'.format(trial_num,graph_num),delta_set)
    np.savetxt('Outputs/demo_trial{}_{}_prob.txt'.format(trial_num,graph_num), delta_set, delimiter=',')



def run_zkc(num_trials=num_trials,G=nx.karate_club_graph(),k=2,save_output=True,rounds=num_rounds):
    lamb = 4
    # k=2
    delta_gen = lambda: rand.exponential(lamb,[34])
    full_info_dict = {}
    # G = nx.karate_club_graph()
    deg_centrality = nx.degree_centrality(G)
    eig_centrality = nx.eigenvector_centrality(G)
    close_centrality = nx.closeness_centrality(G)
    for i in G:
        # print(deg_centrality)
        # print(deg_centrality[i])
        full_info_dict[i] = {}
        full_info_dict[i]['deg_centrality'] = deg_centrality[i]
        full_info_dict[i]['eig_centrality'] = eig_centrality[i]
        full_info_dict[i]['close_centrality'] = close_centrality[i]
        full_info_dict[i]['delta_set'] = []
        full_info_dict[i]['total_utility'] = []
        full_info_dict[i]['games_played'] = []
    for j in range(num_trials):

        games_played,total_utility,delta_set = karate_club(delta_gen(),rounds,k=k)
        print("\n\n\nEnd trial {}".format(j))
        for i in G:
             full_info_dict[i]['delta_set'].append(delta_set[rounds+1][i])
             full_info_dict[i]['total_utility'].append(total_utility[rounds+1][i])
             full_info_dict[i]['games_played'].append(games_played[rounds+1][i])
        if save_output:
            np.save('Outputs/ZKC/zachkc_{}_{}_deltas.npy'.format(rounds+2,k),delta_set)
            np.savetxt('Outputs/ZKC/zachkc_{}_{}_deltas.txt'.format(rounds+2,k), delta_set, delimiter=',')
            np.save('Outputs/ZKC/zachkc_{}_{}_total_utility.npy'.format(rounds+2,k),total_utility)
            np.savetxt('Outputs/ZKC/zachkc_{}_{}_total_utility.txt'.format(rounds+2,k), total_utility, delimiter=',')
            np.save('Outputs/ZKC/zachkc_{}_{}_games_played.npy'.format(rounds+2,k),games_played)
            np.savetxt('Outputs/ZKC/zachkc_{}_{}_games_played.txt'.format(rounds+2,k), games_played, delimiter=',')
    return full_info_dict

def zkc_var_set(delta0,node_set,delta_min=0,delta_max=50):
    deltas = delta0.copy()
    for i in node_set:
        deltas = delta0.copy()
        zkc_delta_var(deltas,i)
    plt.show()
        # zkc_delta_var(deltas,i)


def zkc_delta_var(deltas,node,delta_min=0,delta_max=15):
    delta_node = np.linspace(delta_min,delta_max,num=51)
    node_util = []
    node_games = []
    for i in delta_node:
        deltas[node] = i
        print(deltas[node])
        seed = 0
        rand.seed(seed=seed)
        gp,tu,_ = karate_club(deltas,num_rounds)
        node_util.append(tu[num_rounds+1][node]/num_rounds)
        node_games.append(gp[num_rounds+1][node]/num_rounds)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.plot(delta_node,node_util,'o',delta_node,node_util,'k')
    ax1.set_xlabel('$\delta {}$'.format(node+1))
    ax1.set_ylabel('Average utility per round node {}'.format(node+1))
    ax1.set_title('Node {}'.format(node+1))
    ax2 = fig1.add_subplot(122)
    ax2.scatter(delta_node,node_games)
    ax2.set_xlabel('$\delta {}$'.format(node+1))
    ax2.set_ylabel('Average games played per round node {}'.format(node+1))
    ax2.set_title('Node {}'.format(node+1))
    plt.draw()
    # plt.show()




def plot_helper3d(info_dict,n,trials,x,y,z):
    vx = []
    vy = []
    vz = []
    for i in range(n):
        for r in range(trials):
            # print(info_dict[i][z])
            # print(info_dict[i][y])
            # print(info_dict[i][x])
            vz.append(info_dict[i][z][r])
            vy.append(info_dict[i][y][r])
            vx.append(info_dict[i][x])
    return vx,vy,vz


def plot_multi_trial(info_dict,n):
    # fig1 = plt.figure()
    # ax1 = plt.axes(projection='3d')
    x,y,z = plot_helper3d(info_dict,n,num_trials,'deg_centrality','delta_set','total_utility')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121, projection='3d')
    ax1.scatter(x,y,z)
    ax1.set_xlabel('Degree Centrality')
    ax1.set_ylabel('Delta')
    ax1.set_zlabel('Total Utility')
    # plt.show()
    # fig2 = plt.figure()
    x,y,z = plot_helper3d(info_dict,n,num_trials,'eig_centrality','delta_set','total_utility')
    ax2 = fig1.add_subplot(122, projection='3d')
    ax2.scatter(x,y,z)
    ax2.set_xlabel('Eigenvector Centrality')
    ax2.set_ylabel('Delta')
    ax2.set_zlabel('Total Utility')
    plt.show()

    # fig2 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
    #                                mode='markers')])
    # fig2.show()


def draw_zkc():
    G = nx.karate_club_graph()
    a = {}
    b = {}
    n = 34
    for i in range(n):
        a[i] = i*100
        b[i*100] = i+1
    G = nx.relabel_nodes(G,a)
    G = nx.relabel_nodes(G,b)
    print(G.nodes)
    nx.draw_circular(G,with_labels=True,node_color='#3388FF')
    plt.show()


if __name__ == "__main__":
    # seed = 0
    # rand.seed(seed=seed)

    print("Here")
    if zkc:
        delta_gen = lambda: rand.exponential(4,[34])
        deltas = delta_gen()
        node_set = [0,1,11]
        zkc_var_set(deltas,node_set)
    elif dem2:
        demo2()

    # draw_zkc()
    # lamb = 4
    # deltas = rand.exponential(lamb,[34])
    # nodes = [0,1,11]
    # for node in nodes:
    #     d = deltas[node]
    #     zkc_delta_var(deltas,node,delta_min=0,delta_max=15)
    #     deltas[node] = d
    # plt.show()
    # full_info_dict = run_zkc()
    # plot_multi_trial(full_info_dict,34)
    # low35 = np.zeros(num_rounds)
    # low53 = np.zeros(num_rounds)
    # high35 = np.zeros(num_rounds)
    # high53 = np.zeros(num_rounds)
    # perc_deltas = np.zeros([num_rounds,4])
    # for i in range(trials):
    #     print('Trial: ',i)
    #     a,b,c,d = demo1(sys.argv[1])
    #     # print(a,b,c,d)
    #     # print(perc_deltas)
    #     perc_deltas[:,0] += a
    #     perc_deltas[:,2] += b
    #     perc_deltas[:,1] +=c
    #     perc_deltas[:,3] += d
    #     # print(perc_deltas)
    # perc_deltas /= trials
    # print(perc_deltas)
    # np.savetxt('demo_delta_estimate35.txt',perc_deltas,delimiter=',')
    # print(sys.argv[1])
