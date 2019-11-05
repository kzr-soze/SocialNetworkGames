import numpy as np
import numpy.random as rand


# Plays a Leader-Follower game
def playLFE(A,B,delta1,delta2,assumed_delta2=-1):
    if assumed_delta2 == -1:
        assumed_delta2 = delta2
    C = A+B
    response2 = np.zeros(2)
    assumed_response2 = np.zeros(2)
    greedy_response2 = np.argmax(B,axis=1)
    for k in range(2):
        i = greedy_response2[k]
        j = (i+1) % 2

        # Compute how follower will respond to k
        if C[k,j] >= C[k,i] and B[k,i] - B[k,j] <= delta2:
            response2[k] = j
        else:
            response2[k] = i

        # Compute how leader assumes follower will respond to k
        if C[k,j] >= C[k,i] and B[k,i] - B[k,j] <= assumed_delta2:
            assumed_response2[k] = j
        else:
            assumed_response2[k] = i
    i = -1
    best1 = -1
    # print(assumed_response2)
    for k in range(2):
        respk = int(assumed_response2[k])
        if A[k,respk] >= best1:
            i = k
            best1 = A[k,respk]

    j = (i+1) % 2
    respj = int(assumed_response2[j])
    respi = int(assumed_response2[i])
    if C[j,respj] >= C[i,respi] and best1 - B[j,respj] <= delta1:
        playL = j
        playF = respj
    else:
        playL = i
        playF = respi

    return (playL,playF)



class SNGame:
    def __init__(self,n,k,network,distributionsL,distributionsF,deltas=[],fixed_deltas=[], game_type ="collab"):
        self.n = n
        self.k = k
        if len(deltas) == 0:
            deltas = np.zeros([n,1])
        self.network = network # assumed to be connectivity matrix
        self.deltas = deltas
        self.fixed_deltas = fixed_deltas
        self.game_type = game_type
        self.total_values = np.zeros([n,1])
        self.distributionsL = distributionsL
        self.distributionsF = distributionsF
        self.neighbors = {}
        self.rounds = 0
        self.games_played = np.zeros([n,1])
        for i in range(n):
            entry = []
            for j in range(n):
                if network[i,j] == 1:
                    entry.append(j)
            self.neighbors[i] = entry

    def print_game(self):
        print("Number of players: %s" % self.n)
        print("Maximum games per round: %s" %self.k)
        print("Game type: %s" %self.game_type)
        print("Network: ")
        print(self.network)
        print("Delta values: %s" %self.deltas)
        print("Total rounds played: %s" %self.rounds)
        print("Total values per player: %s" %self.total_values)
        print("Total games played per player: %s" % self.games_played)


    def play_game(self,leader,follower,updating=False):
        delta1 = self.deltas[leader]
        delta2 = self.deltas[follower]
        A = self.distributionsL[leader][follower]()
        B = self.distributionsF[follower][leader]()
        (playL,playF) = playLFE(A,B,delta1,delta2)
        if not updating:
            self.total_values[leader] += A[playL,playF]
            self.total_values[follower] += B[playL,playF]
            self.games_played[leader] += 1
            self.games_played[follower] += 1

    # Currently only enabled for uniform games with game_type = "collab"
    def play_round(self):
        if self.game_type == "collab":
            self.play_round_collab_uniform()
            self.rounds +=1
        elif self.game_type == "partner":
            self.play_round_partnership_uniform()
            self.rounds +=1
        elif self.game_type == "dictator":
            self.play_round_dictator_uniform()
            self.rounds +=1
        else:
            print("Invalid game type")

    def play_round_collab_uniform(self):
        for i in range(self.n):
            helper = lambda x: self.deltas[x]
            neighbors = sorted(self.neighbors[i], key = helper,reverse=True)
            followers = neighbors[:self.k]
            for j in followers:
                self.play_game(i,j)

    def play_round_partnership_uniform(self):
        order = []
        k = self.k
        for i in range(self.n):
            order.append(i)
        helper = lambda x:self.deltas[x]
        order = sorted(order, key = helper,reverse=True)
        plays = np.zeros(self.n)
        for i in order:
            neighbors = sorted(self.neighbors[i],key = helper,reverse=True)
            for j in neighbors:
                if plays[i] >= k:
                    break
                elif plays[j] < k:
                    flip = rand.randint(2)
                    if flip == 1:
                        leader = i
                        follower = j
                    else:
                        leader = j
                        follower = i
                    self.play_game(leader,follower)
                    plays[i] +=1
                    plays[j] +=1

    def play_round_dictator_uniform(self):
        order = rand.permutation(self.n)
        helper = lambda x:self.deltas[x]
        k = self.k
        plays = np.zeros(self.n)
        for i in order:
            neighbors = sorted(self.neighbors[i],key = helper,reverse=True)
            for j in neighbors:
                if plays[i] >= k:
                    break
                elif plays[j] < k:
                    flip = rand.randint(2)
                    if flip == 1:
                        leader = i
                        follower = j
                    else:
                        leader = j
                        follower = i
                    self.play_game(leader,follower)
                    plays[i] +=1
                    plays[j] +=1
