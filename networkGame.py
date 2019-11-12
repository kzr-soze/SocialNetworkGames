import numpy as np
import numpy.random as rand

exp_util_memory = 150

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
    def __init__(self,n,k,network,distributionsL,distributionsF,deltas=[],fixed_deltas=[], game_type ="collab",delta_max=100.0,m=501):
        self.n = n
        self.k = k  # Num games per round
        self.m = m
        if len(deltas) == 0:
            deltas = np.zeros([n,1])
        self.network = network # assumed to be connectivity matrix
        self.deltas = np.zeros([n,1])
        self.fixed_deltas = fixed_deltas
        self.game_type = game_type
        self.total_values = np.zeros([n,1])
        self.distributionsL = distributionsL
        self.distributionsF = distributionsF
        self.expected_utility = np.zeros([n,n,m,m,2])
        self.expected_utility /= exp_util_memory
        self.neighbors = {}
        self.rounds = 0
        self.delta_max = delta_max
        self.games_played = np.zeros([n,1])
        for i in range(n):
            entry = []
            for j in range(n):
                if network[i,j] == 1:
                    entry.append(j)
            self.neighbors[i] = entry
        print("Estimating Expectations...")
        print("Delta values: %s" %self.deltas)
        print(deltas)
        for i in range(n):
            for j in range(n):
                print((i,j))
                if j in self.neighbors[i]:
                    for g in range(m):
                        for h in range(m):
                            for place_holder in range(exp_util_memory):
                                self.deltas[i] = g * (delta_max/(m-1))
                                self.deltas[j] = h * (delta_max/(m-1))
                                (leader_utility,_) = self.play_game(i,j,updating=True)
                                self.expected_utility[i,j,g,h,0] += leader_utility
                                (_,follower_utility) = self.play_game(j,i,updating=True)
                                self.expected_utility[i,j,g,h,1] += follower_utility
        self.deltas = deltas
        print("Delta values: %s" %self.deltas)
        print(deltas)
        print("Finished Estimations, beginning rounds")

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
        return (A[playL,playF],B[playL,playF])

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

    def select_collaborators(self,j,update=False):
        helper = lambda x:self.expected_utility_game(j,x,self.deltas[j],self.deltas[x],update=update)
        neighbors = sorted(self.neighbors[j],key=helper,reverse=True)
        collaborators = neighbors[:self.k]
        return collaborators

    def predict_invites(self,j):
        neighbors = self.neighbors[j]
        invites = []
        for i in neighbors:
            collab_i = self.select_collaborators(i,update = False)
            if j in collab_i:
                invites.append(i)
        return invites


    def expected_utility_game(self,leader,follower,delta1,delta2,update=True):
        (util_L,util_F) = self.play_game(leader,follower,updating=True)
        m1 = int(round(delta1*(self.m-1)/self.delta_max))
        m2 = int(round(delta2*(self.m-1)/self.delta_max))
        # print(leader,follower,m1,m2)
        # print(m1)
        # print(m2)
        # print(delta1)
        # print(delta2)
        util_L = util_L/exp_util_memory + self.expected_utility[leader,follower,m1,m2,0]*(exp_util_memory-1.0)/exp_util_memory
        util_F = util_F/exp_util_memory + self.expected_utility[leader,follower,m1,m2,1]*(exp_util_memory-1.0)/exp_util_memory
        if update:
            self.expected_utility[leader,follower,m1,m2,0] = util_L
            self.expected_utility[leader,follower,m1,m2,1] = util_F
        return util_L,util_F

    def expected_utility_total(self,player,delta,update=True):
        old_delta = self.deltas[player]
        self.deltas[player] = delta
        collabs = self.select_collaborators(player,update=update)
        util = 0
        m1 = int(round(delta*(self.m-1)/self.delta_max))
        for i in collabs:
            m2 = int(round(self.deltas[i]*(self.m-1)/self.delta_max))
            util += self.expected_utility[player,i,m1,m2,0]
        invites = self.predict_invites(player)
        for i in invites:
            m2 = int(round(self.deltas[i]*(self.m-1)/self.delta_max))
            util += self.expected_utility[i,player,m2,m1,1]
        return util


    def update_player_collab(self,j):
        if j not in self.fixed_deltas:
            # candidates = self.neighbors[j]
            #
            # # Construct 2-hop neighborhood
            # for i in self.neighbors[j]:
            #     two_hop = self.neighbors[i]
            #     for l in two_hop:
            #         if l not in candidates and l !=j:
            #             candidates.append(l)
            #
            # helper = lambda x:self.deltas[x]
            # # print(self.deltas[j])
            # delta_set = [0,self.deltas[j]]
            # # delta_set = [delta_set.append(x+(self.delta_max/(self.m-1))) for x in self.deltas[ordered] if x+ (self.delta_max/(self.m-1)) not in delta_set ]
            # for entry in candidates:
            #     temp = min(self.deltas[entry] + (self.delta_max/(self.m-1)),self.delta_max)
            #     # print(temp)
            #     # print(self.deltas[entry] + (self.delta_max/(self.m-1)))
            #     delta_set.append(temp)
            # print(delta_set)
            # print(self.rounds)
            delta_set = [self.delta_max/(self.m-1)*i for i in range(self.m)]
            best_util = 0
            best_delta = 0
            for delta in delta_set:
                util = self.expected_utility_total(j,delta)
                if util >= best_util:
                    best_util = util
                    best_delta=delta
            return best_delta
        else:
            return self.deltas[j]

    def update_collab(self):
        new_deltas = np.zeros([1,self.n])
        for i in range(self.n):
            new_deltas[0,i] = self.update_player_collab(i)
        # print(new_deltas)
        new_deltas = new_deltas[0]
        print(self.rounds)
        # print(new_deltas)
        # print("Updated")
        self.deltas = new_deltas
