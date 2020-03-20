import numpy as np
import numpy.random as rand
import random

exp_util_memory = 150
estimate_n = 100

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
    if C[j,respj] >= C[i,respi] and best1 - A[j,respj] <= delta1:
        playL = j
        playF = respj
    else:
        playL = i
        playF = respi

    return (playL,playF)



class SNGame:
    def __init__(self,n,k,network,distributionsL,distributionsF,deltas=[],fixed_deltas=[],
                game_type ="collab",delta_max=50.0,m=251,uniform=True,known_deltas=True):
        self.n = n
        self.k = k  # Num games per round
        self.m = m
        self.known_deltas=known_deltas
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
        self.uniform = uniform
        for i in range(n):
            entry = []
            for j in range(n):
                if network[i,j] == 1:
                    entry.append(j)
            self.neighbors[i] = entry
        if not self.known_deltas:
            self.perceived_deltas= {}
            for i in range(n):
                perceptions = {}
                for j in self.neighbors[i]:
                    entry = {}
                    entry['delta_min'] = 0
                    entry['delta_max'] = delta_max
                    entry['delta_estimate'] = (entry['delta_max'] - entry['delta_min'])/2.0 + entry['delta_min']
                    perceptions[j] = entry
                self.perceived_deltas[i] = perceptions
        print("Estimating Expectations...")
        print("Delta values: %s" %self.deltas)
        print(deltas)
        if self.uniform and self.known_deltas:
            print("Uniform Game")
            i = 0
            j = self.neighbors[i][0]
            for g in range(m):
                for h in range(m):
                    for place_holder in range(exp_util_memory):
                        self.deltas[i] = g * (delta_max/(m-1))
                        self.deltas[j] = h * (delta_max/(m-1))
                        (leader_utility,_) = self.play_game(i,j,estimating=True)
                        self.expected_utility[i,j,g,h,0] += leader_utility
                        (_,follower_utility) = self.play_game(j,i,estimating=True)
                        self.expected_utility[i,j,g,h,1] += follower_utility
                print(g)
            self.expected_utility /= exp_util_memory
            for l1 in range(n):
                for l2 in range(n):
                    self.expected_utility[l1,l2,:,:,:] = self.expected_utility[i,j,:,:,:]

        # else:
        #     for i in range(n):
        #         for j in range(n):
        #             print((i,j))
        #             if j in self.neighbors[i]:
        #                 for g in range(m):
        #                     for h in range(m):
        #                         for place_holder in range(exp_util_memory):
        #                             self.deltas[i] = g * (delta_max/(m-1))
        #                             self.deltas[j] = h * (delta_max/(m-1))
        #                             (leader_utility,_) = self.play_game(i,j,estimating=True)
        #                             self.expected_utility[i,j,g,h,0] += leader_utility
        #                             (_,follower_utility) = self.play_game(j,i,estimating=True)
        #                             self.expected_utility[i,j,g,h,1] += follower_utility
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


    def playLFE_real(self,A,B,leader,follower):
        flag = leader == 5 or follower == 5
        debugging = False
        delta1 = self.deltas[leader]
        delta2 = self.deltas[follower]
        if self.known_deltas:
            assumed_delta2 = delta2
        else:
            assumed_delta2 = self.perceived_deltas[leader][follower]['delta_estimate']
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
        # Compute player 1's assumed highest utility
        for k in range(2):
            respk = int(assumed_response2[k])
            if A[k,respk] >= best1:
                i = k
                best1 = A[k,respk]

        j = (i+1) % 2
        respj = int(assumed_response2[j])
        respi = int(assumed_response2[i])
        if C[j,respj] >= C[i,respi] and best1 - A[j,respj] <= delta1:
            playL = j
            playF = int(response2[j])
        else:
            playL = i
            playF = int(response2[i])

        #Update leader's perception of follower
        exp_playF = int(assumed_response2[playL])

        if flag and debugging:
            print('a')
            print(response2)
            print('b')
            print(greedy_response2)
            print('c')
            print(assumed_response2)
            print('d')
            print([playL,playF])

        a11 = self.perceived_deltas[leader][follower]['delta_min']
        a12 = self.perceived_deltas[leader][follower]['delta_max']
        a21 = self.perceived_deltas[follower][leader]['delta_min']
        a22 = self.perceived_deltas[follower][leader]['delta_max']
        # Follower behaved unexpectedly
        if exp_playF != playF:
            # Follower generated higher than expected social utility
            if C[playL,exp_playF] < C[playL,playF]:
                if (leader == 5 or follower == 5) and debugging:
                    print(1)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([(playL,exp_playF),(playL,playF)])
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_min']
                temp = greedy_response2[playL]
                self.perceived_deltas[leader][follower]['delta_min'] = np.maximum(B[playL,temp] - B[playL,playF],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            # Follower generated lower than expected social utility
            elif C[playL,exp_playF] > C[playL,playF]:
                if (leader == 5 or follower == 5) and debugging:
                    print(2)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([(playL,exp_playF),(playL,playF)])
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_max']
                temp = greedy_response2[playL]
                self.perceived_deltas[leader][follower]['delta_max'] = np.minimum(B[playL,temp] - B[playL,exp_playF],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
        else:
            temp = greedy_response2[playL]
            j = (temp+1)%2
            # Follower did what was expected, but still gave something up
            if C[playL,playF] > C[playL,temp]:
                if (leader == 5 or follower == 5) and debugging:
                    print(3)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_min']
                self.perceived_deltas[leader][follower]['delta_min'] = np.maximum(B[playL,temp] - B[playL,playF],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            # Follower did what was expected, gave nothing up
            elif C[playL,playF] < C[playL,j]:
                if (leader == 5 or follower == 5) and debugging:
                    print(4)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_max']
                self.perceived_deltas[leader][follower]['delta_max'] = np.minimum(B[playL,temp] - B[playL,j],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])

        #Update follower's perception of leader
        assumed_delta1 = self.perceived_deltas[follower][leader]['delta_estimate']
        (exp_playL,exp_playFF) = playLFE(A,B,assumed_delta1,assumed_delta2)

        # Leader behaved unexpectedly
        if exp_playL != playL:
            # Leader expected to generate more social utility than follower expected it to expect
            if C[playL,exp_playF] > C[exp_playL,exp_playFF]:
                if (leader == 5 or follower == 5) and debugging:
                    print(5)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([(exp_playL,exp_playFF),(playL,exp_playF)])
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_min']
                self.perceived_deltas[follower][leader]['delta_min'] = np.maximum(best1 - A[playL,exp_playF],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            # Leader expected to generate less social utility than follower expected it to expect
            elif C[playL,exp_playF] < C[exp_playL,exp_playF]:
                if (leader == 5 or follower == 5) and debugging:
                    print(6)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([(exp_playL,exp_playFF),(playL,exp_playF)])
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_max']
                self.perceived_deltas[follower][leader]['delta_max'] = np.minimum(best1 - A[exp_playL,exp_playFF],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])

        # Leader behaves as expected
        else:
            if C[playL,exp_playF] > C[i,respi]:
                if (leader == 5 or follower == 5) and debugging:
                    print(7)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_min']
                # print([holder,A[i,respi],A[playL,exp_playF]])
                self.perceived_deltas[follower][leader]['delta_min'] = np.maximum(best1 - A[playL,exp_playF],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            if C[playL,exp_playF] < C[i,respi]:
                if (leader == 5 or follower == 5) and debugging:
                    print(8)
                    print([leader,follower])
                    print(A)
                    print(B)
                    print(C)
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_max']
                # print([holder,A[i,respi],A[playL,exp_playF]])
                self.perceived_deltas[follower][leader]['delta_max'] = np.minimum(A[i,respi] - A[playL,exp_playF],holder)
                if (leader == 5 or follower == 5) and debugging:
                    print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])

        b11 = self.perceived_deltas[leader][follower]['delta_min']
        b12 = self.perceived_deltas[leader][follower]['delta_max']
        b21 = self.perceived_deltas[follower][leader]['delta_min']
        b22 = self.perceived_deltas[follower][leader]['delta_max']
        if (b11 < a11 or b12 > a12 or b21 < a21 or b22 > a22 or b22 <= b21 or b12 <= b11 or b22 < delta1 or b12 < delta2):
            print("\n\nLOOK HERE!!\n\n")
            print([leader,follower])
            print(A)
            print(B)
            print(C)
            # print(holder)
            print(best1)
            print(greedy_response2[playL])
            print([(a11,a12),(a21,a22)])
            print([playL,playF])
            print([exp_playL,exp_playF,exp_playFF])
            print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
        return (playL,playF)


    def play_game(self,leader,follower,estimating=False):
        delta1 = self.deltas[leader]
        delta2 = self.deltas[follower]
        A = self.distributionsL[leader][follower]()
        B = self.distributionsF[follower][leader]()
        if self.known_deltas:
            (playL,playF) = playLFE(A,B,delta1,delta2)
        elif estimating:
            assumed_delta2 = self.perceived_deltas[leader][follower]['delta_estimate']
            (playL,playF) = playLFE(A,B,delta1,assumed_delta2,assumed_delta2=assumed_delta2)
        else:
            (playL,playF) = self.playLFE_real(A,B,leader,follower)
        if not estimating:
            self.total_values[leader] += A[playL,playF]
            self.total_values[follower] += B[playL,playF]
            self.games_played[leader] += 1
            self.games_played[follower] += 1
        return (A[playL,playF],B[playL,playF])

    # Currently only enabled for uniform games with game_type = "collab"
    def play_round(self):
        if self.game_type == "collab":
            self.play_round_collab()
            self.rounds +=1
        elif self.game_type == "partner":
            self.play_round_partnership_uniform()
            self.rounds +=1
        elif self.game_type == "dictator":
            self.play_round_dictator_uniform()
            self.rounds +=1
        else:
            print("Invalid game type")

    def estimate_game_value(self,leader,follower):
        sum = 0
        for i in range(estimate_n):
            (val,_) = self.play_game(leader,follower,estimating=True)
            sum += val
        return (val*1.0)/estimate_n

    def play_round_collab(self):
        if self.known_deltas and self.uniform:
            for i in range(self.n):
                helper = lambda x: self.deltas[x]
                neighbors = sorted(self.neighbors[i], key = helper,reverse=True)
                followers = neighbors[:self.k]
                for j in followers:
                    self.play_game(i,j)
        else:
            for i in range(self.n):
                estimates = {}
                for j in self.neighbors[i]:
                    estimates[j] = self.estimate_game_value(i,j)
                helper = lambda x: estimates[x]
                neighbors = sorted(self.neighbors[i], key = helper, reverse=True)
                if self.known_deltas:
                    followers = neighbors[:self.k]
                    for j in followers:
                        self.play_game(i,j)
                else:
                    exploit = self.exploit_num() # Select number of "handles" to exploit vs. explore
                    followers = neighbors[:exploit]
                    remainder = neighbors[exploit:]
                    explore = self.k - exploit
                    # print([followers,remainder,exploit])
                    followers = followers + random.sample(remainder,explore)
                    for j in followers:
                        self.play_game(i,j)
            if not self.known_deltas:
                for i in range(self.n):
                    for j in self.neighbors[i]:
                        # Update delta_estimate
                        self.perceived_deltas[i][j]['delta_estimate'] = ((self.perceived_deltas[i][j]['delta_max']-self.perceived_deltas[i][j]['delta_min'])/2 +
                            self.perceived_deltas[i][j]['delta_min'])
                        self.perceived_deltas[j][i]['delta_estimate'] = ((self.perceived_deltas[j][i]['delta_max']-self.perceived_deltas[j][i]['delta_min'])/2 +
                            self.perceived_deltas[j][i]['delta_min'])


    def exploit_num(self):
        epsilon = 1.0/(self.rounds+1)
        return rand.binomial(self.k,1-epsilon)

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
        (util_L,util_F) = self.play_game(leader,follower,estimating=True)
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
