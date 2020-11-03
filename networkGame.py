import numpy as np
import numpy.random as rand
import random

exp_util_memory = 150
exploit_param = 0
estimate_n = 1000
update_scheme = 2
epoch = 100
explore_scalar =0.85
update_prob = 1.0/100
lazy_factor = 0.0
cost = 0

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
                game_type ="collab",delta_max=30.0,m=91,uniform=True,known_deltas=True):
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
        self.total_values = np.zeros([n])
        self.distributionsL = distributionsL
        self.distributionsF = distributionsF
        self.expected_utility = np.zeros([n,n,m,m,2])
        self.expected_utility /= exp_util_memory
        self.neighbors = {}
        self.explore_prob = {}
        self.rounds = 0
        self.delta_max = delta_max
        self.games_played = np.zeros([n])
        self.uniform = uniform
        self.exploit_param = exploit_param
        for i in range(n):
            entry = []
            for j in range(n):
                if network[i,j] == 1:
                    entry.append(j)
            self.neighbors[i] = entry
            temp = len(entry)
            self.explore_prob[i] = {}
            for j in entry:
                self.explore_prob[i][j] = 1.0/temp
        if not self.known_deltas:
            self.perceived_deltas= {}
            for i in range(n):
                perceptions = {}
                for j in self.neighbors[i]:
                    entry = {}
                    entry['delta_min'] = 0
                    entry['delta_max'] = delta_max
                    entry['delta_estimate'] = (entry['delta_max'] + entry['delta_min'])/2.0
                    perceptions[j] = entry
                self.perceived_deltas[i] = perceptions
        # print("Estimating Expectations...")
        # print("Delta values: %s" %self.deltas)
        # print(deltas)
        # if self.uniform and self.known_deltas:
        #     # print("Uniform Game")
        #     i = 0
        #     j = self.neighbors[i][0]
        #     for g in range(m):
        #         for h in range(m):
        #             for place_holder in range(exp_util_memory):
        #                 self.deltas[i] = g * (delta_max/(m-1))
        #                 self.deltas[j] = h * (delta_max/(m-1))
        #                 (leader_utility,_) = self.play_game(i,j,estimating=True)
        #                 self.expected_utility[i,j,g,h,0] += leader_utility
        #                 (_,follower_utility) = self.play_game(j,i,estimating=True)
        #                 self.expected_utility[i,j,g,h,1] += follower_utility
        #         # print(g)
        #     self.expected_utility /= exp_util_memory
        #     for l1 in range(n):
        #         for l2 in range(n):
        #             self.expected_utility[l1,l2,:,:,:] = self.expected_utility[i,j,:,:,:]

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
        # print("Delta values: %s" %self.deltas)
        # print(deltas)
        # print("Finished Estimations, beginning rounds")

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
                # if (leader == 5 or follower == 5) and debugging:
                    # print(1)
                    # print([leader,follower])
                    # print(A)
                    # print(B)
                    # print(C)
                    # print([(playL,exp_playF),(playL,playF)])
                    # print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_min']
                temp = greedy_response2[playL]
                self.perceived_deltas[leader][follower]['delta_min'] = np.maximum(B[playL,temp] - B[playL,playF],holder)

                # New delta_min greater than or equal to delta_max, indicating a change in delta
                if self.perceived_deltas[leader][follower]['delta_min'] >= self.perceived_deltas[leader][follower]['delta_max']:
                    self.perceived_deltas[leader][follower]['delta_max'] = self.delta_max
                    self.explore_prob[leader][follower] = 1/len(self.neighbors[leader])
                # if (leader == 5 or follower == 5) and debugging:
                    # print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            # Follower generated lower than expected social utility
            elif C[playL,exp_playF] > C[playL,playF]:
                # if (leader == 5 or follower == 5) and debugging:
                    # print(2)
                    # print([leader,follower])
                    # print(A)
                    # print(B)
                    # print(C)
                    # print([(playL,exp_playF),(playL,playF)])
                    # print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_max']
                temp = greedy_response2[playL]
                self.perceived_deltas[leader][follower]['delta_max'] = np.minimum(B[playL,temp] - B[playL,exp_playF],holder)

                # New delta_max less than or equal to delta_min, indicating a change in delta
                if self.perceived_deltas[leader][follower]['delta_min'] >= self.perceived_deltas[leader][follower]['delta_max']:
                    self.perceived_deltas[leader][follower]['delta_min'] = 0
                    self.explore_prob[leader][follower] = 1/len(self.neighbors[leader])
                # if (leader == 5 or follower == 5) and debugging:
                    # print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
        else:
            temp = greedy_response2[playL]
            j = (temp+1)%2
            # Follower did what was expected, but still gave something up
            if C[playL,playF] > C[playL,temp]:
                # if (leader == 5 or follower == 5) and debugging:
                    # print(3)
                    # print([leader,follower])
                    # print(A)
                    # print(B)
                    # print(C)
                    # print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_min']
                self.perceived_deltas[leader][follower]['delta_min'] = np.maximum(B[playL,temp] - B[playL,playF],holder)
                # if (leader == 5 or follower == 5) and debugging:
                    # print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            # Follower did what was expected, gave nothing up
            elif C[playL,playF] < C[playL,j]:
                # if (leader == 5 or follower == 5) and debugging:
                #     print(4)
                #     print([leader,follower])
                #     print(A)
                #     print(B)
                #     print(C)
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[leader][follower]['delta_max']
                self.perceived_deltas[leader][follower]['delta_max'] = np.minimum(B[playL,temp] - B[playL,j],holder)
                # if (leader == 5 or follower == 5) and debugging:
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])

        #Update follower's perception of leader
        assumed_delta1 = self.perceived_deltas[follower][leader]['delta_estimate']
        (exp_playL,exp_playFF) = playLFE(A,B,assumed_delta1,assumed_delta2)

        # Leader behaved unexpectedly
        if exp_playL != playL:
            # Leader expected to generate more social utility than follower expected it to expect
            if C[playL,exp_playF] > C[exp_playL,exp_playFF]:
                # if (leader == 5 or follower == 5) and debugging:
                #     print(5)
                #     print([leader,follower])
                #     print(A)
                #     print(B)
                #     print(C)
                #     print([(exp_playL,exp_playFF),(playL,exp_playF)])
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_min']
                self.perceived_deltas[follower][leader]['delta_min'] = np.maximum(best1 - A[playL,exp_playF],holder)

                # Check if change in delta detected
                if self.perceived_deltas[follower][leader]['delta_min'] >= self.perceived_deltas[follower][leader]['delta_max']:
                    self.perceived_deltas[follower][leader]['delta_max'] = self.delta_max
                    self.explore_prob[follower][leader] = 1/len(self.neighbors[follower])
                # if (leader == 5 or follower == 5) and debugging:
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            # Leader expected to generate less social utility than follower expected it to expect
            elif C[playL,exp_playF] < C[exp_playL,exp_playF]:
                # if (leader == 5 or follower == 5) and debugging:
                #     print(6)
                #     print([leader,follower])
                #     print(A)
                #     print(B)
                #     print(C)
                #     print([(exp_playL,exp_playFF),(playL,exp_playF)])
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_max']
                self.perceived_deltas[follower][leader]['delta_max'] = np.minimum(best1 - A[exp_playL,exp_playFF],holder)

                # Check if change in delta detected
                if self.perceived_deltas[follower][leader]['delta_min'] >= self.perceived_deltas[follower][leader]['delta_max']:
                    self.perceived_deltas[follower][leader]['delta_min'] = 0
                    self.explore_prob[follower][leader] = 1/len(self.neighbors[follower])
                # if (leader == 5 or follower == 5) and debugging:
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])

        # Leader behaves as expected
        else:
            if C[playL,exp_playF] > C[i,respi]:
                # if (leader == 5 or follower == 5) and debugging:
                #     print(7)
                #     print([leader,follower])
                #     print(A)
                #     print(B)
                #     print(C)
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_min']
                # print([holder,A[i,respi],A[playL,exp_playF]])
                self.perceived_deltas[follower][leader]['delta_min'] = np.maximum(best1 - A[playL,exp_playF],holder)
                # if (leader == 5 or follower == 5) and debugging:
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
            if C[playL,exp_playF] < C[i,respi]:
                # if (leader == 5 or follower == 5) and debugging:
                #     print(8)
                #     print([leader,follower])
                #     print(A)
                #     print(B)
                #     print(C)
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])
                holder = self.perceived_deltas[follower][leader]['delta_max']
                # print([holder,A[i,respi],A[playL,exp_playF]])
                self.perceived_deltas[follower][leader]['delta_max'] = np.minimum(A[i,respi] - A[playL,exp_playF],holder)
                # if (leader == 5 or follower == 5) and debugging:
                #     print([self.perceived_deltas[leader][follower],self.perceived_deltas[follower][leader]])

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
            print(self.deltas[leader],self.deltas[follower])
            print([(a11,a12),(a21,a22)])
            print([(b11,b12),(b21,b22)])
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
            self.total_values[leader] += A[playL,playF] - cost
            self.total_values[follower] += B[playL,playF] - cost
            self.games_played[leader] += 1
            self.games_played[follower] += 1
        return (A[playL,playF],B[playL,playF])

    # Currently only enabled for uniform games with game_type = "collab"
    def play_round(self):
        if self.game_type == "collab":
            self.play_round_collab()
            self.rounds +=1
            self.exploit_param +=1
        elif self.game_type == "partner":
            self.play_round_partnership_uniform()
            self.rounds +=1
        elif self.game_type == "dictator":
            self.play_round_dictator_uniform()
            self.rounds +=1
        else:
            print("Invalid game type")

    def estimate_game_value(self,leader,follower):
        sumL = 0
        sumF = 0
        # for i in range(estimate_n):
        (valL,valF) = self.play_game(leader,follower,estimating=True)
        sumL += valL - cost
        sumF += valF - cost
        return (sumL*1.0),(sumF*1.0)

    def select_uniform_collaborators(self,i,neighbors):
        if self.known_deltas:
            helper = lambda x: self.deltas[x]
            # neighbors = sorted(self.neighbors[i], key = helper,reverse=True)
        else:
            helper = lambda x: self.perceived_deltas[i][x]['delta_estimate']
            # neighbors = sorted(self.neighbors[i], key = helper, reverse = True)
        if (len(neighbors) <= self.k) or (helper(neighbors[self.k-1]) > helper(neighbors[self.k])):
            followers = neighbors[:self.k]
            return followers
        else:
            # There are several more elements with same delta.
            # print(i,neighbors,helper(neighbors))
            l = len(neighbors)
            x = neighbors.pop(0)
            temp  = [x]
            idx = 0
            count = 1
            go = True
            while go and len(neighbors)>0:
                x = neighbors.pop(0)
                if count < self.k or helper(temp[-1]) == helper(x):
                    if helper(x) < helper(temp[idx]):
                        idx = count
                    temp.append(x)
                    count += 1
                else:
                    go = False
            if count == self.k:
                return temp
            else:

                followers = temp[:idx]
                sampler = temp[idx:]
                # print(i,temp,followers,sampler, helper(temp))
                samples = self.k-idx
                followers = followers+(random.sample(sampler,samples))
                # print(i,followers)
                return followers




    def play_round_collab(self):
        if self.known_deltas and self.uniform:
            for i in range(self.n):
                helper = lambda x: self.deltas[x]
                neighbors = sorted(self.neighbors[i], key = helper,reverse=True)
                # print(i,neighbors)
                followers = self.select_uniform_collaborators(i,neighbors)
                # print(followers)
                for j in followers:
                    self.play_game(i,j)
        else:
            for i in range(self.n):
                estimates = {}
                if self.uniform:
                    for j in self.neighbors[i]:
                        estimates[j] = self.perceived_deltas[i][j]['delta_estimate']
                else:
                    for j in self.neighbors[i]:
                        for l in range(estimate_n):
                            temp,_ =self.estimate_game_value(i,j)
                            estimates[j] += temp/estimate_n
                helper = lambda x: estimates[x]
                neighbors = sorted(self.neighbors[i], key = helper, reverse=True)
                if len(neighbors) > self.k:
                    if self.known_deltas:
                        followers = neighbors[:self.k]
                        for j in followers:
                            self.play_game(i,j)
                    else:
                        exploit = self.exploit_num() # Select number of "handles" to exploit vs. explore
                        followers = neighbors[:exploit]
                        remainder = neighbors[exploit:]
                        explore = self.k - exploit
                        # print([i,followers,remainder,exploit])
                        if update_scheme == 1:
                            prob = np.zeros(len(remainder))
                            # print(i, remainder,prob,explore)
                            prob_tot = 0
                            count = 0
                            for j in remainder:
                                prob[count] = self.explore_prob[i][j]
                                prob_tot += self.explore_prob[i][j]
                                count +=1
                            # print(prob_tot,prob/prob_tot)
                            extra = list(rand.choice(remainder,explore,replace=False,p= prob/prob_tot))
                            # print(followers,extra)
                            followers = followers + extra
                        elif update_scheme ==2:
                            followers = followers + random.sample(remainder,explore)
                        for j in followers:
                            self.play_game(i,j)
                else:
                    for j in neighbors:
                        self.play_game(i,j)
            if not self.known_deltas:
                for i in range(self.n):
                    for j in self.neighbors[i]:
                        # Update delta_estimate
                        self.perceived_deltas[i][j]['delta_estimate'] = ((self.perceived_deltas[i][j]['delta_max']-self.perceived_deltas[i][j]['delta_min'])/2 +
                            self.perceived_deltas[i][j]['delta_min'])
                        self.perceived_deltas[j][i]['delta_estimate'] = ((self.perceived_deltas[j][i]['delta_max']-self.perceived_deltas[j][i]['delta_min'])/2 +
                            self.perceived_deltas[j][i]['delta_min'])

    def exploit_num(self,node = 0):
        if update_scheme == 1:
            prob = 0
            for i in self.neighbors[node]:
                prob += self.explore_prob[node][i]
                self.explore_prob[node][i] *= explore_scalar
            epsilon = np.minimum(1.0,2.0*prob)
        elif update_scheme == 2:
            epsilon = np.minimum(1.0,2.0/(self.exploit_param+1))
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

    # Does not worry about explore-exploit problem
    def select_collaborators(self,j,update=False):
        # For known deltas, use true delta values to estimate
        # if self.uniform:
        #     if self.known_deltas:
        #         helper = lambda x:self.deltas[x]
        #     else:
        #         helper = lambda x:self.perceived_deltas[j][x]['delta_estimate']
        if self.known_deltas:
            if self.uniform:
                helper = lambda x: self.deltas[x]
            else:
                helper = lambda x:self.expected_utility_game(j,x,self.deltas[j],self.deltas[x],update=update)
            # For unknown deltas and a hypothetical delta_j, use that and perceived deltas to estimate
        else:
            if self.uniform:
                helper = lambda x: self.perceived_deltas[j][x]['delta_estimate']
            else:
                helper = lambda x:self.expected_utility_game(j,x,self.deltas[j],self.perceived_deltas[j][x]['delta_estimate'],update=update)
        # if not self.uniform:
        neighbors = sorted(self.neighbors[j],key=helper,reverse=True)
        # print(neighbors)
        # print(helper(neighbors))
        if not self.uniform:
            collaborators = neighbors[:self.k]
            return collaborators
        else:
            return self.select_uniform_collaborators(j,neighbors)

    # Only for uniform systems
    def predict_invites(self,j):
        neighbors = self.neighbors[j]
        invites = []
        if self.known_deltas:
            for i in neighbors:
                # per_delta = self.perceived_deltas[i][j]['delta_estimate']
                collab_i = self.select_collaborators(i,update = False)
                if j in collab_i:
                    invites.append(i)

        # Uses j's perception of delta_i for true delta_i
        else:
            for i in neighbors:
                temp = self.perceived_deltas[i][j]['delta_estimate']
                self.perceived_deltas[i][j]['delta_estimate'] = self.deltas[j]
                collab_i = self.select_collaborators(i,update=False)
                self.perceived_deltas[i][j]['delta_estimate'] = temp
                if j in collab_i:
                    invites.append(i)
        return invites


    def expected_utility_game(self,leader,follower,delta1,delta2,update=True):
        util_L = 0
        util_F = 0
        d1 = self.deltas[leader]
        d2 = self.deltas[follower]
        self.deltas[leader] = delta1
        self.deltas[follower] = delta2
        for i in range(estimate_n):
            (val_L,val_F) = self.play_game(leader,follower,estimating=True)
            util_L += val_L
            util_F += val_F
        #
        # # print(leader,follower)
        # # print(delta1,delta2)
        # m1 = int(round(delta1*(self.m-1)/self.delta_max))
        # m2 = int(round(delta2*(self.m-1)/self.delta_max))
        # # print(leader,follower,m1,m2)
        # # print(m1)
        # # print(m2)
        # # print(delta1)
        # # print(delta2)
        # util_L = util_L/exp_util_memory + self.expected_utility[leader,follower,m1,m2,0]*(exp_util_memory-1.0)/exp_util_memory
        # util_F = util_F/exp_util_memory + self.expected_utility[leader,follower,m1,m2,1]*(exp_util_memory-1.0)/exp_util_memory
        # if update:
        #     self.expected_utility[leader,follower,m1,m2,0] = util_L
        #     self.expected_utility[leader,follower,m1,m2,1] = util_F
        self.deltas[leader] = d1
        self.deltas[follower] = d2
        return util_L #,util_F

    # Estimate the expected utility of using delta for player
    def expected_utility_total(self,player,delta,update=True):
        old_delta = self.deltas[player]
        self.deltas[player] = delta
        # print(delta)
        collabs = self.select_collaborators(player,update=update)
        util = 0
        for i in collabs:
            for l in range(estimate_n):
                valL,_ = self.estimate_game_value(player,i)
                util += valL/estimate_n
        if self.uniform and self.known_deltas:
            for l in range(estimate_n):
                invites = self.predict_invites(player)
                for i in invites:
                    _,valF = self.estimate_game_value(i,player)
                    util += valF/estimate_n
        else:
            invites = self.predict_invites(player)
            for l in range(estimate_n):
                for i in invites:
                    _,valF = self.estimate_game_value(i,player)
                    util += valF/estimate_n
        self.deltas[player] = old_delta
        return util


        # m1 = int(round(delta*(self.m-1)/self.delta_max))
        # for i in collabs:
        #     m2 = int(round(self.deltas[i]*(self.m-1)/self.delta_max))
        #     util += self.expected_utility[player,i,m1,m2,0]
        # invites = self.predict_invites(player)
        # for i in invites:
        #     m2 = int(round(self.deltas[i]*(self.m-1)/self.delta_max))
        #     util += self.expected_utility[i,player,m2,m1,1]
        # return util


    def update_player_collab(self,j):
        if j not in self.fixed_deltas:
            delta_set = [self.delta_max/(self.m-1)*i for i in range(self.m)]
            best_util = 0
            best_delta = 0
            for delta in delta_set:
                util = self.expected_utility_total(j,delta)
                print(self.rounds,j,delta,util)
                if util >= best_util:
                    best_util = util
                    best_delta=delta
            util = self.expected_utility_total(j,self.deltas[j])
            if util >= best_util + lazy_factor:
                best_delta = self.deltas[j]
            return best_delta
        else:
            return self.deltas[j]

    def update_collab(self):
        new_deltas = np.zeros([1,self.n])
        for i in range(self.n):
            if self.update_player(i):
                print("Updating",self.rounds)
                new_deltas[0,i] = self.update_player_collab(i)
            else:
                new_deltas[0,i] = self.deltas[i]
        # print(new_deltas)
        new_deltas = new_deltas[0]
        # print(self.rounds)
        # print(new_deltas)
        # print("Updated")
        self.deltas = new_deltas

    def update_player(self,j):
        update = False
        # Independent probability
        if update_scheme == 1:
            if rand.random() < update_prob:
                update = True
        # Epoch scheme
        if update_scheme == 2:
            if self.rounds % epoch == epoch -1:
                update = True
                exploit_param = 0

        return update
