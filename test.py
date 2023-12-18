import time, random, math

def q_learn(game, timeMax): 
	def superState(s, action):
		col = row = 0

		# total yards left/time left
		if s[3] == 0:
			agg1 = float('inf')
		else:
			agg1 = s[0]/s[3]
   
		# yards till 1st down/downs left
		if s[1] == 0:
			agg2 = float('inf')
		else:
			agg2 = s[2]/s[1]
		
		if agg1 > 4:
			row = 2
		elif agg1 > 2:
			row = 1
		else:
			row = 0
		
		if agg2 > 2:
			col = 2
		elif agg2 > 1:
			col = 1
		else:
			col = 0

		return (action, row, col)

	def findMaxAction(s):
		maxAction = 0
		maxQ = float('-inf')
		for action in range(game.offensive_playbook_size()):
			result = superState(s, action)
			Q = qtable[result]
			if Q > maxQ:
				maxQ = Q
				maxAction = action
		return maxAction
    
	epsilon = 0.15
	learningRate = 0.25
	qtable = {} 
	alpha_table = {}
	for a in range(game.offensive_playbook_size()):
		for row in range(3):
			for col in range(3):
				qtable[(a,row,col)] = 0
				alpha_table[(a,row,col)] = learningRate
    
	start = time.time()
	while  (time.time() - start) < timeMax:
		s = game.initial_position()
		while not game.game_over(s):
			action = 0
			if (random.random() < epsilon):
				action = random.randint(0, game.offensive_playbook_size()-1)
			else: 
				action = findMaxAction(s)
			
			sPrime = game.result(s, action)[0]
			reward = 0
			if not game.game_over(sPrime):
				reward = 0
			elif game.win(sPrime):
				reward = 1
			else:
				reward = -1
    
			superStateKey = superState(s,action)
			if game.game_over(sPrime):
				qtable[superStateKey] += (alpha_table[superStateKey] * (reward  - qtable[superStateKey]))
			else:
				aPrime = findMaxAction(sPrime)
				superStatePrimeKey = superState(sPrime,aPrime)
				qtable[superStateKey] += (alpha_table[superStateKey] * (reward + (0.9 * qtable[superStatePrimeKey] - qtable[superStateKey])))
			alpha_table[superStateKey] *= 0.999
			s = sPrime
   
	return findMaxAction