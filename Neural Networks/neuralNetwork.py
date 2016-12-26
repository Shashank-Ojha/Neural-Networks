from random import random, choice, shuffle
trials = 3000
alpha = 0.25
#inputs  = [[0,0,-1,0],[0,1,-1,1],[1,0,-1,1],[1,1,-1,1]] # OR Gates
#inputs  = [[0,0,-1,1],[0,1,-1,0],[1,0,-1,0],[1,1,-1,1]] # XOR Gates 2D
#inputs  = [[0,0,1,-1,0],[0,1,0,-1,1],[1,0,0,-1,1],[1,1,0,-1,0]] # XOR Gates 3D
inputs  = [[0.01,2,-1,0,0],[2,0.01,-1,0,0],[-0.01,2,-1,0,1],[-2,0.01,-1,0,1],[-2,-0.01,-1,1,1],[-0.01,-2,-1,1,1],[0.01,-2,-1,1,0],[2, -0.01,-1,1,0]] #2 inputs
#inputs  = [[0,4,-1,0],[4,0,-1,0],[2.01,2.01,-1,1]] #little range

# def f(w,x): # 2D plane
# 	return int(w[0]*x[0] + w[1]*x[1] + w[2]*x[2] > 0)

# def f(w,x): # 3D plane
# 	return int(w[0]*x[0] + w[1]*x[1] + w[2]*x[2] + w[3]*x[3] > 0)

def f(w,x): #3D plane and 2 inputs
	return [ int(w[0][0]*x[0] + w[1][0]*x[1] + w[2][0]*x[2] > 0), int(w[0][1]*x[0] + w[1][1]*x[1] + w[2][1]*x[2] > 0)]

# def trained(w):
# 	for trials in range(len(inputs)):
# 		x = inputs[trials]
# 		if f(w,x) != x[len(x)-1]:
# 			return False
# 	return True

def trained(w): #2 inputs
	for trials in range(len(inputs)):
		x = inputs[trials]
		if f(w,x) != [x[len(x)-2], x[len(x)-1]]:
			return False
	return True

def verifyNetwork(w, epochs):
	print "Epoches", epochs
	for trials in range(len(inputs)):
		x = inputs[trials]
		print trained(w), f(w,x), x 

def myRandom():
	return random()*2-1

# def trainPerceptionWeights(): #one input
# 	epochs = 0
# 	alpha = 0.1
# 	w = [myRandom() for x in range(len(inputs[0])-1)]
# 	limit = 200000
# 	while(not trained(w) and epochs < limit):
# 		# shuffle(inputs)
# 		for trials in range(len(inputs)):
# 			x = choice(inputs)  #inputs[trials]
# 			t = x[len(x)-1]
# 			y = f(w,x)
# 			w[0] = w[0] - alpha*(y-t)*x[0] 
# 			w[1] = w[1] - alpha*(y-t)*x[1] 
# 			w[2] = w[2] - alpha*(y-t)*x[2]
# 			# w[3] = w[3] - alpha*(y-t)*x[3]
# 		epochs+=1
# 	printline(w)
# 	return w, epochs

def trainPerceptionWeights(): #two inputs
	epochs = 0
	alpha = 0.1
	w = [[myRandom() for x in range(2)] for y in range(len(inputs[0])-1)]
	limit = 200000
	while(not trained(w) and epochs < limit):
		# shuffle(inputs)
		for trials in range(len(inputs)):
			x = choice(inputs)  #inputs[trials]
			t = [x[len(x)-2],x[len(x)-1]]
			y = f(w,x)
			w[0][0] = w[0][0] - alpha*(y[0]-t[0])*x[0] 
			w[1][0] = w[1][0] - alpha*(y[0]-t[0])*x[1] 
			w[2][0] = w[2][0] - alpha*(y[0]-t[0])*x[2]
			w[0][1] = w[0][1] - alpha*(y[1]-t[1])*x[0] 
			w[1][1] = w[1][1] - alpha*(y[1]-t[1])*x[1] 
			w[2][1] = w[2][1] - alpha*(y[1]-t[1])*x[2]
		epochs+=1
	print "m1 =", round(w[0][0]/w[1][0], 2)
	print "b1 =", round(w[2][0]/w[1][0], 2)
	print "m2 =", round(w[0][1]/w[1][1], 2)
	print "b2 =", round(w[2][1]/w[1][1], 2)
	return w, epochs

def printline(w):
	print "m =", round(w[0]/w[1], 2)
	print "b =", round(w[2]/w[1], 2)

w, epochs = trainPerceptionWeights()
verifyNetwork(w, epochs)
print 
test = inputs + [[1,1,-1,0,0],[-1,1,-1,0,1],[-1,-1,-1,1,1],[1,-1-1,1,0]]
verifyNetwork(w, epochs)
