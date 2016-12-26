'''
+---------------------+
|                     |
|    Shashank Ojha    |
|                     |
+---------------------+
'''
from random import random, choice, shuffle, uniform
from math import exp


TRIALS = 30000
ALPHA = 0.5
INPUTS = [[1,0,0,0,0,0,0,0,-1], [0,1,0,0,0,0,0,0,-1], [0,0,1,0,0,0,0,0,-1], [0,0,0,1,0,0,0,0,-1], [0,0,0,0,1,0,0,0,-1], [0,0,0,0,0,1,0,0,-1],
			[0,0,0,0,0,0,1,0,-1], [0,0,0,0,0,0,0,1,-1]]


def f(x):
	return 1.0/(1 + exp(-x))

def copyArrayAndRound(array):
	copy = []
	for x in range(len(array)):
		copy.append(round(array[x],1))
	return copy

def initializeWeights():
	w = [[myRand() for col in range(3)] for row in range(9)] #9 x 3 matrix
	v = [[myRand() for col in range(8)] for row in range(4)] #4 x 8 matrix
	#
	# w = [[ 0.1, -0.2,  0.3],
	# 	 [ 0.4,  0.1,  0.1],
	# 	 [ 0.3,  0.3,  0.7],
	# 	 [ 0.6,  0.7, -0.8],
	# 	 [-0.5,  0.4,  0.2],
	# 	 [ 0.1,  0.5,  0.5],
	# 	 [ 0.4,  0.6,  0.3],
	# 	 [ 0.2,  0.9, -0.1],
	# 	 [-0.4, -0.1,  0.6]]

	# v = [[ 0.4, -0.8,  0.2,  0.6,  0.7, -0.5,  0.2, -0.4],
	# 	 [ 0.9,  0.7,  0.3,  0.2, -0.1,  0.6,  0.4, -0.8],
	# 	 [ 0.1,  0.2,  0.7, -0.8,  0.5,  0.4,  0.1,  0.1],
	# 	 [ 0.9,  0.1,  0.8, -0.4,  0.2,  0.7, -0.3,  0.8]]

	return w, v

def myRand():
	return uniform(-1, 1) #random float from [-1,1]

def mult(v, m):
	assert len(v) == len(m) # [len(v), len(m)]
	return [sum([v[i]*m[i][j] for i in range(len(v))]) for j in range(len(m[0]))]

def trained(w,v):
	for x in INPUTS:
		h, y = feedForward(x,w,v)
		copy = copyArrayAndRound(y)
		t = x[:len(x)-1]
		if copy != t:
			return False
	return True

def trainNetwork():
	epochs = 0
	w, v = initializeWeights()
	while(epochs < TRIALS and not trained(w,v)):
		for x in INPUTS:
			h, y = feedForward(x,w,v)
			# printAllData(w,h,v,y,x)
			w, v = backPropagation(x, w, v, h, y)
		epochs += 1
		if epochs%100 == 0:
			print error(y, x)
	print "epochs =", epochs 
	# print "x = " , INPUTS[0]
	# a,b,c,y =feedForward(INPUTS[0], w, v)
	# print "y = " , y
	return epochs, w, v

def backPropagation(x, w, v, h, y):
	deltaV = []
	for i in range(len(y)): ## 8
		fprime = y[i]*(1-y[i])
		deltaT = y[i] - x[i]
		deltaV.append(deltaT*fprime)
	######################
	deltaW = []
	for i in range(len(h)-1): ##3
		partial = sum([deltaV[j]*v[i][j] for j in range(len(y))])
		fprime = h[i]*(1-h[i])
		deltaW.append( partial * fprime) 


	for j  in range(len(v)):
		for k in range(len(v[0])):
			v[j][k] = v[j][k] - ALPHA*(deltaV[k]*h[j])

	for j in range(len(w)):
		for k in range(len(w[0])):
			w[j][k] = w[j][k] - ALPHA*(deltaW[k]*x[j])

	return w, v


def feedForward(x, w, v):
	dp1 = mult(x, w)

	h = []
	for val in dp1:
		h.append(f(val))
	h = h + [-1]

	dp2 = mult(h, v)

	y = []
	for val in dp2:
		y.append(f(val))

	return h, y

def error(y, t):
	val = 0
	for i in range(len(y)):
		val += 0.5*(y[i]-t[i])**2
	val = round(val,3)
	return val

def verifyNetwork(epochs, w, v):
	print ("Epochs = ", epochs)
	for x in INPUTS:
		t = x[:len(x)-1]
		h, y = feedForward(x,w,v)
		copy = copyArrayAndRound(y)
		print ('%5s'% ( copy == t ), '-->', copy, t)

def printAllData(w,h,v,y,x):
	print "x = "
	for cell in INPUTS:
		print(cell)
	print '='*50

	print "w = "
	for row in w:
		for cell in row:
			print(cell),
		print
	print '='*50

	print "h = "
	for cell in h:
		print(cell)
	print '='*50

	print "v = "
	for row in v:
		for cell in row:
			print cell,
		print
	print '='*50

	print "y = "
	for cell in y:
		print(cell)
	print '='*50

	print "E = "
	print error(y, x)

epochs, w, v = trainNetwork()
verifyNetwork(epochs, w, v)

