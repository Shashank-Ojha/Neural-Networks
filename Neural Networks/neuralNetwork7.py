'''
+---------------------+
|                     |
|    Shashank Ojha    |
|                     |
+---------------------+
'''
from random import random, choice, shuffle, uniform

TRIALS = 9000
ALPHA = 0.25
INPUTS = [[0,0,-1,0],[0,1,-1,1],[1,0,-1,1],[1,1,-1,0]] 

def yValue(x,w,v):
	h = [0,0,-1]
	h[0] = int( (w[0][0]*x[0] + w[1][0]*x[1] + w[2][0]*x[2]) > 0)
	h[1] = int( (w[0][1]*x[0] + w[1][1]*x[1] + w[2][1]*x[2]) > 0)
	y = int( (v[0]*h[0] + v[1]*h[1] + v[2]*h[2]) > 0)
	return y

def trained(w,v):
	for x in INPUTS:
		y = yValue(x,w,v)
		t = x[3]
		if y != t:
			return False
	return True

def verifyNetwork(epochs, w, v):
	print ("Epochs = ", epochs)
	for x in INPUTS:
		t = x[3]
		y = yValue(x,w,v)
		print ('%5s'% ( y == t ), '-->', yValue(x,w,v), x)
	print("Statistics")
	print(' x = %2d, %2d, %2d, %2d ' %(x[0], x[1], x[2], x[3]))
	print(' w = %5.2f, %5.2f' %(w[0][0], w[0][1]))
	print('     %5.2f, %5.2f' %(w[1][0], w[1][1]))
	print('     %5.2f, %5.2f' %(w[2][0], w[2][1]))
	print(' w = %5.2f, %5.2f, %5.2f' %(v[0], v[1], v[2]))
	print (' y =', y)

def initializeWeights():
	w = [[myRand(),myRand()],[myRand(),myRand()], [myRand(), myRand()]]
	v = [myRand(),myRand(),myRand()]
	return w, v

def myRand():
	return uniform(-1, 10)

def improveWeights(w,v):
	w = [[myRand(),myRand()],[myRand(),myRand()], [myRand(), myRand()]]
	v = [myRand(),myRand(),myRand()]
	return w,v

def trainNetwork():
	epochs = 0
	h = [0,0,-1]
	w, v = initializeWeights()
	while epochs < TRIALS and not trained(w,v):
		x = choice(INPUTS)
		w, v = improveWeights(w,v)
		h[0] = int((w[0][0]*x[0] + w[1][0]*x[1] + w[2][0]*x[2]) > 0)
		h[1] = int((w[0][1]*x[0] + w[1][1]*x[1] + w[2][1]*x[2]) > 0)
		epochs += 1
	return epochs, w , v

epochs, w, v = trainNetwork()
verifyNetwork(epochs, w, v)


