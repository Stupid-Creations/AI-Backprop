from matrix import *

class n_layer:
    def __init__(self,a,b):
        self.i_l = a
        self.o_l = b  
        self.weights = random_matrix(self.o_l,self.i_l)
        self.bias = random_matrix(self.o_l,1)
    def activate(self,m):
        self.m = m
        self.z = add(dot(self.weights,self.m),self.bias)
        self.pred = sigmoid_matrix(self.z)


class neural_net:
    def __init__(self,dim):
        self.brain = []
        for i in range(len(dim)-1):
            self.brain.append(n_layer(dim[i],dim[i+1]))
            
    def activate(self,m):
        self.brain[0].activate(m)
        for i in range(1,len(self.brain)):
            self.brain[i].activate(self.brain[i-1].pred)
        self.pred = self.brain[-1].pred
        self.z = self.brain[-1].z

    def backprop(self,answer):
        correct_answer = answer
        weights = []
        biases  = []
        self.cost = subtract(self.pred,correct_answer)
        delta = hadamard(self.cost,sigmoid_prime(self.z))
        biases.append(delta)
        weights.append(dot(delta,transpose(self.brain[-1].m)))
        
        for i in range(len(self.brain)-2,-1,-1):
            delta = hadamard(dot(transpose(self.brain[i+1].weights),delta),sigmoid_prime(self.brain[i].z))
            biases.append(delta)
            weights.append(dot(delta,transpose(self.brain[i].m)))

        return weights,biases
        # This returns the gradients of all the weights and biases in case stochastic gradient descent is not used, and a sum of the these values is needed

sigmoid = lambda x: 1/(1+(2.7182818284**(float(-x))))
sig_p = lambda x: (sigmoid(x)*(1-sigmoid(x)))
sigmoid_matrix = lambda x: [[sigmoid(y)for y in j]for j in x]
sigmoid_prime = lambda x:[[sig_p(y)for y in j]for j in x]
