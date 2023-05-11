import random

def transpose(X):
    return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

def s_mult(X,scalar):
    return [[x*scalar for x in j] for j in X]

def subtract(X,Y):
    output = X
    for i in range(len(X)-1):
        for j in range(len(X[0])-1):
            output[i][j] = X[i][j]-Y[i][j]
    return output

def add(X,Y):
    output = zero_matrix(get_cols(X),get_rows(X))
    for i in range(get_cols(X)):
        for j in range(get_rows(X)):
            output[i][j] = X[i][j]+Y[i][j]
    return output

def v_m(X):
    return[[i] for i in X]

def m_v(X):
    return[i[0] for i in X]

def hadamard(X,Y):
    a,b = m_v(X),m_v(Y)
    try:
        return v_m([x[0]*x[1] for x in zip(a,b)])
    except:
        raise Exception("Oops! (Hadamard)")

def dot(X,Y):
        try:
            return [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
        except:
            print('a')
            raise Exception("Wrong Matrix Shape")

def random_matrix(a,b):
    return [[random.random() for i in range(b)] for j in range(a)]

def zero_matrix(a,b):
    return [[0 for i in range(b)]for j in range(a)]

def get_shape(a):
    return [len(a),len(a[0])]

def get_cols(a):
    return len(a)

def get_rows(a):
    return len(a[0])
