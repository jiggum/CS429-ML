import numpy as np
from parse import get_dataset, parse_2
import cvxopt as co
from random import shuffle
import matplotlib.pyplot as plt

# m : featuers
# n : samples
# matrix : n*m

def k_fold_data(data, k):
    shuffle(data)
    data_set = []
    data_len = len(data)
    for i in range(k):
        cv_part_front = (data_len/k) * i
        cv_part_back = (data_len/k) * (i+1)
        train_data = data[:cv_part_front] + data[cv_part_back:]
        cv_data = data[cv_part_front:cv_part_back]
        data_set.append({ 'train':train_data, 'cv':cv_data})
    return data_set
def fold_data(data, k):
    shuffle(data)
    data_set = []
    data_len = len(data)
    cv_part_front = 0
    cv_part_back = (data_len/5)
    train_data = data[:cv_part_front] + data[cv_part_back:]
    cv_data = data[cv_part_front:cv_part_back]
    data_set.append({ 'train':train_data, 'cv':cv_data})
    return data_set


def kernel_linear(xi, xj):
    return xi.dot(xj)


def svm_training(X,Y,C, kernel):
    def gramian_matrix(X, kernel):
        n,m = X.shape
        G = np.zeros((n,n))
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):

                G[i,j] = kernel(xi,xj)

        return G




    X = np.array(X)
    #Y = np.array(Y)
    n,m = X.shape
    print 'p'

    P = co.matrix(gramian_matrix(X,kernel) * np.outer(Y,Y))
    print 'q'
    q = co.matrix(np.ones(n)*(-1))
    print 'g'
    G_cmp_0 = np.identity(n) * -1
    G_cmp_C = np.identity(n)
    G = co.matrix(np.vstack((G_cmp_0,G_cmp_C)))
    print 'h'
    h_cmp_0 = np.zeros(n)
    h_cmp_C = np.ones(n) * C
    h = co.matrix(np.vstack((h_cmp_0,h_cmp_C)).flatten())
    print 'a'
    A = co.matrix(Y,(1,n))
    print 'b'
    b = co.matrix(np.zeros(1))
    print 'start qp'
    result = co.solvers.qp(P,q,G,h,A,b)['x']
    return result

def get_w_b(X,Y,a,C):
    a = np.array(a)
    X = np.array(X)
    Y = np.array(Y)
    w = np.sum( a*X*Y, axis=0)
    b = 0
    for i, a_i in enumerate(a):
        if a_i > float(C)/100000.0:
            b = 1.0/Y[i] - (np.dot(w,X[i]))
    return w,b

def svm_predict(w,b,X):
    return (np.dot(X,w) +b).astype(int)

def svm_predict_error(data,w,b):
    X = []
    Y = []
    for line in data:
        X.append(line[1:])
        Y.append([float(line[0])])
    print 'start_predict'
    return np.average(np.abs(svm_predict(w,b,X) - Y)/2)

def svm_train(data_set,C, kernel):
    predict_errors = []
    w = range(len(data_set))
    b = range(len(data_set))
    for i, data in enumerate(data_set):
        print 'cv_set ',i+1
        X = []
        Y = []
        X_cv = []
        Y_cv = []
        for line in data['train']:
            X.append(line[1:])
            Y.append([float(line[0])])
        for line in data['cv']:
            X_cv.append(line[1:])
            Y_cv.append([float(line[0])])
        print 'start_train'

        w[i], b[i] = get_w_b(X,Y,svm_training(X,Y, C, kernel),C)
        print 'get_predict_error'
        predict_errors.append(np.average(np.abs(svm_predict(w[i],b[i],X_cv) - Y_cv)/2))

    return w, b, np.mean(predict_errors)


def svm_lenear_test():
    C = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,100,1000]
    predict_errors = []
    predict_errors_test = []

    data =  get_dataset('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a',parse_2)
    test_data =  get_dataset('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t',parse_2)[:1000]
    data_set = k_fold_data(data,5)
    #data_set = fold_data(data,5)


    C_log = np.log(np.array(C)).tolist()

    for c in C:
        w, b ,predict_error = svm_train(data_set,c, kernel_linear)
        predict_errors.append(predict_error)
        predict_error = svm_predict_error(test_data,w[-1],b[-1])
        predict_errors_test.append(predict_error)
    print predict_errors
    print predict_errors_test




    plt.plot(C_log,predict_errors , label = 'cv_error')
    plt.plot(C_log,predict_errors_test, label = 'test_error')
    plt.legend()
    plt.xlabel("log(C)")
    plt.ylabel("predict_errors")
    plt.show()

svm_lenear_test()
"""

for i in range(len(data)):
    print data[i]


    """

