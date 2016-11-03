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


def kernel_gausian(xi, xj, sigma):
    try:
        temp = np.linalg.norm(xi-xj)
        return np.exp(-np.sqrt(temp**2 / (2*sigma**2)))
    except:
        print "------------error norm:"  , np.linalg.norm(xi-xj), "sigma: ",sigma
def svm_training(X,Y,C, kernel, sigma):
    def gramian_matrix(X, kernel, sigma):
        n,m = X.shape
        G = np.zeros((n,n))
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                if sigma:
                    G[i,j] = kernel(xi,xj, sigma)
                else:
                    G[i,j] = kernel(xi,xj)
        return G
    X = np.array(X)
    #Y = np.array(Y)
    n,m = X.shape
    print 'p'

    P = co.matrix(gramian_matrix(X,kernel,sigma) * np.outer(Y,Y))
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


def get_bias(X,Y,a,sigma):
    return np.mean(svm_predict(a,X,Y,X,sigma) )
def get_SV(X,Y,a,C):
    a = np.array(a)
    X = np.array(X)
    Y = np.array(Y)
    ret = []
    for i, a_i in enumerate(a):
        if a_i > float(C)/100000.0:
            ret.append((X[i],Y[i]))
    return ret

def svm_predict(a,x_train,y_train,X, sigma, bias = 0):
    ret = []
    for n in range(len(X)):
        pred = bias
        temp = a*y_train
        for m in range(len(a)):
                pred += temp[m]*kernel_gausian(np.array(x_train[m]),np.array(X[n]), sigma)
        ret.append(pred)

    return ret
def svm_predict_error(data,x_train,y_train,a, sigma, bias=0):
    X = []
    Y = []
    print bias
    for line in data:
        X.append(line[1:])
        Y.append([float(line[0])])
    X = np.array(X)
    predict = svm_predict(a,x_train,y_train,X, sigma, bias)
    return np.mean(1 - np.sign(np.array(predict) * Y))
def svm_train(data_set,C, kernel, sigma):
    a_list = []
    for i, data in enumerate(data_set):
        print 'cv_set ',i+1, ' c:', C,' sigma:',sigma
        X = []
        Y = []
        for line in data['train']:
            X.append(line[1:])
            Y.append([float(line[0])])
        X= np.array(X)
        Y= np.array(Y)
        print 'start_train'
        a_list.append(svm_training(X,Y, C, kernel, sigma))

    return a_list



def svm_gausian_test():
    #C = [0.00001,0.0001, 0.001,0.01,0.1,1,10]
    C = [0.1]
    sigma = [10]
    predict_errors = []
    predict_errors_test = []

    data =  get_dataset('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a',parse_2)

    test_data =  get_dataset('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t',parse_2)[5000:7000]
    data_set = k_fold_data(data,5)[:3]
    #data_set = fold_data(data,5)



    C_log = np.log(np.array(C)).tolist()
    sigma_log = np.log(np.array(sigma)).tolist()
    w=0
    b=0

    for c in C:
        for s in sigma:

            a = []
            a = svm_train(data_set,c, kernel_gausian, s)
            predict_error_parts = []
            for i, data in enumerate(data_set):

                X = []
                Y = []
                for line in data['train']:
                    X.append(line[1:])
                    Y.append([float(line[0])])
                X = np.array(X)
                Y = np.array(Y)
                bias = get_bias(X,Y,a[i],s)
                predict_error = svm_predict_error(data['cv'],X,Y,a[i],s, bias)
                predict_error_parts.append(predict_error)
                print 'predict_error: ', predict_error
            predict_errors.append(np.mean(predict_error_parts))
            a = a[-1]
            X = []
            Y = []
            for line in data_set[-1]['train']:
                X.append(line[1:])
                Y.append([float(line[0])])
            X = np.array(X)
            Y = np.array(Y)
            bias = get_bias(X,Y,a,s)
            predict_error = svm_predict_error(test_data,X,Y,a,s, bias)
            predict_errors_test.append(predict_error)

    print predict_errors
    print predict_errors_test




    plt.plot(C_log,  predict_errors, label = 'cv_error')
    #plt.plot(C_log,predict_errors_test, label = 'test_error')
    plt.legend()
    plt.xlabel("log(C)")
    plt.ylabel("predict_errors")
    plt.show()
svm_gausian_test()
"""

for i in range(len(data)):
    print data[i]


    """

