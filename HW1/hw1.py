import numpy as np
import requests
import re
from random import shuffle
import matplotlib.pyplot as plt
def get_dataset(url,parse_func):
    source = requests.get(url)
    data = []
    if(source):
        data = parse_func(source.text)
    return data

def parse_2(source_txt):
    data = []
    data_pre = source_txt.strip().split('\n')
    data = [[element for element in re.split('\s',line.strip())] for line in data_pre]
    for line in data:
        for i in range(len(line)):
            line[i] = int(line[i].replace(":1",""))
    c_data = []
    for line in data:
        c_line = []
        for i in range(130):
            c_line.append(0)
        c_line[0] = line[0]
        for i in line:
            c_line[i] = 1;
        c_data.append(c_line)
    return c_data

def parse_1(source_txt):
    data = []
    data_pre = source_txt.strip().split('\n')
    data = [[float(element) for element in re.split('\s.{0,2}:',line)] for line in data_pre]
    return data

def LS_analytic(url,parse_func):
    def compute_cost(x,y,theta):
        J = 0
        J = np.average(np.square(np.subtract(np.dot(x,theta) ,y)))
        return J
    data = get_dataset(url,parse_func)
    shuffle(data)
    num_input = len(data)
    train_data = data[0:int(num_input * 0.8)]
    test_data = data[int(num_input * 0.8):]
    y = [line[0] for line in train_data]
    x = [line[1:] for line in train_data]
    x_t = np.transpose(x)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_t,x)), x_t),y)
    test_y = [line[0] for line in test_data]
    test_x = [line[1:] for line in test_data]
    Y = np.dot(test_x, theta)
    prediction_error =  np.average(np.fabs(np.subtract(Y ,test_y)))
    return (Y,prediction_error,compute_cost(x,y,theta));

def LS_gradient(url, parse_func, num_iters,step_size):
    def compute_cost(x,y,theta):
        J = 0
        J = np.average(np.square(np.subtract(np.dot(x,theta) ,y)))
        return J
    data = get_dataset(url,parse_func)
    shuffle(data)
    num_input = len(data)
    num_param = len(data[0])-1
    train_data = data[0:int(num_input * 0.8)]
    test_data = data[int(num_input * 0.8):]
    y = [line[0] for line in train_data]
    x = [line[1:] for line in train_data]
    num_train = len(train_data)
    test_y = [line[0] for line in test_data]
    test_x = [line[1:] for line in test_data]
    J_temp = []
    theta = np.zeros((num_param,1))
    for i in range(num_iters):
        ph1 = np.subtract(np.transpose(np.dot(x,theta)), y)
        ph2 = np.dot(ph1,x)
        theta_delta = np.divide(np.transpose(ph2),(num_input/2))
        theta = np.subtract(theta,  np.multiply(theta_delta,step_size))
        J_temp.append(compute_cost(x,y,theta));
    Y = np.dot(test_x, theta)
    prediction_error =  np.average(np.fabs(np.subtract(Y ,test_y)))
    return (Y,prediction_error,J_temp)

def LG_gradient(url, parse_func, num_iters,step_size):
    def sigmoid(x):
        return np.divide(1,np.add(1,np.exp(np.dot(-1,x))))
    def compute_cost(x,y,theta):
        num = x.shape[0]
        ph_0 = np.log(sigmoid(np.dot(x,theta)))
        ph_1 = np.dot(np.transpose(-1*y) , ph_0)
        ph_2 = np.dot(np.transpose(1-y) , np.log(1-sigmoid(np.dot(x,theta))))
        J = (1./num) * (ph_1 - ph_2)
        return J
    def compute_grad(x,y,theta):
        num = x.shape[0]
        ph_1 = np.transpose(sigmoid(np.transpose(np.dot(x,theta))) - y)

        return np.dot(np.transpose(x), ph_1 )
    data = get_dataset(url,parse_func)
    shuffle(data)
    num_input = len(data)
    num_param = len(data[0])-1
    train_data = data[0:int(num_input * 0.8)]

    test_data = data[int(num_input * 0.8):]
    y_pre = [line[0] for line in train_data]
    for i in range(len(y_pre)):
        if y_pre[i] <0:
            y_pre[i] = 0
    y = np.array(y_pre)
    x =  np.array([line[1:] for line in train_data])
    num_train = len(train_data)
    test_y_pre = [line[0] for line in test_data]
    for i in range(len(test_y_pre)):
        if test_y_pre[i] <0:
            test_y_pre[i] = 0
    test_y = np.array(test_y_pre)
    test_x =  np.array([line[1:] for line in test_data])
    J_temp = []
    theta = np.zeros((num_param,1))
    for i in range(num_iters):
        theta_delta = compute_grad(x,y,theta)
        theta = np.subtract(theta,  np.multiply(theta_delta,step_size))
       # for i in np.dot(x,theta):
            #print i
        J_temp.append((compute_cost(x,y,theta)[0],step_size));
    Y = sigmoid(np.dot(test_x, theta))
    for i in range(len(Y)):
        if Y[i] < 0.5:
            Y[i] = 0
        else:
            Y[i] = 1
    prediction_error =  np.average(np.fabs(np.transpose(Y) - np.transpose(test_y)))
    return (test_y_pre,Y,prediction_error,J_temp)

def LG_gradient_back(url, parse_func, num_iters,step_size_init, alpha, beta):
    step_size = step_size_init;
    def sigmoid(x):
        return np.divide(1,np.add(1,np.exp(np.dot(-1,x))))
    def compute_cost(x,y,theta):
        num = x.shape[0]
        ph_0 = np.log(sigmoid(np.dot(x,theta)))
        ph_1 = np.dot(np.transpose(-1*y) , ph_0)
        ph_2 = np.dot(np.transpose(1-y) , np.log(1-sigmoid(np.dot(x,theta))))
        J = (1./num) * (ph_1 - ph_2)
        return J
    def compute_grad(x,y,theta):
        num = x.shape[0]
        ph_1 = np.transpose(sigmoid(np.transpose(np.dot(x,theta))) - y)

        return np.dot(np.transpose(x), ph_1 )
    data = get_dataset(url,parse_func)
    shuffle(data)
    num_input = len(data)
    num_param = len(data[0])-1
    train_data = data[0:int(num_input * 0.8)]

    test_data = data[int(num_input * 0.8):]
    y_pre = [line[0] for line in train_data]
    for i in range(len(y_pre)):
        if y_pre[i] <0:
            y_pre[i] = 0
    y = np.array(y_pre)
    x =  np.array([line[1:] for line in train_data])
    num_train = len(train_data)
    test_y_pre = [line[0] for line in test_data]
    for i in range(len(test_y_pre)):
        if test_y_pre[i] <0:
            test_y_pre[i] = 0
    test_y = np.array(test_y_pre)
    test_x =  np.array([line[1:] for line in test_data])
    J_temp = []
    theta = np.zeros((num_param,1))
    for i in range(num_iters):
        theta_delta = compute_grad(x,y,theta)
        while(compute_cost(x,y,np.subtract(theta,  np.multiply(theta_delta,step_size)))[0] >= compute_cost(x,y,theta)[0] ):
            step_size = step_size * beta
        theta = np.subtract(theta,  np.multiply(theta_delta,step_size))
        J_now = compute_cost(x,y,theta)[0]
        J_temp.append((J_now,step_size))
    Y = sigmoid(np.dot(test_x, theta))
    for i in range(len(Y)):
        if Y[i] < 0.5:
            Y[i] = 0
        else:
            Y[i] = 1
    prediction_error =  np.average(np.fabs(np.transpose(Y) - np.transpose(test_y)))
    return (test_y_pre,Y,prediction_error,J_temp)

def LS_analytic_test():
    LS_analytic_error_ave = [];
    for i in range(10):
        LS_analytic_error_ave . append(LS_analytic('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1)[1])
    print LS_analytic_error_ave
    print reduce(lambda x, y: x+y , LS_analytic_error_ave) / len (LS_analytic_error_ave)

def LS_gradient_test():
    LS_gradient_error_ave = [];
    for i in range(10):
        LS_gradient_error_ave.append(LS_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1,1000, 0.3)[1])
    print LS_gradient_error_ave
    return reduce(lambda x, y: x+y , LS_gradient_error_ave) / len (LS_gradient_error_ave)


def LG_gradient_test():
    LG_gradient_error_ave = [];
    for i in range(10):
        LG_gradient_error_ave.append(LG_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,100, 0.001)[2])
    print LG_gradient_error_ave
    return reduce(lambda x, y: x+y , LG_gradient_error_ave) / len (LG_gradient_error_ave)

def LG_gradient_back_test():
    LG_gradient_error_ave = [];
    for i in range(10):
        LG_gradient_error_ave.append(LG_gradient_back('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,100, 0.001, 0.5,0.9)[2])
    print LG_gradient_error_ave
    return reduce(lambda x, y: x+y , LG_gradient_error_ave) / len (LG_gradient_error_ave)


"""
LG_gradient_test()
LG_gradient_back_test()

(a,b,c,d) =  LG_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,100, 0.001)
print c
print d

(a,b,c,d) =  LG_gradient_back('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,100, 0.01, 0.5,0.9)
print c
print d


ofv = LG_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,1000, 0.001)[3]
ofv_back = LG_gradient_back('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2,1000, 0.001, 0.3,0.8)[3]
plt.plot(range(len(ofv)),ofv, label="fixed")
plt.plot(range(len(ofv_back)),ofv_back, label="backtracking")
plt.legend()
plt.show()

ofv = LS_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1,100, 0.005)[2]
ofv_back_val = LS_analytic('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1)[2]
ofv_back = []
for i in range(len(ofv)):
    ofv_back.append(ofv_back_val)
plt.plot(range(len(ofv)),ofv)
plt.plot(range(len(ofv)),ofv_back)
plt.show()
LS_analytic('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a',parse_2)[1]

for i in [0.1,0.01, 0.001,0.0001]:
    ofv = LS_gradient('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1,1000, i)[2]
    plt.plot(range(len(ofv)),ofv, label = str(i))
ofv_back_val = LS_analytic('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/housing_scale',parse_1)[2]
ofv_back = []
for i in range(len(ofv)):
    ofv_back.append(ofv_back_val)
plt.plot(range(len(ofv)),ofv_back,label = "optimal")
plt.legend()
plt.show()
"""
