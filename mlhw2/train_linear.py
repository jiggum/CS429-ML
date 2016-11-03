import numpy as np
from parse import get_dataset
from random import shuffle

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