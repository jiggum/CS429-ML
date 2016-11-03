import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import multivariate_normal as normal
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
ax = plt.gca()

#mlab.normpdf(x,mu,sig)

def draw_gmm(n, max_x_i):
    x = []
    y = []
    mean = []
    sigma = []
    for i in range(int(n*1)):
        c=np.random.rand(3,)
        size = random.randrange(10,max_x_i)
        mean.append([random.randrange(-50,50),random.randrange(-50,50)])
        sigma.append( [[size,0],[0,size]])
        x_buffer, y_buffer = np.random.multivariate_normal(mean[i],sigma[i], random.randrange(10,max_x_i)).T
        x += x_buffer.tolist()
        y += y_buffer.tolist()
    plt.plot(x,y, 'x',color='b')
    plt.axis('equal')
    return len(x),x,y,mean,sigma

def draw_ellipse(sigma,color,r):
    ax = plt.gca()
    val, vec = np.linalg.eigh(sigma)
    width, height = 2 * np.sqrt(val[:, None] * r)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))
    ellipse = Ellipse(mean[i], width=width, height=height, angle=rotation,color=color, fill=False)
    ax.add_artist(ellipse)

def predicted_group(X,gamma):
    n, m = gamma.shape

    ret_X = range(m)
    for i in range(len(ret_X)):
        ret_X[i] = []
    for i in range(n):
        predicted_group_number = 0
        for j in range(m):
            if gamma[i,predicted_group_number] < gamma[i,j]:
                predicted_group_number = j
        ret_X[predicted_group_number].append(X[i])
    return ret_X











n=6
N_all ,x,y, mean_actual, sigma_actual=draw_gmm(n, 500)
plt.plot(np.array(mean_actual)[:,0],np.array(mean_actual)[:,1], 'o',color='r')
print 'mean :',mean_actual
print 'sigma : ', sigma_actual
X = np.vstack((x,y)).T

mean = np.zeros((n,2))
sigma = np.ones(n*4).reshape((n,2,2))
N = np.zeros(n)
pie = np.zeros(n)
gamma = np.zeros((N_all,n))

##set random n cluster

color_list = []
for i in range(n):
    color_list.append(np.random.rand(3,))
    size = random.randrange(10,50)
    sigma[i] = [[size,0],[0,size]]
    mean[i] = [random.randrange(-50,50),random.randrange(-50,50)]
    N[i] = int(N_all/n)
    pie[i] = 1.0/n
    draw_ellipse(sigma[i],color_list[i],3);

N[0] = N_all - np.sum(N[1:])
plt.show()
kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
kmean_center = np.array(kmeans.cluster_centers_)
for iter in range(300):
    print 'iter: ', iter
    #e step
    for i in range(N_all):
        for j in range(n):
            normal_val = normal.pdf(X[i],mean[j],sigma[j])
            upper = pie[j]*normal_val#sigma[j])
            under = 0
            for k in range(n):
                under += pie[k]*normal.pdf(X[i],mean[k],sigma[k])
            gamma[i,j] = upper/under

    #m step
    for j in range(n):
        N[j] = np.sum(gamma[:,j])
        pie[j] = N[j]/N_all
        mean[j] =  np.sum( np.vstack((gamma[:,j],gamma[:,j])).T * X,axis=0)/N[j]
        sigma_buffer = 0
        for i in range(N_all):
            sigma_buffer += gamma[i,j]* np.dot((X[i] - mean[j]).reshape(2,1),(X[i] - mean[j]).reshape(1,2))
        sigma[j] = sigma_buffer / N[j]

    #draw ellipse
    predicted_X = predicted_group(X,gamma)
    plt.plot(np.array(mean_actual)[:,0],np.array(mean_actual)[:,1], 'o',color='r')
    for i in range(n):
        draw_ellipse(sigma[i],color_list[i],3)
        if predicted_X[i]:
            plt.plot(np.array(predicted_X[i])[:,0],np.array(predicted_X[i])[:,1], 'x',color=color_list[i])

    if(iter>10 and iter%20!=0):
        continue
    plt.plot(np.array(mean)[:,0],np.array(mean)[:,1], 'o',color='b')
    plt.plot(kmean_center[:,0], kmean_center[:,1], 'o', color='y')
    plt.axis('equal')
    plt.show()
    print 'mean_predict : ', mean
    print 'sigma_predict : ', sigma


#plt.plot(x,y, 'x',color='r')


#gamma
#mean
#sigma
#pie
#N
