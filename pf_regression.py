#09-27
#Giuseppina
# pf on regression problem
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(0)
def relu(X):
    return np.maximum(X, 0)

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def indeces(weights,N):
    ind=np.random.choice(np.arange(N), N, p=weights)
    return ind

def resample_like(like,N):
    u=like/np.sum(like)
    u=u.flatten()
    old=u
    Nf=1. / np.sum(np.square(u))
    if Nf <N/2:
        print('resampling')
        ind=indeces(u,N)
        u=(1/N)*np.ones(N) 
    else:
        ind= np.arange(N)
        u=old
    return u, ind

def square_loss(y,y_pred):
    m=y.shape[0]
    residuals = y_pred - y
    cost = np.sum((residuals ** 2)) / (2 * m)
    RMSE=np.sqrt(cost)
    return cost, RMSE

def likelihood(y,y_pred):
    m=y.shape[0]
    N=y_pred.shape[0]
    y=np.ones([N, 1, 1]) * y
    residuals = y_pred - y
    like = -np.sum((residuals ** 2),axis=-2) / (2 * m)
    return like

def feed_forward(X, params):
    cache = {}

    cache["Z1"] = np.matmul(X, params["W1"]) + params["b1"]
    cache["A1"] = relu(cache["Z1"])
    cache["Z2"] = np.matmul( cache["A1"], params["W2"]) + params["b2"]
    cache["A2"] = relu(cache["Z2"])
    cache["Y_pred"] = np.matmul(cache["A2"] , params["W3"]) + params["b3"]
    return cache

def forward_1(X,Y,N,init_sigma, params):

    part_layer1 = {}
    like_1= {}

    feat=params["W1"].shape[0]   #feature=input size
    hidden=params["W1"].shape[1]  # hidden layer size

    partW = params["W1"] .flatten()
    partW =  np.random.multivariate_normal(partW, init_sigma*np.eye(hidden*feat), size=N) #dim=[#particles,n_hidden*num_features]
    part_layer1["pf_W1_prior"]=partW.reshape((N,feat,hidden))
    
    partb = params["b1"] .flatten()
    partb =  np.random.multivariate_normal(partb, init_sigma*np.eye(hidden), size=N)
    part_layer1["pf_b1_prior"]= partb.reshape([N,1,hidden])

    X=np.ones([N, 1, 1]) * X
    Z1 = np.matmul(X, part_layer1["pf_W1_prior"]) + part_layer1["pf_b1_prior"]
    A1 = relu(Z1)
    W2=np.ones([N, 1, 1]) * params["W2"]
    b2=np.ones([N, 1, 1]) * params["b2"]
    Z2 = np.matmul( A1,W2 ) + b2
    A2 = relu(Z2)
    W3=np.ones([N, 1, 1]) * params["W3"]
    b3=np.ones([N, 1, 1]) * params["b3"]
    y_pred= np.matmul( A2, W3) + b3 #dim=[N, m, 1]
    like_1["original"]= likelihood(Y, y_pred) #shape Y=[m,1] lik=[N,1]
    like_1["new"], indeces =resample_like(like_1["original"],N)
    part_layer1["pf_W1"]= np.take(partW,indeces,axis=0)  # shape=[N,feat*hidd]
    part_layer1["pf_b1"]= np.take(partb,indeces,axis=0) 
    return part_layer1, like_1

def forward_2(X,Y,N,init_sigma, params):

    part_layer2 = {}
    like_2= {}
    hidden1=params["W1"].shape[1]  # hidden layer1 size
    hidden2=params["W2"].shape[1]  # hidden layer2 size
    
    partW = params["W2"] .flatten()
    partW =  np.random.multivariate_normal(partW, init_sigma*np.eye(hidden1*hidden2), size=N) #dim=[#particles,n_hidden1*n_hidden2]
    part_layer2["pf_W2_prior"]=partW.reshape((N,hidden1,hidden2))
    
    partb = params["b2"] .flatten()
    partb =  np.random.multivariate_normal(partb, init_sigma*np.eye(hidden2), size=N)
    part_layer2["pf_b2_prior"]= partb.reshape([N,1,hidden2])

    Z1 = np.matmul( X, params["W1"]) + params["b1"]
    A1 = relu(Z1)
    Z2 = np.matmul( A1,part_layer2["pf_W2_prior"]) + part_layer2["pf_b2_prior"]
    A2 = relu(Z2)  
    y_pred= np.matmul( A2, params["W3"]) + params["b3"] #dim=[N, m, 1]
    like_2["original"]= likelihood(Y, y_pred) #shape Y=[m,1] lik=[N,1]
    like_2["new"], indeces =resample_like(like_2["original"],N)
    part_layer2["pf_W2"]= np.take(partW,indeces,axis=0)  # shape=[N,hidd1*hidd2]
    part_layer2["pf_b2"]= np.take(partb,indeces,axis=0)
    return part_layer2, like_2


def forward_3(X,Y,N,init_sigma, params):

    part_layer3 = {}
    like_3= {}
    hidden2=params["W2"].shape[1]  # hidden layer2 size
    
    partW = params["W3"] .flatten()
    partW =  np.random.multivariate_normal(partW, init_sigma*np.eye(hidden2), size=N) #dim=[#particles,n_hidden1*n_hidden2]
    part_layer3["pf_W3_prior"]=partW.reshape((N,hidden2,1))
    
    partb = params["b3"] .flatten()
    partb =  np.random.multivariate_normal(partb, init_sigma*np.eye(1), size=N)
    part_layer3["pf_b3_prior"]= partb.reshape([N,1,1])

    Z1 = np.matmul( X, params["W1"]) + params["b1"]
    A1 = relu(Z1)
    Z2 = np.matmul( A1,params["W2"]) + params["b2"]
    A2 = relu(Z2)  
    y_pred= np.matmul( A2, part_layer3["pf_W3_prior"]) + part_layer3["pf_b3_prior"] #dim=[N, m, 1]
    like_3["original"]= likelihood(Y, y_pred) #shape Y=[m,1] lik=[N,1]
    like_3["new"], indeces =resample_like(like_3["original"],N)
    part_layer3["pf_W3"]= np.take(partW,indeces,axis=0)  # shape=[N,hidd1*hidd2]
    part_layer3["pf_b3"]= np.take(partb,indeces,axis=0)
    return part_layer3, like_3

def back_propagate(X, Y, params, cache):
    m=Y.shape[0]
    dY=(cache["Y_pred"] - Y)/m
    dW3 = (1./m) * np.matmul(cache["A2"].T,dY)
    db3 = (1./m) * np.sum(dY, axis=0, keepdims=True)
    dA2 = np.matmul( dY, params["W3"].T)
    dZ2 = dA2 *dRelu(cache["Z2"])
    
    dW2 = (1./m) * np.matmul(cache["A1"].T,dZ2)
    db2 = (1./m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.matmul( dZ2,params["W2"].T)
    
    dZ1 = dA1 *dRelu(cache["Z1"])
    dW1 = (1./m) * np.matmul( X.T, dZ1)
    db1 = (1./m) * np.sum(dZ1, axis=0, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return grads

#########################################
#data creation
def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return 10 * np.sin(2 * np.pi * (x)) + epsilon

train_size = 1000
noise = 1.0

X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1) # size=[#points,1]
Y = f(X, sigma=noise) #size =[#points,1]
y_true = f(X, sigma=0.0)
###################################################
#parameters to set up
n_x = X.shape[1] # input features size
n_h1 = 30 # hideen layer 1
n_h2 = 20 # hideen layer 2
learning_rate = 0.08
beta = .9
# particles for each layer
N1=100
N2=100
N3=100
sigma=0.01
# where to save files:
date="09_28"
folder='Results_PF/{}'.format(date)


def initialization(folder, continued=False):
    if continued==True:
        params = {}
        file1 = open('./{}/particles.pkl'.format(folder), 'rb')        
        part_layer1, part_layer2,part_layer3, like_1, like_2,like_3=   pickle.load(file1) 
        file1.close()
        params["W1"]=np.mean(part_layer1["pf_W1"],axis=0)
        params["b1"]=np.mean(part_layer1["pf_b1"],axis=0)
        params["W2"]=np.mean(part_layer2["pf_W2"],axis=0)
        params["b2"]=np.mean(part_layer2["pf_b2"],axis=0)
        params["W3"]=np.mean(part_layer3["pf_W3"],axis=0)
        params["b3"]=np.mean(part_layer3["pf_b3"],axis=0)
    else:
        params = { "W1": 0.01*np.random.randn(X.shape[1], n_h1),
           "b1": np.zeros([1,n_h1]),
           "W2": 0.01*np.random.randn(n_h1,n_h2),
           "b2": np.zeros([1,n_h2]),
            "W3": 0.01*np.random.randn(n_h2,1),
           "b3": np.zeros([1,1])}
    return params


init_fold='./{}/100_total'.format(folder)
params=initialization(init_fold,continued=False)

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)
V_dW3 = np.zeros(params["W3"].shape)
V_db3 = np.zeros(params["b3"].shape)

loss_train_history=[]
RMSE_train_history =[]

# train
for i in range(1000):

    part_layer1, like_1= forward_1(X,Y,N1,sigma, params)
    W1= np.average(part_layer1["pf_W1"], weights=like_1["new"], axis=0)    #dimention= [n_hidden*num_features,]
    W1=W1.reshape((X.shape[1], n_h1))
    params["W1"]=W1
    b1=np.average(part_layer1["pf_b1"], weights=like_1["new"], axis=0) 
    b1= b1.reshape((1,n_h1))
    params["b1"]=b1

    part_layer2, like_2= forward_2(X,Y,N2,sigma, params)
    W2= np.average(part_layer2["pf_W2"], weights=like_2["new"], axis=0)    #dimention= [n_hidden*num_features,]
    W2=W2.reshape((n_h1, n_h2))
    params["W2"]=W2
    b2=np.average(part_layer2["pf_b2"], weights=like_2["new"], axis=0) 
    b2= b2.reshape((1,n_h2))
    params["b2"]=b2

    part_layer3, like_3= forward_3(X,Y,N3,sigma, params)
    W3= np.average(part_layer3["pf_W3"], weights=like_3["new"], axis=0)    #dimention= [n_hidden*num_features,]
    W3 =W3.reshape((n_h2,1))
    params["W3"]=W3

    b3=np.average(part_layer3["pf_b3"], weights=like_3["new"], axis=0) 
    b3= b3.reshape((1,1))
    params["b3"]=b3


    cache = feed_forward(X, params)
    cost,error = square_loss(Y, cache["Y_pred"])
    #print("...computing backprop")
    grads = back_propagate(X, Y, params, cache)

    params["W1"] = params["W1"] - learning_rate * grads["dW1"]
    params["b1"] = params["b1"] - learning_rate *  grads["db1"]
    params["W2"] = params["W2"] - learning_rate * grads["dW2"]
    params["b2"] = params["b2"] - learning_rate * grads["db2"]
    params["W3"] = params["W3"] - learning_rate * grads["dW3"]
    params["b3"] = params["b3"] - learning_rate * grads["db3"]


    if i % 100 == 0:
        print("Epoch {}: training cost = {}".format(i+1 ,cost))

    loss_train_history.append(cost)
    RMSE_train_history.append(error)

    
#saving particles layer 1, layer2, importance weights
f1 = open('./{}/particles.pkl'.format(folder), 'wb')
pickle.dump([part_layer1, part_layer2,part_layer3, like_1, like_2, like_3], f1)
f1.close()

# Creating Plot for training
fig = plt.figure(figsize=(15,7))    
plt.plot(loss_train_history)
plt.title('model final cost = {}'.format(cost))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('./{}/cost_plot.png'.format(folder))
plt.close(fig)

fig = plt.figure(figsize=(15,7))    
plt.plot(loss_train_history)
plt.title('model final RMSE = {}'.format(error ))
plt.ylabel('error')
plt.xlabel('epoch')
plt.savefig('./{}/RMSE_plot.png'.format(folder))
plt.close(fig)
print('plot done!')
#####################################
#TESTING

print("test time")
def predictions_pf(X,part_layer1,part_layer2,part_layer3,N1,N2,N3):

    part_layer1["pf_W1"]=part_layer1["pf_W1"].reshape([N1,X.shape[1], n_h1])
    part_layer1["pf_b1"]=part_layer1["pf_b1"].reshape([N1,1, n_h1])
    part_layer2["pf_W2"]=part_layer2["pf_W2"].reshape([N2,n_h1,n_h2])
    part_layer2["pf_b2"]=part_layer2["pf_b2"].reshape([N2,1, n_h2])
    part_layer3["pf_W3"]=part_layer3["pf_W3"].reshape([N3,n_h2,1])
    part_layer3["pf_b3"]=part_layer3["pf_b3"].reshape([N3,1, 1])

    Z1 = np.matmul(X, part_layer1["pf_W1"]) + part_layer1["pf_b1"]
    A1 =relu(Z1)
    A1_avr=np.mean(A1,axis=0)
    
    Z2 = np.matmul( A1_avr,part_layer2["pf_W2"]) + part_layer2["pf_b2"]
    A2 = relu(Z2)
    A2_avr=np.mean(A2,axis=0)
    
    pred = np.matmul( A2_avr,part_layer3["pf_W3"]) + part_layer3["pf_b3"]  #shape=[N3,points,1]
    return pred



X_test = np.linspace(-1.5, 1.5, 10).reshape(-1, 1)
#predictions shape=[N3,#points,1]
predictions = predictions_pf(X_test,part_layer1,part_layer2,part_layer3,N1,N2,N3 )
#print(predictions)
y_mean = np.mean(predictions, axis=0)
y_mean =y_mean.flatten()
y_sigma = np.std(predictions, axis=0)
y_sigma =y_sigma.flatten()
y_true = f(X_test, sigma=0.0)

E=[]
for i in range(N3):
    cost_t,rmse_t = square_loss(y_true,predictions[i,:,:])
    E.append(rmse_t)
E =np.array(E)
RMSE_test = np.mean(E)
print('Test RMSE particles =',RMSE_test )

fig = plt.figure(figsize=(15,7))
plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
plt.scatter(X, Y, marker='+', label='Training data')
plt.scatter(X_test, y_true, marker='*', label='Test data')
plt.fill_between(X_test.ravel(), 
                 y_mean + 2 * y_sigma, 
                 y_mean - 2 * y_sigma, 
                 alpha=0.5, label='Uncertainty')
plt.title('Particles Prediction')
plt.legend()
plt.savefig('./{}/prediction & uncertainty.png'.format(folder))
plt.close(fig)
