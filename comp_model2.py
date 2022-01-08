import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


def resampling(X):
    Y = one_hot(X[:,-1])
    n = len(Y)
    n1 =  len(Y[0][Y[0]!=0])
    n2 =  len(Y[1][Y[1]!=0])
    n3 =  len(Y[2][Y[2]!=0])
    
    l1 = int(n1/((n1/n2)**(1/3)))
    l2 = int(n2*((n1/n2)**(1/3)))
    l3 = int(n1/((n1/n2)**(1/3))*(n3/n1)**(1/3))
    
    new_X = np.zeros(l1+l2+l3)
    
    new_X1 = np.zeros((n,l1))
    new_X2 = np.zeros((n,l2))
    new_X3 = np.zeros((n,l3))
    
    new_X1 = X[Y[0,:]==1][:l1]   # downsampling
    
    rep = round(l2/n2)
    new_X2 = np.repeat(X[Y[1,:]==1],rep,axis = 0)
    
    new_X3 =  X[Y[2,:]==1]
    
    new_X = np.concatenate((new_X1,new_X2,new_X3))
    np.random.shuffle(new_X)
    
    return new_X

def one_hot(labels):

    label_list = np.unique(labels)
    length = len(labels)
    D = len(label_list)
                    
    new_one_hot = np.zeros((length,D))
    
    for i in range(length):
        ii = np.where(label_list == labels[i])
        new_one_hot[i,ii] = 1

    return new_one_hot.transpose()


###################   LECTURE DES FICHIERS .CSV  ##########################
data = [0,0,0]
for _, _, files in os.walk(os.getcwd()): # assurez-vous que le .py et les .csv soient dans le meme fichier 
       i=0
       for filename in files:
           if((filename[-4:] ==".csv") & (filename[:4] != "pred")):
               if(filename[:4] == "samp") : data[i] = pandas.read_csv(filename)
               else: data[i] = pandas.read_csv(filename).to_numpy()
               i+=1
               
sample_submission, test_data, train_data = data  
###########################################################################


#################### PREPARATION DES DONNEES ##############################

test_data = np.delete(test_data,9,axis=1) # donnees inutiles
train_data = np.delete(train_data,9,axis=1)

# separation des donnes en matrice de variable X et liste de labels
X,Y,X_test = train_data[:,1:],train_data[:,-1], test_data[:,1:-1]

########## NETTOYER PRECT ##############
zeros0 = np.where((X[:,9] == 0) & (X[:,-1] == 0))
zeros1 = np.where((X[:,9] == 0) & (X[:,-1] == 1))
zeros2 = np.where((X[:,9] == 0) & (X[:,-1] == 2))

not_zeros0 = np.where((X[:,9] != 0) & (X[:,-1] == 0)) # on redistribue les donnees au lieu des 0
not_zeros1 = np.where((X[:,9] != 0) & (X[:,-1] == 1))
not_zeros2 = np.where((X[:,9] != 0) & (X[:,-1] == 2))

ind_redistribution0 = np.random.randint(len(not_zeros0[0]), size = len(zeros0[0]))
ind_redistribution1 = np.random.randint(len(not_zeros1[0]), size = len(zeros1[0]))
ind_redistribution2 = np.random.randint(len(not_zeros2[0]), size = len(zeros2[0]))

#X[zeros0,9] = X[ind_redistribution0[0],9]
#X[zeros1,9] = X[ind_redistribution1,9]
#X[zeros2,9] = X[ind_redistribution2[0],9]

#X[:,9] = np.log(np.abs(X[:,9]))
########################################

X = np.delete(X,11,axis=1)
X_test = np.delete(X_test,11,axis=1)


length = int(len(X)/8)

X_train = X[length:,:]
X_train = X[length:,:]
X_val = X[:length,:]
Y_val = one_hot(X_val[:,-1])
X_val = X_val[:,:-2]
X_train = resampling(X_train)
Y_train = X_train[:,-1]
Y_train = one_hot(Y_train)
X_train = X_train[:,:-2]  ### resampling et enlever les colonnes de temps et labels


normaliser = sum(np.abs(X_train))/len(X_train) # normaliser
mean = sum(X_train)/len(X_train) #centrer


X_train=((X_train-mean)/normaliser).T # normaliser et centrer X_train
X_val=((X_val-mean)/normaliser).T 
X_test=((X_test-mean)/normaliser).T 

Y_one_hot = one_hot(Y)

###########################################################################

##################### FONCTIONS ########################################
  


########################## FUNCTIONS D'ENTRAÎNEMENT ###############################
def f(G): return np.argmax(G,axis=0)

def accuracy(Y,G,n): return sum(np.argmax(G,axis=0) == np.argmax(Y,axis=0))/n

def Lost(Y_train_one_hot,G): return -np.einsum("ij,ji->i",Y_train_one_hot.transpose(),np.log(G))

def Risk(Y_train_one_hot,G): return (1/len(Y_train_one_hot[0]))*sum(Lost(Y_train_one_hot,G))  #lenY = n

def pred(α,αKb):

    Z = sum(np.exp(αKb).T)
    softMax = (np.exp(αKb).T)/Z
    return softMax  

def Kernel(sig,X1=X_train,X2=X_train):
    
    n = len(X2.T)
    m = len(X1.T)
    K = np.zeros((n,m+1),dtype = "float16")
    for i in range(n):
        
        XX = X1.T-X2.T[i]
        K[i,:-1] =np.exp(-sig*np.einsum("ij,ji->i",XX,XX.T)+10)
        if((i+1)%5000 == 0):  print("k",K[i,:4])

    K[:,-1]=1 #beta
    return np.exp(-10)*csr_matrix(K)

def Grad(X,Y,α,lam,P,K):
    
    n = len(Y.T)
    var = -Y.T + P.T + lam*α[:-1]
    var_beta = -Y.T + P.T
     
    grad_alpha = (40/n)*K[:,:-1].dot(var)
    grad_beta = [0.1*(40/n)*sum(var_beta)]
    grad = np.concatenate((grad_alpha,grad_beta),axis=0)
    print("grad",grad[0])
    return grad
 

def train(α,Y,Y_val,lam,K,K_val,sig,it,lr=0.03):
    
    n_t = len(α)
    acc_list = np.zeros(it) 
    
    for i in range(it): # descente de gradient
        
        print(i)
        
        ak = K.dot(α)
        ak_val =  K_val.dot(α)
        P = pred(α,ak)
        P_val=pred(α,ak_val)
        acc = accuracy(Y,P,n_t)
        acc_val = accuracy(Y_val,P_val,n_v)
        acc_list[i]=acc_val
        print("\ntrain accuracy=", acc)
        print("validation accuracy=", acc_val,"\n")
                
        grad = Grad(X_train,Y,α,lam,P,K)
        α = α - lr*grad      

    return acc_list,α   

#############################################################################

######################### FONCTION POUR PRODUIRE LE FICHIER CVS #############

def produce_pred_file(pred):
    
    pred = np.argmax(pred,axis = 0)
    
    data = np.zeros((len(pred),2))
    data[:,0] = np.arange(len(pred))
    data[:,1] = pred
    
    data_frame = pandas.DataFrame({"S.No":data[:,0],"LABELS":data[:,1]}).astype(int)
    
    i = 1
    while(i<20):
        try: data = data_frame.to_csv(os.getcwd()+"\\pred"+str(i)+".csv",index = False)
        except: True
        else : break
        finally: 
            i+=1
            if(i==20): print("trop de fichiers prediction") 

##############################################################################



#################### FONCTIONS DE VISUALISATIONS #############################
    
def graph_data(data = train_data):
    X,Y = data[:,1:-2],data[:,-1]  
    for i in range(len(X[0])):
        plt.figure(i)
        plt.plot(X[:,i],Y,",")
        
def graph_data_density(data = train_data,Y=Y_one_hot):
    X = data[:,3:-2]
    name = ["TMQ:total (vertically integrated) precipitable water [kg/m^2]",
            "U850: zonal wind at 850 mbar pressure surface [m/s]",
            "V850: meridional wind at 850 mbar pressure surface [m/s]",
            "UBOT: lowest level zonal wind [m/s]",
            "VBOT: lowest model level meridional wind [m/s]",
            "QREFHT: reference height humidity [kg/kg]",
            "PS",
            "PSL: sea level pressure [Pa]",
            "T200: temperature at 200 mbar pressure surface [K]",
            "T500: temperature at 500 mbar pressure surface [K]",
            "PRECT: total (convective and large-scale) precipitation rate (liq + ice) [m/s]",
            "TS: surface temperature (radiative) [K]",
            "TREFHT: reference height temperature [K]",
            "Z1000: geopotential Z at 1000 mbar pressure surface [m]",
            "Z200: geopotential Z at 200 mbar pressure surface [m]",
            "ZBOT: lowest modal level height [m]"]
    for ii in range(len(X[0])):
        a,b,c = Y_one_hot*X[:,ii].T
        a,b,c = a[a!=0], b[b!=0], c[c!=0]
        if(ii==10): a,b,c = np.log(a),np.log(b),np.log(c)
        ra,rb,rc = max(a)-min(a),max(b)-min(b),max(c)-min(c)
        inc = 100
        
        
        
        A,B,C = np.arange(inc),np.arange(inc),np.arange(inc)
        ra,rb,rc = ra/inc,rb/inc,rc/inc
        for i in range(inc):
            A[i] = len(a[(a<(min(a)+(i+1)*ra)) & (a>=(min(a)+i*ra))])
            B[i] = len(b[(b<(min(b)+(i+1)*rb)) & (b>=(min(b)+i*rb))])
            C[i] = len(c[(c<(min(c)+(i+1)*rc)) & (c>=(min(c)+i*rc))])
            
        A,B,C = A/sum(A),B/sum(B),C/sum(C)
        A[A==0],B[B==0],C[C==0] = np.nan,np.nan,np.nan
        Ax,Bx,Cx = np.linspace(min(a),max(a),inc),np.linspace(min(b),max(b),inc),np.linspace(min(c),max(c),inc)
        plt.figure(ii,dpi=400)
        plt.title(name[ii])
        plt.ylabel("normalised density")
        plt.xlabel("x"+str(ii))
        plt.plot(Ax,A,linewidth=0.3,label = "0") 
        plt.plot(Bx,B,linewidth=0.3,label = "1")
        plt.plot(Cx,C,linewidth=0.3,label = "2")
        plt.legend()        
####################################################################################        
        

########################### fonctions test heuristiques ################################
def test_sig(lam):
    sig_list = np.array([2,4,8,15,25,40])
    m = len(sig_list)
    train_result = np.zeros((m,50))
    for i in range(m):
        α = np.random.rand(n_t+1,3)-0.5
        α[-1] = 0
        K = Kernel(sig_list[i])
        K_val = Kernel(sig_list[i],X_train,X_val)
        train_result[i],α=train(α,Y_train,Y_val,lam,K,K_val,sig_list[i],it=50,lr=0.03*sig_list[i])
        del K
        
    plt.figure(dpi=400)
    plt.title("test sur les sigmas")
    plt.xlabel("iteration")
    plt.ylabel("taux de prediction")
    plot = plt.plot(train_result.T,linewidth=0.5)
    plt.legend(iter(plot),sig_list)
    return train_result

def test_div(α,Y,lam,sig):
    div_list = np.array([0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000,3000])
    K = Kernel(sig)
    m = len(div_list)
    train_result = np.zeros((m,10))
    for i in range(m):
        α = np.random.rand(n_t,3)-0.5
        train_result[i]=train(α,Y,lam,K,sig,10,div_list[i])
    plt.figure(dpi=400)    
    plot = plt.plot(train_result.T,linewidth=0.5)
    plt.legend(iter(plot),div_list)
    return train_result
 
def test_lr(lam,sig):
    lr_list = np.array([0.08,0.3,0.8,3])
    K = Kernel(sig)
    K_val = Kernel(sig,X2=X_val)
    m = len(lr_list)
    train_result = np.zeros((m,100))
    for i in range(m):
        α = np.random.rand(n_t+1,3)-0.5
        train_result[i],α=train(α,Y_train,Y_val,lam,K,K_val,sig=8,it=150,lr=lr_list[i])
    plt.figure(dpi=400)  
    plt.title("test sur les learning rate")
    plt.ylabel("taux de prediction (validation)")
    plt.xscale("itérations")
    plot = plt.plot(train_result.T,linewidth=0.5)
    plt.legend(iter(plot),lr_list)
    return train_result

#######################################################################################
#########################################################################



############################### ENTRAINEMENT ############################

del train_data
del test_data
del data
del files
del sample_submission   # nettoyer un peu de RAM

n_t = len(X_train[0,:])
n_v = len(X_val[0,:])
m = len(Y_one_hot)

α = (np.random.rand(n_t+1,m)-0.5)/30
α[-1]=0 # initialise beta à 0

sig = 8
lam = 0.5

K, K_val, K_test = Kernel(5),Kernel(5,X2=X_val), Kernel(5,X2 = X_test) # calcul des noyaux

res,α = train(α,Y_train,Y_val,lam,K,K_val,sig,it=70,lr=0.2)

ak_test = K_test.dot(α)
P_test = pred(α,ak_test)
produce_pred_file(P_test)


###########################################################################