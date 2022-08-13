import os
import pandas
import numpy as np
import matplotlib.pyplot as plt



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
X, Y, X_test = train_data[:,3:-2],train_data[:,-1], test_data[:,1:-1]
time = train_data[:,-2]

########## NETTOYER PRECT ##############
zeros0 = np.where((X[:,9] == 0) & (Y == 0))
zeros1 = np.where((X[:,9] == 0) & (Y == 1))
zeros2 = np.where((X[:,9] == 0) & (Y == 2))

not_zeros0 = np.where((X[:,9] != 0) & (Y == 0)) # on redistribue les donnees au lieu des 0
not_zeros1 = np.where((X[:,9] != 0) & (Y == 1))
not_zeros2 = np.where((X[:,9] != 0) & (Y == 2))

ind_redistribution0 = np.random.randint(len(not_zeros0[0]), size = len(zeros0[0]))
ind_redistribution1 = np.random.randint(len(not_zeros1[0]), size = len(zeros1[0]))
ind_redistribution2 = np.random.randint(len(not_zeros2[0]), size = len(zeros2[0]))

X[zeros0,9] = X[ind_redistribution0[0],9]
X[zeros1,9] = X[ind_redistribution1,9]
X[zeros2,9] = X[ind_redistribution2[0],9]

X[:,9] = np.log(np.abs(X[:,9]))
########################################


normaliser = sum(np.abs(X))/len(X) # normaliser
mean = sum(X)/len(X) #centrer
X=((X-mean)/normaliser).T # normaliser et centrer X

normaliser = sum(np.abs(X_test))/len(X_test)
mean = sum(X_test)/len(X_test) 
X_test=((X_test-mean)/normaliser).T 


length = int(len(X.T)/9)
X_train = X[:,length:] # separation en training et validation
X_val = X[:,:length]

# X -> v1  v2  v3 v_n
#   x1|#   #   #   # |
#   x2|#   #   #   # |
#   x3|#   #   #   # |
#  x_d|#   #   #   # |


def one_hot(labels):

    label_list = np.unique(labels)
    length = len(labels)
    D = len(label_list)
                    
    new_one_hot = np.zeros((length,D))
    
    for i in range(length):
        ii = np.where(label_list == labels[i])
        new_one_hot[i,ii] = 1

    return new_one_hot.transpose()

Y_one_hot = one_hot(Y)

Y_train_one_hot = Y_one_hot[:,length:]
Y_val_one_hot = Y_one_hot[:,:length]


#  Y_one_hot -> v1  v2  v3 v_n
#            y1|1   0   0   # |
#            y2|0   1   0   # |
#           y_m|0   0   1   # |

###########################################################################


##################### FONCTIONS ########################################


# def sigmoid(x): return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

 

def g(X,W,b): 
    
    wx_b = np.dot(W.transpose(),X) + np.expand_dims(b,axis=1) # Wx+b
    Z = sum(np.exp(wx_b))
    softMax = np.exp(wx_b)/Z
    return softMax              # fonction de probabilite

# x ->  v1 v2 v3 x_n    W ->  1  2  m    W_t ->  w1 w2 w3 w_d   b_t -> |b1      g ->    pred1 pred2 pred3 pred_n
#   x1 |#  #  #  # |       w1|#  #  # |        1|#  #  #  # |           b2        prob1 | #     #      #     #  |
#   x2 |#  #  #  # |       w2|#  #  # |        2|#  #  #  # |           b_m|      prob2 | #     #      #     #  |
#   x3 |#  #  #  # |       w3|#  #  # |        m|#  #  #  # |                    prob_m | #     #      #     #  |
#  v_d |#  #  #  # |      w_d|#  #  # |                                       
 

def f(G): return np.argmax(G,axis=0)

def accuracy(Y,G,n): return sum(np.argmax(G,axis=0) == np.argmax(Y,axis=0))/n

# v pour vecteur


def Lost(Y_train_one_hot,G): return -np.einsum("ij,ji->i",Y_train_one_hot.transpose(),np.log(G))

#  Y_one_hot.transpose ->  y1  y2  y_m     g  ->  pred1 pred2 pred3 pred_n    Lost -> | L1  L2  L3  L_n |
#                      v1 |1   0   0 |      prob1 | #     #     #      #  |
#                      v2 |0   1   0 |      prob2 | #     #     #      #  |
#                      v3 |0   0   1 |     prob_m | #     #     #      #  |
#                     v_n |#   #   # |   



def Risk(Y_train_one_hot,G): return (1/len(Y_train_one_hot[0]))*sum(Lost(Y_train_one_hot,G))  #lenY = n

# risk -> |R|


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
            "PSL: sea level pressure [Pa]",
            "T200: temperature at 200 mbar pressure surface [K]",
            "T500: temperature at 500 mbar pressure surface [K]",
            "PRECT: total (convective and large-scale) LOG precipitation rate (liq + ice) [m/s]",
            "TS: surface temperature (radiative) [K]",
            "TREFHT: reference height temperature [K]",
            "Z1000: geopotential Z at 1000 mbar pressure surface [m]",
            "Z200: geopotential Z at 200 mbar pressure surface [m]",
            "ZBOT: lowest modal level height [m]"]
    for ii in range(len(X[0])):
        a,b,c = Y_one_hot*X[:,ii].T
        a,b,c = a[a!=0], b[b!=0], c[c!=0]
        #if(ii==10): a,b,c = np.log(a),np.log(b),np.log(c)
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
        #if(ii==10): plt.xscale("log")
        plt.ylabel("normalised density")
        plt.xlabel("x"+str(ii))
        plt.plot(Ax,A,linewidth=0.3,label = "0") 
        plt.plot(Bx,B,linewidth=0.3,label = "1")
        plt.plot(Cx,C,linewidth=0.3,label = "2")
        plt.legend()        
        
#########################################################################





############################### ENTRAINEMENT ############################

d = len(X_train)
n_t = len(X_train[0,:])
n_v = len(X_val[0,:])
m = len(np.unique(Y))


b_init = 0.1*(np.random.rand(m)-0.5)
W_init = (np.random.rand(d,m)-0.5)




wx_b = np.dot(W_init.transpose(),X_train) + np.expand_dims(b_init,axis=1)


G=g(X_train,W_init,b_init) #test devrait donner des probs environ [0.33,0.33,0.33]

L = Lost(Y_train_one_hot,G)
print(L,"\n")

R = Risk(Y_train_one_hot,G)
print(R)


W1 = W_init     #dw
b1 = b_init


Y = Y_train_one_hot
Y_val = Y_val_one_hot

lost_skewer = np.sqrt((len(Y.T)/len(Y))/sum(Y.T)) # pour donner plus de Lost aux donnees moins frequentes
Y = (Y.T*lost_skewer).T
Y = (Y.T*sum(sum(Y.T))/n_t).T  # pour que sum(Y) = n j'imagine que ca peux aider
                             # au moins ca permet de comparer les valeurs de Risk

ii=1
h=0.1


# methode 1

for i in range(25000): # descente de gradient
    
    dW_,db_= (np.random.rand(d,m)-0.5),(np.random.rand(m)-0.5)
    
    dW,db =  h * dW_, h * db_
    
    W2, b2 = W1 + dW, b1 + db

    G1,G2 = g(X_train,W1,b1),g(X_train,W2,b2)
    
    dLost = (Lost(Y,G2)-Lost(Y,G1))/h
    dR = sum(dLost)/n_t
    
    if((i+1)%500==0):
        ii=2000/(i+1500)
        G_val = g(X_val,W1,b1)
        val_acc = accuracy(Y_val,G_val,n_v)
        train_acc= accuracy(Y,G1,n_t)
        print("R=%0.4f"%Risk(Y,G1 )," dR=%0.2e"%(dR/n_t)," i=",i,"\ntraining_acc=%0.2f"%train_acc," validation_acc=%0.2f"%val_acc,"\n")
        
        
    W1,b1 = W1-ii*dW_* dR, b1-ii*db_* dR




# methode 2
"""
for i in range(15000):
    
    dW_,db_= (np.random.rand(d,m)-0.5),(np.random.rand(m)-0.5)
    
    dW,db =  h * dW_, h * db_
    
    W2, b2 = W1 + dW, b1 + db

    G1,G2 = g(X_train,W1,b1),g(X_train,W2,b2)
    
    dLost = (Lost(Y,G2)-Lost(Y,G1))/h
    dR = sum(dLost)/n_t
    
    if(dR<0): W1,b1 = W2,b2
    
    if((i+1)%500==0): 
        h = h*(2500/(2500+i))
        G_val = g(X_val,W1,b1)
        val_acc = accuracy(Y_val,G_val,n_v)
        train_acc= accuracy(Y,G1,n_t)
        print("R=%0.2f"%Risk(Y,G1 )," dR=%0.2e"%(dR/n_t),"\ntraining_acc=%0.2f"%train_acc," validation_acc=%0.2f"%val_acc,"\n")
        


"""



L1 = Lost(Y,G1)
L2 = Lost(Y,G2)

###########################################################################