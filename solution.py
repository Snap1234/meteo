import numpy as np
iris = np.genfromtxt('iris.txt')







######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        length = len(iris[:,0])
        µ = sum(iris)/length
        return µ[:-1]
        pass

    def covariance_matrix(self, iris):
        µ = self.feature_means(self,iris)
        length = len(iris[:,0])
        width = len(iris[0,:-1])
        cov = np.zeros((width,width))
        for i in range(width):
           for ii in range(i+1):
                co = sum((iris[:,i]-µ[i])*(iris[:,ii]-µ[ii])/length)
                cov[i][ii] = co
                cov[ii][i] = co
                
        return cov 
        pass

    def feature_means_class_1(self, iris):
        length = np.where(iris[:,4] == 2)[0][0] #la derniere ligne = 1
        µ = sum(iris[:length])/length
        return µ[:-1]
            
        pass

    def covariance_matrix_class_1(self, iris):
        µ = self.feature_means_class_1(self,iris)
        length = np.where(iris[:,4] == 2)[0][0]
        width = len(iris[0,:-1])
        cov = np.zeros((width,width))
        for i in range(width):
           for ii in range(i+1):
                co = sum((iris[:length,i]-µ[i])*(iris[:length,ii]-µ[ii])/length)
                cov[i][ii] = co
                cov[ii][i] = co
                
        return cov         
        
        pass


class HardParzen:
    def __init__(self, h):
        self.h = h
        


    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        length = len(train_labels)
        list_classes = np.zeros(length)
        nbr_classes = 0
        for i in train_labels:
            if (i not in list_classes):
                list_classes[nbr_classes] = i
                nbr_classes += 1
                
        one_hot = np.zeros((length,nbr_classes))
        
        for i in range(length):
            ii = np.where(list_classes == train_labels[i])
            one_hot[i,ii] = 1       # tout ca pour transformer Y en one hot
        

        self.one_hot = one_hot
        self.X = train_inputs
       
 
        pass

    def compute_predictions(self, test_data):
        
        def K(X,x,h = self.h):
            dist = np.sqrt(sum(((X-x).transpose())**2)) # transpose pour faire la sum sur un point
            return 1*(dist < h)
        
        def f(x , X = self.X , Y = self.one_hot):
            Y = Y[:len(X)]
            if (sum(K(X,x)) == 0 ): return draw_rand_label(x, self.label_list)
            else:return np.argmax((1/sum(K(X,x)))*np.dot(K(X,x),Y))+1
        
        length = len(test_data)
        
        pred = np.zeros(length)
        
        for i in range(length):
            
            x = test_data[i]
            pred[i]= f(x)
            
        return pred
        pass


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma
        

            

    def train(self, train_inputs, train_labels):
        
        self.X = train_inputs
        self.label_list = np.unique(train_labels)
        length = len(train_labels)
        self.D = len(self.label_list)
                
        one_hot = np.zeros((length,self.D))

        for i in range(length):
            ii = np.where(self.label_list == train_labels[i])
            one_hot[i,ii] = 1
        
        self.one_hot = one_hot
        pass

    def compute_predictions(self, test_data):
        
        def K(X,x):
            dist = np.sqrt(sum(((X-x).transpose())**2))
            k = (1/((self.sigma**self.D)*(2*np.pi)**(self.D/2)))*np.exp(-0.5*(dist/self.sigma)**2)
                
            return k
        
        def f(x , X = self.X , Y = self.one_hot):
            Y = Y[:len(X)]
            
            return np.argmax((1/sum(K(X,x)))*np.dot(K(X,x),Y))+1
        
        length = len(test_data)
        pred = np.zeros(length)
        
        for i in range(length):
            x = test_data[i]
            pred[i]= f(x)
            
        return pred
        
        pass


def split_dataset(iris):
    a,b,c = iris[::5],iris[1::5],iris[2::5]
    
    length = len(a) + len(b) + len(c)
    width = len(iris[0,:])
    
    train_set = np.zeros((length,width))
    train_set[::3] = a
    train_set[1::3] = b
    train_set[2::3] = c
    
    validation_set = iris[3::5]
    test_set = iris[4::5]
    
    return (train_set,validation_set,test_set)
    pass


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        
        f1 = HardParzen(h)
        f1.train(self.x_train,self.y_train)
        pred = f1.compute_predictions(self.x_val)
        error = sum(pred == self.y_val)/len(self.y_val) 
        return 1-error
        pass

    def soft_parzen(self, sigma):
        
        f2 = SoftRBFParzen(sigma)
        f2.train(self.x_train,self.y_train)
        pred = f2.compute_predictions(self.x_val)
        error = sum(pred == self.y_val)/len(self.y_val) 
        return 1-error
        pass


def get_test_errors(iris):
    h = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0])
    sigma = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0])
    
    split = split_dataset(iris)

    test_error = ErrorRate(split[0][:,:-1],split[0][:,-1],split[1][:,:-1],split[1][:,-1])
    

    errorHard = np.zeros(len(h))

    for i in range(len(h)):
        errorHard[i] = test_error.hard_parzen(h[i])
        
  
    #max_h = h[np.argmax(errorHard)]
    
    errorSoft = np.zeros(len(sigma))

    for i in range(len(sigma)):
        errorSoft[i] = test_error.soft_parzen(h[i])
        
   # max_sigma = sigma[np.argmax(errorSoft)]
    
    return [min(errorHard),min(errorSoft)]
        
    pass



def random_projections(X, A):
    pass



h = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0])
sigma = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0])
    
split = split_dataset(iris)

test_error = ErrorRate(split[0][:,:-1],split[0][:,-1],split[1][:,:-1],split[1][:,-1])

errorHard = np.zeros(len(h))

for i in range(len(h)):
    errorHard[i] = test_error.hard_parzen(h[i])


import matplotlib.pyplot as plt
plt.figure(1)
plt.title("HARDparzen")
plt.xlabel("h")
plt.ylabel("erreur de test")
plt.xscale("log")
plt.plot(h,errorHard, label = "HARDparzen")


errorSoft = np.zeros(len(sigma))

for i in range(len(sigma)):
    errorSoft[i] = test_error.soft_parzen(h[i])


plt.title("SOFT et HARD parzen")
plt.ylabel("erreur de test")
plt.xlabel("sigma ou h")
plt.xscale("log")
plt.plot(h,errorSoft, label = "SOFTparzen")
plt.legend()
