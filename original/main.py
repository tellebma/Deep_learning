import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


def initialisation(X:np.ndarray)->(np.ndarray,int):
    """

    :param X: Matrix grand X.
    :return:tuple(np.ndarray, int) (W, b)
    """
    W = np.random.randn(X.shape[1],1)
    b = np.random.randn(1)
    return (W,b)

def model(X:np.ndarray,W:np.ndarray,b:int)->np.ndarray:
    """
    
    :param X:matrix X 
    :param W: 
    :param b: 
    :return: 
    """
    Z = X.dot(W) + b
    A = 1/(1+np.exp(-Z))


    return A

def logloss(A:np.ndarray, y:np.ndarray)->int:
    cost = 1/ len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
    return cost

def gradients(A:np.ndarray,X:np.ndarray,y:np.ndarray)->(np.ndarray,int):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW , db)

def update(dW:np.ndarray,db:int, W:np.ndarray, b:int, learning_rate:int)->None:
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W,b)

def artificialNeuron(X,y,learning_rate:int=0.1,n_iter:int=100):
    #initialisation des paramètres W, b
    W, b = initialisation(X)

    lossList = []
    for i in range(n_iter):
        A = model(X, W, b)
        lossList.append(logloss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X,W,b)
    print(accuracy_score(y,y_pred))

    plt.plot(lossList)
    plt.show()

    return (W, b)

def predict(X,W,b):
    """
    Return 1 si > à 0.5 sinon 0

    :param X:
    :param W:
    :param b:
    :return:
    """
    A = model(X, W, b)
    return A >= 0.5




if __name__ == '__main__':
    X,y = make_blobs(n_samples=100, n_features=2, centers=2,random_state=0)
    y = y.reshape((y.shape[0],1))
    print(f'dimensions de x: {X.shape}')
    print(f'dimensions de y: {y.shape}')
    plt.scatter(X[:,0],X[:,1],c=y,cmap='summer')
    plt.show(block=True)

    (W,b) = artificialNeuron(X,y)

    new_plant = np.array([2, 1])

    x0 = np.linspace(-1,4,100)
    x1 = (-W[0] * x0 - b)/W[1]


    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
    plt.scatter(new_plant[0],new_plant[1], c='r')
    plt.plot(x0,x1, c='orange',lw=3)#lw = épaisseur de la ligne.
    plt.show()
    print(predict(new_plant,W,b))




