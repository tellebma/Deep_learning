import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from utilities import *
from tqdm import tqdm

def initialisation(X: np.ndarray) -> (np.ndarray, int):
    """

    :param X: Matrix grand X.
    :return:tuple(np.ndarray, int) (W, b)
    """
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def model(X: np.ndarray, W: np.ndarray, b: int) -> np.ndarray:
    """

    :param X:matrix X
    :param W:
    :param b:
    :return:
    """
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))

    return A


def logloss(A: np.ndarray, y: np.ndarray) -> int:
    """
    log loss : A = 0 ou inf
    Mais log(0) erreur domain donc on rajoute un epsilon eps qui est = a 0.00000000000001 comme ca log(0) = log(eps) = possible.
    """
    eps = 1e-15
    cost = 1 / len(y) * np.sum(-y * np.log(A + eps) - (1 - y) * np.log(1 - A + eps))
    return cost


def gradients(A: np.ndarray, X: np.ndarray, y: np.ndarray) -> (np.ndarray, int):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


def update(dW: np.ndarray, db: int, W: np.ndarray, b: int, learning_rate: int) -> None:
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)


def artificialNeuron(X_train, y_train,X_test,y_test, learning_rate: int = 0.1, n_iter: int = 1000):
    # initialisation
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    W, b = initialisation(X_train)



    for i in tqdm(range(n_iter)):


        A = model(X_train, W, b)
        if i % 10 == 0:
            #~~~~train
            # calcul du coût
            train_loss.append(logloss(A, y_train))
            # calcul de l'accuracy
            train_acc.append(accuracy_score(y_train, predict(X_train, W, b)))

            # ~~~~test
            A_test = model(X_test,W,b)#test du modèle avec le test.
            # calcul du coût
            test_loss.append(logloss(A_test, y_test))
            # calcul de l'accuracy
            test_acc.append(accuracy_score(y_test, predict(X_test, W, b)))

        # update
        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss,label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_acc,label='train acc')
    plt.plot(test_acc, label='test acc')
    plt.show()

    return (W, b)


def predict(X, W, b):
    """
    Return 1 si > à 0.5 sinon 0

    :param X:
    :param W:
    :param b:
    :return:
    """
    A = model(X, W, b)
    return A >= 0.5


def normalisationMinMax(X):
    """
          X - Xmin
    X =  ----------
         Xmax - Xmin

    """

    return (X - X.min()) / (X.max() - X.min())


if __name__ == '__main__':
    # X,y = make_blobs(n_samples=100, n_features=2, centers=2,random_state=0)
    # y = y.reshape((y.shape[0],1))

    # https://youtu.be/5TpBe7KTAHE?t=2107
    # moi rien capiche:

    X_train, y_train, X_test, y_test = load_data()

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    """
    #plt.figure(figsize=(32, 8))
    for i in range(1, 16):
        plt.subplot(4, 5, i)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(y_train[i])
        plt.tight_layout()
    plt.show()
    """

    # print(f'dimensions de x: {X.shape}')
    # print(f'dimensions de y: {y.shape}')
    # plt.scatter(X[:,0],X[:,1],c=y,cmap='summer')
    # plt.show(block=True)

    # redimensionnement de train_set#
    X_train_reshape = X_train.reshape(X_train.shape[0],
                                      -1)  # reshape <=> 1er element ne change pas, 2ième = -1 = tout ce qu'il reste, possible autre valeur : X_train.shape[1]*X_train.shape[2]
    X_train_reshape_normalized = normalisationMinMax(X_train_reshape)

    # X_train_reshape.shape()
    # (1000, 4096)

    # par ailleurs on modifie aussi test set (qui servira plustard)
    X_test_reshape = X_test.reshape(X_test.shape[0], -1)
    X_test_reshape_normalized = normalisationMinMax(X_test_reshape)
    # X_test_reshape.shape()
    # (200, 4096)

    (W, b) = artificialNeuron(X_train_reshape_normalized, y_train,X_test_reshape_normalized, y_test,0.01,10000)

    x0 = np.linspace(-1, 4, 100)
    x1 = (-W[0] * x0 - b) / W[1]

    predict(X_test_reshape_normalized, W, b)

