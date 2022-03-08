lim = 10
h = 100
W1 = np.linspace(-lim,lim,h)
W2 = np.linspace(-lim,lim,h)

W11,W22 = np.meshgrid(W1,W2)

b= 0

#ravel = applatir les tableau
#c_ = concatenate.
#.T = transposer

W_final = np.c_[W11.ravel(),W22.ravel()].T

Z = X.dot(W) + b