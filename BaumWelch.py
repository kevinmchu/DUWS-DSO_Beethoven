#Function using the log-sum-exp trick#
def logSumExp(a):
    if np.all(np.isinf(a)):
        return np.log(0)
    else:
        b = np.max(a)
        return(b + np.log(np.sum(np.exp(a-b))))


def pForward(g, x):
    pXf = logSumExp(g[len(x)-1,:])
    return(pXf)
def indicator(a, b):
    if a == b:
        return 1
    else:
        return 0
def pForwardARHMM(g, x):
    pXf = logSumExp(g[len(x)-1,:, :])
    return(pXf)

    
import numpy as np

def forwardAlg(n, m, k, pi, Tmat, phi, x):
    g = np.zeros((n,m))
    for i in range(0,m):
        g[0,i] = (pi[i]) + (phi[i, x[0]])
    
    for j in range(1, n):
        for l in range(0, m):
            g[j,l] = logSumExp(np.asarray(g[j-1, :])+np.asarray(Tmat[:,l])+(phi[l,x[j]]))
    return(g)

def backwardAlg(n, m, k, pi, Tmat, phi, x):
    r = np.zeros((n,m))
    for j in range(n-2, -1, -1):
        for l in range(0, m):
            r[j, l] = logSumExp(np.asarray(r[j+1,: ]) + np.asarray(Tmat[l,:]) + phi[:, x[j+1]])
    
    return(r)

def Viterbi(n, m, k, pi, Tmat, phi, x):
    f = np.zeros(shape = (n,m))
    alpha = np.zeros(shape = (n,m))
    zStar = np.zeros(n)
    
    for t in range(0, n):
        for i in range(0,m):
            if t == 0:
                f[0, i] = pi[i] + phi[i, x[0]]
            else:
                u = np.asarray(f[t-1, :]) + np.asarray(Tmat[:, i]) + phi[i, x[t]]
                f[t,i] = np.max(u)
                alpha[t,i] = np.argmax(u)
    zStar[n-1] = np.argmax(np.asarray(f[n-1, :]))
    for i in range(n-2, -1, -1):
        zStar[i] = alpha[i+1, int(zStar[i+1])]
    return zStar

def first_order(n, m, k, x, tol):
    #randomly initialize pi, phi and T#
    vals = np.random.rand(m)
    pi = np.log(vals/np.sum(vals))
    Tmat = np.zeros(shape = (m, m))
    phi = np.zeros(shape = (m, k))
    gamma = np.zeros(shape = (n, m))
    beta = np.zeros(shape = (n,m,m))
    iterations = 0
    convergence = 0
    count = 0
    pOld = 1E10
    pNew = 0
    criteria = 0
    #cdef double[:,:] p = np.zeros(shape = (n,m))
    
    vals1 = np.random.rand(m,m)
    vals2 = np.random.rand(m,k)
    Tmat = np.log(vals1/np.sum(vals1, axis=1)[:,None])
    phi = np.log(vals2/np.sum(vals2, axis = 1)[:,None])
    
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        #Perform forward and backward algorithms# 
        g = forwardAlg(n, m, k, pi, Tmat, phi, x)
        h = backwardAlg(n, m, k, pi, Tmat, phi, x)
        pNew = pForward(g, x)
        
        ##E-Step##
    
        #Calculate gamma and beta#
        for t in range(0, n):
            for i in range(0,m):
                gamma[t,i] = g[t,i] + h[t,i] - pNew
        #p = np.full((n,m), pNew)
        #gamma = g+h-p
        for t in range(1, n):
            for i in range(0, m):
                for j in range(0, m):
                    beta[t,i,j] = Tmat[i,j] + phi[j, x[t]] + g[t-1, i] + h[t, j] - pNew
        ##M-Step##
    
        #Update pi, phi and Tmat#
        pi = gamma[0,:] - logSumExp(gamma[0,:])
        for i in range(0, m):
            for j in range(0, m):
                Tmat[i,j] = logSumExp(beta[1::, i, j]) - logSumExp(beta[1::, i,:])
        for i in range(0,m):
            for w in range(0, k):
                j = 0
                count = 0
                for t in range(0,n):
                    if x[t] == w:
                        count = count+1
                indicies = np.zeros(count)
                for t in range(0,n):
                    if x[t] == w:
                        indicies[j] = gamma[t,i]
                        j = j+1
                    
                phi[i,w] = logSumExp(indicies) - logSumExp(gamma[:,i])
        
        criteria = abs(pOld - pNew)
        if criteria < tol:
            convergence = 1
        
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
            print(iterations)
    return (iterations, pNew, np.exp(pi), np.exp(phi), np.exp(Tmat))
    
def forwardAlg2(n, m, k, pi, Tmat, T2mat, phi, x):
    g = np.zeros(m)
    alpha = np.zeros((n, m, m))
    g = pi + np.asarray(phi[:, x[0]])
    
    for t in range(1,n):
        for j in range(0,m):
            for l in range(0,m):
                if t ==1:
                    alpha[1,j,l] = g[j] + Tmat[j,l] + phi[l, x[1]]
                else:
                    alpha[t,j,l] = logSumExp(np.asarray(alpha[t-1,:,j]) + np.asarray(T2mat[:,j,l]) + phi[l, x[t]])
    return(alpha)

def pForward2(m,g, x):
    pXf = logSumExp(g[len(x)-1,:,m-1])
    return(pXf)

def backwardAlg2(n, m, k, pi, Tmat, T2mat, phi, x):
    beta = np.zeros((n,m,m))
    for t in range(n-2, -1, -1):
        for j in range(0, m):
            for l in range(0,m):
                beta[t,j, l] = logSumExp(np.asarray(beta[t+1,j,: ]) + np.asarray(T2mat[j,l,:]) + np.asarray(phi[:, x[t+1]]))
    
    return(beta)

#Function to return p(x_1:n) from matrix from backward algorithm
def pBackward2(m,r, pi, phi, x):
    pXb = logSumExp(r[0,:,m-1]+ pi +phi[:,x[0]])
    return(pXb)



def second_order(n, m, k, x, pi, Tmat, phi, tol):
    #randomly initialize T2mat#
    T2mat = np.zeros(shape = (m,m,m))
    iterations = 0
    convergence = 0
    pOld = 1E10
    pNew = 0
    
    vals = np.random.rand(m,m,m)
    T2mat = np.log(vals/np.sum(vals, axis=2)[:,:,None])
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        
        #Perform forward and backward algorithms# 
        alpha = forwardAlg2(n, m, k, pi, Tmat, T2mat, phi, x)
        beta = backwardAlg2(n, m, k, pi, Tmat, T2mat, phi, x)
        pNew = pForward2(m,alpha, x)
        ##M-Step##
        eta = np.zeros((n,m,m,m))
        #Update pi, phi and Tmat#

        for t in range(1, n-1):
            for i in range(0, m):
                for j in range(0, m):
                    for l in range(0,m):
                        eta[t,i,j,l] = alpha[t,i,j] + T2mat[i,j,l] + phi[l, x[t+1]] + beta[t+1, j, l] - pNew
        
        for i in range(0, m):
            for j in range(0, m):
                for l in range(0,m):
                        T2mat[i,j,l] = logSumExp(eta[1::,i,j,l]) - logSumExp(eta[1::,i,j,:])
        print(iterations)
        criteria = abs(pOld - pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (iterations, np.exp(T2mat))
    
def forwardAlg3(n, m, k, pi, Tmat, T2mat, T3mat, phi,  x):
    g = np.zeros(m)
    g2 = np.zeros((m,m))
    alpha = np.zeros((n, m, m, m))
    
    g = pi + np.asarray(phi[:, x[0]])
    
    for t in range(1,n):
        for j in range(0,m):
            for q in range(0,m):
                for l in range(0,m):
                    if t ==1:
                        g2[j,q] = g[j] + Tmat[j,q] + phi[q, x[1]]
                        alpha[1,j,q,l] = g2[j, q] + T2mat[j,q,l] + phi[l, x[2]]
                    else:
                        alpha[t,j,q, l] = logSumExp(np.asarray(alpha[t-1,:,j, q]) + np.asarray(T3mat[:,j,q, l]) + phi[l, x[t]])
    return(alpha)

def pForward3(m,g, x):
    pXf = logSumExp(g[len(x)-1,:,m-1, m-1])
    return(pXf)

def backwardAlg3(n, m, k, pi, Tmat, T3mat, phi, x):
    beta = np.zeros((n,m,m,m))
    for t in range(n-2, -1, -1):
        for j in range(0, m):
            for q in range(0,m):
                for l in range(0,m):
                    beta[t,j, k, l] = logSumExp(np.asarray(beta[t+1,j,q,: ]) + np.asarray(T3mat[j,q,l,:]) + np.asarray(phi[:, x[t+1]]))
    
    return(beta)

def third_order(n, m, k, x, pi, Tmat, T2mat, phi, tol):
    #randomly initialize T3mat#
    T3mat = np.zeros(shape = (m,m,m,m))
    iterations = 0
    convergence = 0
    pOld = 1E10
    pNew = 0
    
    vals = np.random.rand(m,m,m,m)
    T3mat = np.log(vals/np.sum(vals, axis=3)[:,:,:,None])
    
    #Stop iterations when log(p(x_1:n)) differs by tol between iterations#
    while convergence == 0:
        
        #Perform forward and backward algorithms# 
        alpha = forwardAlg3(n, m, k, pi, Tmat, T2mat, T3mat, phi, x)
        beta = backwardAlg3(n, m, k, pi, Tmat, T3mat, phi, x)
        pNew = pForward3(m,alpha, x)
        ##M-Step##
        eta = np.zeros((n,m,m,m, m))
        #Update pi, phi and Tmat#

        for t in range(1, n-1):
            for i in range(0, m):
                for j in range(0, m):
                    for q in range(0, m):
                        for l in range(0,m):
                            eta[t,i,j,q, l] = alpha[t,i,j, q] + T3mat[i,j,q,l] + phi[l, x[t+1]] + beta[t+1, j, q, l] - pNew
        
        for i in range(0, m):
            for j in range(0, m):
                for q in range(0, m):
                    for l in range(0,m):
                            T3mat[i,j,q,l] = logSumExp(eta[1::,i,j, q, l]) - logSumExp(eta[1::,i,j,q,:])
        print(iterations)
        criteria = abs(pOld - pNew)
        if criteria < tol:
            convergence = 1
        elif iterations > 1000:
            convergence = 1
        else:
            convergence = 0
            pOld = pNew
            iterations +=1
    return (iterations, np.exp(T3mat))
