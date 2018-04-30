import numpy as np
import scipy.stats
from pylab import *
from numpy.random import *

T = 100 # 試行回数
list_X = [0,1]
list_Y = list(range(1,6+1))
print(list_X, list_Y)

alpha = [[0.0 for x in list_X] for t in range(T+1)]
beta = [[0.0 for x in list_X] for t in range(T+1)]
gamma = [[0.0 for x in list_X] for t in range(T+1)]


X = [0 for i in range(T+2)] # 隠れ変数
Y = [0 for i in range(T+1)] # 観測変数

h = [[0.0 for j in range(2)] for i in range(2)] # h[i][j]=P(Xt=i|Xt-1=j)
p = [[0.0 for j in range(2)] for i in range(6+1)] # p[i][j]=P(Yt=i|Xt=j)

h[0][0] = 0.9
h[0][1] = 0.1
h[1][0] = 0.1
h[1][1] = 0.9

for i in range(7):
    p[i][0] = 1.0/6.0
    p[i][1] = 0.1
p[6][1] = 0.5



def trans_X(j):
    x_rand = rand()
    if x_rand < h[0][j]:
        return 0
    else:
        return 1
def trans_Y(j):
    y_rand = rand()
    if y_rand < p[1][j]:
        return 1
    elif y_rand < p[1][j]+p[2][j]:
        return 2
    elif y_rand < p[1][j]+p[2][j]+p[3][j]:
        return 3
    elif y_rand < p[1][j]+p[2][j]+p[3][j]+p[4][j]:
        return 4
    elif y_rand < p[1][j]+p[2][j]+p[3][j]+p[4][j]+p[5][j]:
        return 5
    else:
        return 6
def calc_alpha(t, x_now):
    result = 0.0
    for x in list_X:
        result += alpha[t-1][x]*h[x_now][x]
    return result*p[Y[t]][x_now]
def calc_beta(t, x_now):
    result = 0.0
    for x in list_X:
        result += beta[t+1][x]*p[Y[t+1]][x]*h[x][x_now]
    return result

if __name__ == '__main__':
    # 観測値系列の生成
    # t=1では通常サイコロを使うことにする．P(X_1)=1.0
    X_1 = 0

    X[1] = X_1
    for t in range(1,T+1):
        Y[t] = trans_Y(X[t])
        X[t+1] = trans_X(X[t])
    print(X)
    print(Y)

    # alphaの計算
    alpha[1][X_1] = p[Y[1]][X_1]*1.0
    for t in range(2,T+1):
        for x in list_X:
            alpha[t][x] = calc_alpha(t, x)

    # betaの計算
    for x in list_X:
        beta[T][x] = 1.0
    for t in range(T-1,0,-1):
        for x in list_X:
            beta[t][x] = calc_beta(t, x)

    # gammaの計算
    P_X = 0.0
    for x in list_X:
        P_X += alpha[T][x]
    print(P_X)
    for t in range(1,T+1):
        for x in list_X:
            gamma[t][x] = alpha[t][x]*beta[t][x]/P_X

    # イカサマ事後確率 P(Xt=1)
    gamma_list = [0.0 for t in range(T+1)]
    X_list = [0.0 for t in range(T+1)]
    for t in range(1,T+1):
        X_list[t] = X[t]
        gamma_list[t] = gamma[t][1]
        print("Trials=", t, gamma[t][1])

    # グラフに出力
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ln1=ax1.plot(gamma_list,label=r'Prosterior probability', marker="+", lw=5, color='b')
    ax1.set_ylabel('Prosterior probability',fontsize = 20)
    plt.tick_params(labelsize = 18)
    ax2 = ax1.twinx()
    ln2=ax2.plot(X_list,label=r'$X_t$', marker="o", lw=5, color='r')
    ax2.set_ylabel(r'$X_t$',fontsize = 20)
    plt.tight_layout()
    plt.tick_params(labelsize = 18)
    ax1.set_xlabel("Trials", fontsize = 20)
    plt.subplots_adjust(hspace=0.7,bottom=0.2)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2,loc='lower right',fontsize = 15)# loc='center right')
    plt.xlim([1,10])
    plt.yticks([0, 1, 1])
    plt.tight_layout()
    plt.savefig('trials.png')
    plt.show()
