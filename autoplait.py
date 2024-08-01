import numpy as np
from hmmlearn import hmm

cF = 4 * 8
MINK = 1
MAXK = 8
INITIALCUT = 10

'''
GaussianHMM tips:
    means: 各状態における各次元の正規分布の平均 (次元数 * 状態数)
    covars: 各状態における各次元の正規分布の分散 (次元数 * 次元数 * 状態数 = 分散の対角行列(共分散は0) * 状態数)
    transmat: 遷移確率 (状態数 * 状態数)
    startprob: 初期確率 (状態数)
'''

    
class Regime:
    def __init__(self, X, k):
        self.model = hmm.GaussianHMM(k)
        self.model.fit(X)

    @property
    def k(self):
        return self.model.n_components
    
    def b(self, x, i):
        mean = self.model.means_[i]
        var = np.diag(self.model.covars_[i])
        return 1. / np.sum(np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * np.sum((x - mean) ** 2 / var))
    
    def B(self, x):
        return np.array([self.b(x, i) for i in range(self.k)])
    
    def StartP(self, x):
        return self.model.startprob_ * self.B(x)
    
    def NextP(self, x, prevP):
        return np.array([np.max(prevP * self.model.transmat_[:, i]) for i in range(self.k)]) * self.B(x)
    
    def CostM(self):
        k = self.k
        return np.log2(k) + cF * (k + k ** 2 + 2 * k * self.d)
    
    def CostC(self, X):
        P = self.StartP(X[0])
        for x in X[1:]:
            P = self.NextP(x, P)
        return -np.log2(np.max(P))



def estimate_regime(X):
    regime = Regime(X, MINK)
    CostC = regime.CostC(X)
    for k in range(MINK + 1, MAXK + 1):
        new_regime = Regime(X, k)
        new_CostC = new_regime.CostC(X)
        if CostC > new_CostC:
            regime = new_regime
            CostC = new_CostC
    return regime, CostC


def estimate_delta(X, regime1, regime2):
    pass


class AutoPlait:

    def __init__(self, X):
        '''
        model = hmm.GaussianHMM(5)
        model.fit(X)

        print(model.means_)
        print(model.covars_)
        print(model.transmat_)
        print(model.startprob_)
        '''
        self.d = X.shape[1]
        print(self.d)


    def CutPointSearch(X, regime1, regime2, delta):
        P1 = delta[0][0] * regime1.StartP(X[0])
        P2 = delta[1][1] * regime2.StartP(X[0])
        L1 = [[] for i in range(regime1.k)]
        L2 = [[] for i in range(regime2.k)]

        for t, x in enumerate(X[1:]):
            prev_max_p1_index = np.argmax(P1)
            prev_max_p2_index = np.argmax(P2)
            prev_max_p1 = P1[prev_max_p1_index]
            prev_max_p2 = P2[prev_max_p2_index]

            next_P1 = np.zeros(regime1.k)
            next_P2 = np.zeros(regime2.k)

            for i in range(regime1.k):
                either_trans = P1 * regime1.model.transmat_[:, i]
                max_trans_index = np.argmax(either_trans)
                max_trans = delta[0][0] * either_trans[max_trans_index]
                max_switch = delta[1][0] * prev_max_p2 * regime1.model.startprob_[i]
                if max_trans > max_switch:
                    L1[i] = L1[max_trans_index].copy()
                    next_P1[i] = max_trans * regime1.b(x, i)
                else:
                    L1[i] = L2[prev_max_p2_index].copy() + [t]
                    next_P1[i] = max_switch * regime1.b(x, i)

            for u in range(regime2.k):
                either_trans = P2 * regime2.model.transmat_[:, u]
                max_trans_index = np.argmax(either_trans)
                max_trans = delta[1][1] * either_trans[max_trans_index]
                max_switch = delta[0][1] * prev_max_p1 * regime2.model.startprob_[u]
                if max_trans > max_switch:
                    L2[u] = L2[max_trans_index].copy()
                    next_P2[u] = max_trans * regime2.b(x, u)
                else:
                    L2[u] = L1[prev_max_p1_index].copy() + [t]
                    next_P2[u] = max_switch * regime2.b(x, u)
            
            P1 = next_P1
            P2 = next_P2

        max_p1_index = np.argmax(P1)
        max_p2_index = np.argmax(P2)
        max_p1 = P1[max_p1_index]
        max_p2 = P2[max_p2_index]

        Lbest = (L1[max_p1_index] if (max_p1 > max_p2) else L2[max_p2_index]) + [X.shape[1]]
        S1 = []
        S2 = []
        m1 = (len(Lbest) + 1) // 2
        m2 = len(Lbest) // 2
        ts = 0
        for i, l in enumerate(Lbest):
            if i % 2 == 0:
                S1.append((ts, l))
            else:
                S2.append((ts. l))
            ts = l

        if max_p1 > max_p2:
            if len(Lbest) % 2 == 1:
                return m1, m2, S1, S2
            else:
                return m2, m1, S2, S1
        else:
            if len(Lbest) % 2 == 1:
                return m2, m1, S2, S1
            else:
                return m1, m2, S1, S2




    def RegimeSplit(self, X):

        # model initialization
        half = X.shape[0] // 2
        span = X.shape[0] // INITIALCUT
        regime1, CostC1 = estimate_regime(X[:half])
        regime2, CostC2 = estimate_regime(X[half:])
        delta = [[1. / half, (half - 1.) / half], [(half - 1.) / half, 1. / half]]

        for l in range(2, INITIALCUT + 1):
            for s in range(0, INITIALCUT - l + 1):
                for c in range(1, l):
                    start = s * span
                    middle = (s + c) * span
                    end = (s + l) * span
                    new_regime1, new_CostC1 = estimate_regime(X[start:middle])
                    new_regime2, new_CostC2 = estimate_regime(X[middle:end])
                    if new_CostC1 + new_CostC2 < CostC1 + CostC2:
                        regime1 = new_regime1
                        regime2 = new_regime2
                        CostC1 = new_CostC1
                        CostC2 = new_CostC2
                        delta = [[1. / (middle - start), (middle - start - 1.) / (middle - start)], [(end - middle - 1.) / (end - middle), 1. / (end - middle)]]
        

        # model estimation
