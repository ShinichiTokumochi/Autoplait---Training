import numpy as np
from hmmlearn import hmm

cF = 4 * 8
MINK = 1
MAXK = 8
INITIALCUT = 3
ZERO = 1.e-10
HMM_N_ITER=1
MAXEM=10

'''
GaussianHMM tips:
    means: 各状態における各次元の正規分布の平均 (次元数 * 状態数)
    covars: 各状態における各次元の正規分布の分散 (次元数 * 次元数 * 状態数 = 分散の対角行列(共分散は0) * 状態数)
    transmat: 遷移確率 (状態数 * 状態数)
    startprob: 初期確率 (状態数)
'''

# Costなどは計算過程で事前にlog化しないと浮動小数点の問題で計算が難しい
    
class Regime:
    def __init__(self, X, k):
        self.model = hmm.GaussianHMM(k, n_iter=HMM_N_ITER)
        self.model.fit(X)
        self.d = X.shape[1]

    @property
    def k(self):
        return self.model.n_components
    
    def log2b(self, x, i):
        mean = self.model.means_[i]
        var = np.diag(self.model.covars_[i])
        return -0.5 * (len(mean) * np.log2(2 * np.pi) + np.sum(np.log2(var))) + (-0.5 * np.sum((x - mean) ** 2 / var))
    
    def log2B(self, x):
        return np.array([self.log2b(x, i) for i in range(self.k)])
    
    def Startlog2P(self, x):
        return np.log2(self.model.startprob_ + ZERO) + self.log2B(x) # startprobが0のとき用のZERO
    
    def Nextlog2P(self, x, prevlog2P):
        return np.array([np.max(prevlog2P + np.log2(self.model.transmat_[:, i])) for i in range(self.k)]) + self.log2B(x)
    
    def CostM(self):
        k = self.k
        return np.log2(k) + cF * (k + k ** 2 + 2 * k * self.d)
    
    def CostC(self, X):
        log2P = self.Startlog2P(X[0])
        for x in X[1:]:
            log2P = self.Nextlog2P(x, log2P)
        return -np.max(log2P)


def estimate_regime(X, s = 0, e = None):
    if e == None:
        e = len(X)
    regime = Regime(X[s:e], MINK)
    CostC = regime.CostC(X)
    for k in range(MINK + 1, MAXK + 1):
        if e - s <= k * k:
            break
        new_regime = Regime(X[s:e], k)
        new_CostC = new_regime.CostC(X)
        if CostC > new_CostC:
            regime = new_regime
            CostC = new_CostC
    return regime, CostC


def CutPointSearch(X, regime1, regime2, delta):
    log2P1 = np.log2(delta[0][0]) + regime1.Startlog2P(X[0])
    log2P2 = np.log2(delta[1][1]) + regime2.Startlog2P(X[0])
    L1 = [[] for i in range(regime1.k)]
    L2 = [[] for i in range(regime2.k)]

    for t, x in enumerate(X[1:]):
        prev_max_p1_index = np.argmax(log2P1)
        prev_max_p2_index = np.argmax(log2P2)
        prev_max_log2p1 = log2P1[prev_max_p1_index]
        prev_max_log2p2 = log2P2[prev_max_p2_index]

        next_log2P1 = np.zeros(regime1.k)
        next_log2P2 = np.zeros(regime2.k)
        next_L1 = [[] for i in range(regime1.k)]
        next_L2 = [[] for i in range(regime2.k)]

        for i in range(regime1.k):
            either_trans = log2P1 + np.log2(regime1.model.transmat_[:, i])
            max_trans_index = np.argmax(either_trans)
            max_log2trans = np.log2(delta[0][0]) + either_trans[max_trans_index]
            max_log2switch = np.log2(delta[1][0]) + prev_max_log2p2 + np.log2(regime1.model.startprob_[i] + ZERO)
            if max_log2trans > max_log2switch:
                next_L1[i] = L1[max_trans_index].copy()
                next_log2P1[i] = max_log2trans + regime1.log2b(x, i)
            else:
                next_L1[i] = L2[prev_max_p2_index].copy() + [t]
                next_log2P1[i] = max_log2switch + regime1.log2b(x, i)

        for u in range(regime2.k):
            either_trans = log2P2 + np.log2(regime2.model.transmat_[:, u])
            max_trans_index = np.argmax(either_trans)
            max_log2trans = np.log2(delta[1][1]) + either_trans[max_trans_index]
            max_log2switch = np.log2(delta[0][1]) + prev_max_log2p1 + np.log2(regime2.model.startprob_[u] + ZERO)
            if max_log2trans > max_log2switch:
                next_L2[u] = L2[max_trans_index].copy()
                next_log2P2[u] = max_log2trans + regime2.log2b(x, u)
            else:
                next_L2[u] = L1[prev_max_p1_index].copy() + [t]
                next_log2P2[u] = max_log2switch + regime2.log2b(x, u)
            
        log2P1 = next_log2P1
        log2P2 = next_log2P2
        L1 = next_L1
        L2 = next_L2

    max_p1_index = np.argmax(log2P1)
    max_p2_index = np.argmax(log2P2)
    max_log2p1 = log2P1[max_p1_index]
    max_log2p2 = log2P2[max_p2_index]

    Lbest = (L1[max_p1_index] if (max_log2p1 > max_log2p2) else L2[max_p2_index]) + [X.shape[0]]
    S1 = []
    S2 = []
    m1 = (len(Lbest) + 1) // 2
    m2 = len(Lbest) // 2
    ts = 0
    for i, l in enumerate(Lbest):
        if i % 2 == 0:
            S1.append((ts, l))
        else:
            S2.append((ts, l))
        ts = l
    
    if max_log2p1 > max_log2p2:
        if len(Lbest) % 2 == 1:
            return m1, m2, S1, S2
        else:
            return m2, m1, S2, S1
    else:
        if len(Lbest) % 2 == 1:
            return m2, m1, S2, S1
        else:
            return m1, m2, S1, S2


def RegimeSplit(X):

    # model initialization
    half = X.shape[0] // 2
    span = X.shape[0] // INITIALCUT
    regime1, CostC1 = estimate_regime(X, 0, half)
    regime2, CostC2 = estimate_regime(X, half)
    #delta = [[1. / half, (half - 1.) / half], [(half - 1.) / half, 1. / half]]
    delta = [[1., ZERO,], [ZERO, 1.]]

    for l in range(2, INITIALCUT + 1):
        for s in range(0, INITIALCUT - l + 1):
            for c in range(1, l):
                start = s * span
                middle = (s + c) * span
                end = (s + l) * span
                new_regime1, new_CostC1 = estimate_regime(X, start, middle)
                new_regime2, new_CostC2 = estimate_regime(X, middle, end)

                if new_CostC1 + new_CostC2 < CostC1 + CostC2:
                    regime1 = new_regime1
                    regime2 = new_regime2
                    CostC1 = new_CostC1
                    CostC2 = new_CostC2
                    #delta = [[(middle - start - 1.) / (middle - start),  1. / (middle - start)], [1. / (end - middle), (end - middle - 1.) / (end - middle)]]
    
    CostT = CostC1 + CostC2 + regime1.CostM() + regime2.CostM()

    for i in range(MAXEM):
        m1, m2, S1, S2 = CutPointSearch(X, regime1, regime2, delta)
        S1 = np.array(S1)
        S2 = np.array(S2)
        S1_len = S1[:, 1] - S1[:, 0]
        S2_len = S2[:, 1] - S2[:, 0]
        S1_total_len = np.sum(S1_len)
        S2_total_len = np.sum(S2_len)

        target_S1 = S1[np.argmax(S1_len)]
        target_S2 = S2[np.argmax(S2_len)]

        new_regime1, new_CostC1 = estimate_regime(X[target_S1[0]:target_S1[1]])
        new_regime2, new_CostC2 = estimate_regime(X[target_S2[0]:target_S2[1]])
        new_CostT = new_CostC1 + new_CostC2 + new_regime1.CostM() + new_regime2.CostM()

        if new_CostT < CostT:
            regime1 = new_regime1
            regime2 = new_regime2
            CostT = new_CostT
            delta = [[max(S1_total_len - m1, 1) / S1_total_len,  max(m1, 1) / S1_total_len], [max(m2, 1) / S2_total_len, max(S2_total_len - m2, 1) / S2_total_len]]
        else:
            break

    return S1, S2


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

        # model estimation
