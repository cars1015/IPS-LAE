#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import bottleneck as bn
from sklearn.preprocessing import LabelEncoder

#==================================================
# Evaluation Class
#==================================================
class EVAL:
    def __init__(self):
        pass

    def getLabel(self, test, pred):
        res = []
        for i in range(len(test)):
            gt = test[i]
            topk = pred[i]
            match = list(map(lambda x: x in gt, topk))
            res.append(np.array(match).astype("float"))
        return np.array(res).astype('float')

    def NDCG(self, pred, true, k=100):
        n = pred.shape[0]
        part = bn.argpartition(-pred, k, axis=1)
        topk_part = pred[np.arange(n)[:, None], part[:, :k]]
        idx_sort = np.argsort(-topk_part, axis=1)
        topk = part[np.arange(n)[:, None], idx_sort]
        tp = 1. / np.log2(np.arange(2, k + 2))
        dcg = (true[np.arange(n)[:, None], topk].toarray() * tp).sum(axis=1)
        idcg = np.array([tp[:min(m, k)].sum() for m in true.getnnz(axis=1)])
        return dcg / idcg

    def Recall(self, pred, true, k):
        n = pred.shape[0]
        idx = bn.argpartition(-pred, k, axis=1)
        pred_bin = np.zeros_like(pred, dtype=bool)
        pred_bin[np.arange(n)[:, None], idx[:, :k]] = True
        true_bin = (true > 0).toarray()
        hit = np.logical_and(true_bin, pred_bin).sum(axis=1).astype(np.float32)
        recall = hit / np.minimum(k, true_bin.sum(axis=1))
        return recall

    def load_tr_te_data(self, tr_path, te_path, n_items):
        tr = pd.read_csv(tr_path)
        te = pd.read_csv(te_path)
        start = min(tr['uid'].min(), te['uid'].min())
        end = max(tr['uid'].max(), te['uid'].max())
        r_tr, c_tr = tr['uid'] - start, tr['sid']
        r_te, c_te = te['uid'] - start, te['sid']
        X_tr = csr_matrix((np.ones_like(r_tr), (r_tr, c_tr)), dtype='float64', shape=(end - start + 1, n_items))
        X_te = csr_matrix((np.ones_like(r_te), (r_te, c_te)), dtype='float64', shape=(end - start + 1, n_items))
        return X_tr, X_te

#==================================================
# Training
#==================================================
class CORR:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df['uid'])
        items = self.item_enc.fit_transform(df['sid'])
        return users, items

    def compute_inv_propensity_logsigmoid(self, X, beta=0.1):
        freqs = np.ravel(X.sum(axis=0))
        log_freqs = np.log(freqs + 1)
        min_log = np.min(log_freqs)
        max_log = np.max(log_freqs)
        alpha = -beta * (min_log + max_log) / 2
        logits = alpha + beta * log_freqs
        p_i = 1 / (1 + np.exp(-logits))
        return 1 / p_i

    def compute_inv_propensity_powerlaw(self, X, beta=0.4):
        pop = np.ravel(X.sum(axis=0))
        norm_pop = pop / np.max(pop)
        p = np.power(norm_pop, beta)
        return 1 / p

    def _apply_weighting(self, B, X, wflg, wtype, wbeta):
        if not wflg:
            return B
        if wtype == "logsigmoid":
            w = self.compute_inv_propensity_logsigmoid(X, beta=wbeta)
        elif wtype == "powerlaw":
            w = self.compute_inv_propensity_powerlaw(X, beta=wbeta)
        else:
            raise ValueError("Unknown weighting method")
        return B * w

    def fit_model(self, df, model_name, lambda_, alpha, drop_p, wflg, wtype, wbeta):
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0])
        X = csr_matrix((values, (users, items)))
        G = X.T.dot(X).toarray()
        diag_idx = np.diag_indices(G.shape[0])

        if model_name == "ease":
            G[diag_idx] += lambda_
            P = np.linalg.inv(G)
            B = P / -np.diag(P)

        elif model_name == "edlae":
            gamma = np.diag(G) * drop_p / (1 - drop_p) + lambda_
            G[diag_idx] += gamma
            P = np.linalg.inv(G)
            B = P / -np.diag(P)

        elif model_name == "rdlae":
            gamma = np.diag(G) * drop_p / (1 - drop_p) + lambda_
            G[diag_idx] += gamma
            P = np.linalg.inv(G)
            diag_P = np.diag(P)
            cond = (1 - gamma * diag_P) > alpha
            if np.sum(cond) == 0:
                raise ValueError("No diagonal entries satisfy condition in RDLAE.")
            lag = ((1 - alpha) / diag_P - gamma) * cond.astype(float)
            B = P * -(gamma + lag)

        else:
            raise ValueError("Unknown model name")

        B = self._apply_weighting(B, X, wflg, wtype, wbeta)
        B[diag_idx] = 0
        self.B = B
        self.X = X

#==================================================
# Experiment Runner
#==================================================
def run_experiment(df_train, df_test_tr, test_data_te, eval_obj, n_items,
                   lambda_, model_name, alpha, drop_p, wflg, wtype, wbeta):
    model = CORR()
    model.fit_model(df_train, model_name, lambda_, alpha, drop_p, wflg, wtype, wbeta)

    users_te = df_test_tr['uid'].values
    items_te = df_test_tr['sid'].values
    u_enc = LabelEncoder()
    users_id = u_enc.fit_transform(users_te)
    items_id = model.item_enc.transform(items_te)
    values = np.ones(df_test_tr.shape[0])
    shape = (u_enc.classes_.size, model.item_enc.classes_.size)
    X_te_csr = csr_matrix((values, (users_id, items_id)), shape=shape)
    pred = X_te_csr.dot(model.B)
    pred[X_te_csr.nonzero()] = -np.inf

    ndcg100 = np.mean(eval_obj.NDCG(pred, test_data_te, k=100))
    recall20 = np.mean(eval_obj.Recall(pred, test_data_te, k=20))
    recall50 = np.mean(eval_obj.Recall(pred, test_data_te, k=50))

    topk = lambda k: bn.argpartition(-pred, k, axis=1)[:, :k]
    cov = lambda k: np.unique(topk(k)).shape[0] / n_items
    return recall20, recall50, ndcg100, cov(10), cov(30), cov(50), cov(100)

#==================================================
# main
#==================================================
def main():
    parser = argparse.ArgumentParser("All in One file (no separate utils.py)")
    parser.add_argument("--dataset", type=str, default="ml-20m")
    parser.add_argument("--model", type=str, choices=["ease","edlae","rdlae"], default="ease")
    parser.add_argument("--lambda", type=float, default=500.0, dest="lambda_")
    parser.add_argument("--drop_p", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--wflg", action="store_true", help="Apply weighting on B")
    parser.add_argument("--wtype", type=str, choices=["powerlaw","logsigmoid"], default="logsigmoid")
    parser.add_argument("--wbeta", type=float, default=0.1, help="Weighting parameter beta")
    args = parser.parse_args()

    data_dir = f"./data_dir/{args.dataset}/pro_sg/"
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    df_test_tr = pd.read_csv(os.path.join(data_dir, "test_tr.csv"))
    unique_sid = [line.strip() for line in open(os.path.join(data_dir, "unique_sid.txt"))]
    n_items = len(unique_sid)

    eval_obj = EVAL()
    _, test_data_te = eval_obj.load_tr_te_data(
        os.path.join(data_dir, "test_tr.csv"),
        os.path.join(data_dir, "test_te.csv"),
        n_items
    )

    results = run_experiment(df_train, df_test_tr, test_data_te, eval_obj, n_items,
                                args.lambda_, args.model, args.alpha, args.drop_p, args.wflg, args.wtype, args.wbeta)
    print(f"\nModel: {args.model.upper()}, Weighting: {'None' if not args.wflg else args.wtype}")
    print(f"Recall@20:{results[0]:.5f} | Recall@50:{results[1]:.5f} | NDCG@100:{results[2]:.5f}")
    print(f"Coverage@10:{results[3]:.5f} | Coverage@30:{results[4]:.5f} | Coverage@50:{results[5]:.5f} | Coverage@100:{results[6]:.5f}")
if __name__ == "__main__":
    main()
