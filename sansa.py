#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sklearn.utils.sparsefuncs as spfuncs
from sklearn.preprocessing import LabelEncoder
import sksparse.cholmod as cholmod
from sansa_inv.a_inv import ainv_L
from sansa_inv.utils import sparsify
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
    
    def NDCG(self,pred,true,k=100):
        n_users = pred.shape[0]
        dcg_scores = np.zeros(n_users)
        tp = 1. / np.log2(np.arange(2, k + 2))
        for i in range(n_users):
            user_scores_sparse = pred[i]
            user_scores_dense = user_scores_sparse.toarray().flatten()
            top_k_indices_unsorted = np.argpartition(-user_scores_dense, k)[:k]
            top_k_scores = user_scores_dense[top_k_indices_unsorted]
            top_k_indices = top_k_indices_unsorted[np.argsort(-top_k_scores)]
            gains = true[i, top_k_indices].toarray().flatten()
            dcg_scores[i] = np.sum(gains * tp)
        idcg = np.array([tp[:min(m, k)].sum() for m in true.getnnz(axis=1)])
        ndcg = np.divide(dcg_scores, idcg, out=np.zeros_like(dcg_scores, dtype=float), where=idcg!=0)
        return ndcg

    def Recall(self, pred, true, k):
        n_users = pred.shape[0]
        hit_counts = np.zeros(n_users)
        for i in range(n_users):
            user_scores = pred[i].toarray().flatten()
            ids_true = true[i].indices
            if user_scores.size == 0 or ids_true.size == 0:
                continue
            top_k_indices = np.argpartition(-user_scores, k)[:k]
            hit_counts[i] = np.isin(top_k_indices, ids_true).sum()
        true_counts = true.getnnz(axis=1)
        denominator = np.minimum(k, true_counts)
        recall = np.divide(hit_counts, denominator, out=np.zeros_like(hit_counts, dtype=float), where=denominator != 0)
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
    
class SANSA:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df['uid'])
        items = self.item_enc.fit_transform(df['sid'])
        return users, items

    def _compute_inv_propensity_logsigmoid(self,freqs: np.ndarray,
                                           beta: float = 0.8) -> np.ndarray:
        """
        freqs : 1-D array (n_items,)
        Returns: w_i = 1 / σ(α + β log(freq+1))
        """
        logf  = np.log(freqs + 1)
        alpha = -beta * (logf.min() + logf.max()) / 2.0
        p_i   = 1.0 / (1.0 + np.exp(-(alpha + beta * logf)))
        return 1.0 / p_i
    
    def ldlt(
        self,X_T,l2,target_density,
        ):
        factor = cholmod.analyze_AAt(
        X_T
        )
        factor.cholesky_AAt_inplace(
            X_T,
            beta=l2,
            )
        p = factor.P()
        L, D = factor.L_D()
        L = L.tocsc()
        del factor
        # 2. Drop small values from L
        L = sparsify(L, L.shape[0], L.shape[1], target_density)
        
        return L, D, p
    
    def construct_weights(self,X_T,lambda_,wflg,wbeta,density,scans,finetune):
        ainv_params={
            "umr_scans": scans,
            "umr_finetune_steps": finetune,
            "umr_loss_threshold": 1e-4,
        }
        (L, D, p)= self.ldlt(
            X_T,
            l2=lambda_,
            target_density=density,
        )
        item_freq = np.ravel(X_T.T.sum(axis=0))
        #caluculate log sigmoid ips
        wts= self._compute_inv_propensity_logsigmoid(item_freq,wbeta)
        del X_T

        # 2. Compute approximate inverse of L using selected method
        L_inv = ainv_L(
            L,
            target_density=density,
            method_params=ainv_params,
        )  # this returns a pruned matrix
        # Garbage collect L
        del L
        # 3. Construct W = L_inv @ P
        inv_p = np.argsort(p)
        W = L_inv[:, inv_p]
        # Garbage collect L_inv
        del L_inv

        # 4. Construct W_r (A^{-1} = W.T @ W_r)
        W_r = W.copy()
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
            spfuncs.inplace_row_scale(W_r, 1 / D.diagonal())

        # 5. Extract diagonal entries
        diag = W.copy()
        diag.data = diag.data**2
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
            spfuncs.inplace_row_scale(diag, 1 / D.diagonal())
        diagsum = diag.sum(axis=0)  # original
        del diag
        diag = np.asarray(diagsum)[0]
        if wflg:
            scale = (-1.0 / diag) * wts
            with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
                spfuncs.inplace_column_scale(W_r, scale)    
        # 6. Divide columns of the inverse by negative diagonal entries
        # Due to associativity of matrix multiplication, this is equivalent to dividing the columns of W by negative diagonal entries
        else:
            with np.errstate(
                divide="ignore"
                ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
                spfuncs.inplace_column_scale(W_r, -1 / diag)
        # Return list of weight matrices [W.T, W_r]
        return [W.T.tocsr(), W_r.tocsr()]
        
        
    
    def fit_sansa(self, df, lambda_=500, wflg=False, wbeta=0.7, density=0.01, scans=1, finetune=5):
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0])
        X = csr_matrix((values, (users, items)))
        w1,w2=self.construct_weights(X.T,lambda_,wflg,wbeta,density,scans,finetune)
        B = w1 @ w2
        diag_indices = np.diag_indices(B.shape[0])
        B[diag_indices] = 0
        self.B = B
        
def run_experiment(df_train, df_test_tr, test_data_te, eval_obj, n_items,
                   lambda_, wflg, wbeta, density, scans, finetune):
    model = SANSA()
    model.fit_sansa(df_train, lambda_=lambda_, wflg=wflg, wbeta=wbeta,
                    density=density, scans=scans, finetune=finetune)

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
    coo = pred.tocoo()
    df = pd.DataFrame({'user': coo.row, 'item': coo.col, 'score': coo.data})
    topk = lambda k: df.groupby('user', group_keys=False).apply(lambda x: x.nlargest(k, 'score'))['item'].to_numpy()
    cov = lambda k: np.unique(topk(k)).shape[0] / n_items
    return recall20, recall50, ndcg100, cov(10), cov(30), cov(50), cov(100)


def main():
    parser = argparse.ArgumentParser("SANSA Runner")
    parser.add_argument("--dataset", type=str, default="ml-20m")
    parser.add_argument("--wflg", action="store_true")
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--lambda", type=float, default=500.0, dest="lambda_")
    parser.add_argument("--wbeta", type=float, default=0.7)
    parser.add_argument("--scans", type=int, default=1)
    parser.add_argument("--finetune", type=int, default=2)
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
                             args.lambda_, args.wflg, args.wbeta, args.density, args.scans, args.finetune)

    print(f"\nModel: SANSA, Weighted: {args.wflg}, Density: {args.density}, Scans: {args.scans}, Finetune: {args.finetune}")
    print(f"Recall@20:{results[0]:.5f} | Recall@50:{results[1]:.5f} | NDCG@100:{results[2]:.5f}")
    print(f"Coverage@10:{results[3]:.5f} | Coverage@30:{results[4]:.5f} | Coverage@50:{results[5]:.5f} | Coverage@100:{results[6]:.5f}")

if __name__ == "__main__":
    main()
