import numpy as np
import pandas as pd
import tqdm
import sys
import multiprocessing
from scipy import linalg
from PR import test_gauss_bonf, test_ivx_bonf, test_la
import os

ALPHA = 0.1
ALPHA_1 = 0.05
ALPHA_2 = 0.05


def dgp(gam, sig, n, nsim=1):
    d = sig.shape[0]
    X = np.zeros((nsim, n, d))
    gam_powers = np.hstack([np.linalg.matrix_power(gam, i) for i in range(n - 1)])
    gam_triang = np.vstack(
        [
            np.pad(gam_powers, [(0, 0), (d * i, 0)])[:, : d * (n - 1)]
            for i in range(n - 1)
        ]
    )
    eps = np.random.multivariate_normal(np.zeros(d), sig, size=(nsim, n - 1))
    eps_flat = eps.reshape((nsim, -1))
    X[:, 1:, :] = eps_flat.dot(gam_triang).reshape((nsim, n - 1, d))
    if nsim == 1:
        return X.reshape((n, d))
    return X


def sample_gam(d, delt):
    Lam = np.diag([1] * d)
    U = np.random.uniform(size=(d, d))
    Gamma = linalg.solve(U, Lam).dot(U)
    Gamma = np.vstack([np.zeros((1, d)), Gamma])
    Gamma = np.hstack([np.array([0] + [delt] * d).reshape((d + 1, 1)), Gamma])
    return Gamma


def get_test(n, d):
    Sigma = np.eye(d + 1) / 2 + 1 / 2
    out = []
    for i in range(20):
        delt = (i + 1) / 200
        Gamma = sample_gam(d, delt)
        X = dgp(Gamma, Sigma, n)
        try:
            la_out = test_la(X, ALPHA)
            out_la = {
                "t_sq": la_out["t_sq"],
                "thr": la_out["thr"],
                "reject": la_out["t_sq"] > la_out["thr"],
                "method": "la",
                "delt": delt,
                "n": n,
                "d": d,
            }
        except:
            out_la = {
                "t_sq": np.nan,
                "thr": np.nan,
                "reject": np.nan,
                "method": "la",
                "delt": delt,
                "n": n,
                "d": d,
            }
        try:
            ivx_out = test_ivx_bonf(X, ALPHA_1, ALPHA_2, 0.9)
            out_ivx = {
                "t_sq": ivx_out["t_sq_min"],
                "thr": ivx_out["thr"],
                "reject": ivx_out["t_sq_min"] > ivx_out["thr"],
                "method": "ivx",
                "delt": delt,
                "n": n,
                "d": d,
            }
        except:
            out_ivx = {
                "t_sq": np.nan,
                "thr": np.nan,
                "reject": np.nan,
                "method": "ivx",
                "delt": delt,
                "n": n,
                "d": d,
            }
        try:
            gauss_out = test_gauss_bonf(
                X,
                ALPHA_1,
                ALPHA_2,
                viol_tol=1e-3,
                obj_tol=1e-3,
                opt_tol=1e-3,
                max_iter=50,
                verbose=True,
            )
            out_gauss = {
                "t_sq": gauss_out["t_sq_min"],
                "thr": gauss_out["thr"],
                "reject": gauss_out["t_sq_min"] > gauss_out["thr"],
                "method": "gauss",
                "delt": delt,
                "n": n,
                "d": d,
            }
        except:
            out_gauss = {
                "t_sq": np.nan,
                "thr": np.nan,
                "reject": np.nan,
                "method": "gauss",
                "delt": delt,
                "n": n,
                "d": d,
            }
        out += [out_la, out_ivx, out_gauss]
        print(delt)
    return out


def main(argv):
    exp_num = int(argv[1])
    n = int(argv[2])
    d = int(argv[3])
    nsim = int(argv[4])
    name = (
        "./results/PR_power/n_"
        + str(n)
        + "/dim_"
        + str(d)
        + "/num_"
        + str(exp_num)
        + "_rw.csv"
    )
    """with multiprocessing.Pool(processes=processes) as pool:
        # res = pool.map(boot_sim, a_seq * nsim)
        print("mapping...")
        results = tqdm.tqdm(pool.imap(get_test, [(n, d, delt)] * nsim), total=nsim)
        print("running...")
        res = list(results)
        print("done")"""
    res = [get_test(n, d) for i in range(nsim)]
    res = [item for sublist in res for item in sublist]
    res_df = pd.DataFrame(res)
    res_df.to_csv(name, header=True)
    return None


if __name__ == "__main__":
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"
    main(sys.argv)
