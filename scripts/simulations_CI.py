import numpy as np
import pandas as pd
import tqdm
import sys
import multiprocessing
from scipy import linalg
from CI import ci_gauss_proj, ci_la, ci_ivx_proj
import os


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


def sample_gam(n, d):
    Lam = np.diag([1 - 1 / (n ** (1 / i)) for i in range(d - 1, 0, -1)] + [1])
    U = np.random.uniform(size=(d, d))
    return linalg.solve(U, Lam).dot(U)


def get_ci(n, d):
    p = np.array([1] + [0] * (d**2 - 1))
    Sigma = np.eye(d) / 2 + 1 / 2
    Gamma = sample_gam(n, d)
    X = dgp(Gamma, Sigma, n)
    try:
        ci_la_out = ci_la(X, p)
        out_la = {
            "ci_l": ci_la_out[0],
            "ci_u": ci_la_out[1],
            "method": "la",
            "converged": np.nan,
            "n": n,
            "d": d,
            "gam": Gamma[0, 0],
        }
    except:
        out_la = {
            "ci_l": np.nan,
            "ci_u": np.nan,
            "method": "la",
            "converged": np.nan,
            "n": n,
            "d": d,
            "gam": Gamma[0, 0],
        }
    try:
        ci_ivx_out = ci_ivx_proj(X, p)
        out_ivx = {
            "ci_l": ci_ivx_out[0],
            "ci_u": ci_ivx_out[1],
            "method": "ivx",
            "converged": np.nan,
            "n": n,
            "d": d,
            "gam": Gamma[0, 0],
        }
    except:
        out_ivx = {
            "ci_l": np.nan,
            "ci_u": np.nan,
            "method": "ivx",
            "converged": np.nan,
            "n": n,
            "d": d,
            "gam": Gamma[0, 0],
        }
    try:
        g_res = ci_gauss_proj(X, p, viol_tol=3e-2, max_iter=50, verbose=True)
        conv_res = int(
            (g_res["status_u"] == "converged") & (g_res["status_l"] == "converged")
        )
        ci_gauss_out = (g_res["ci_l"], g_res["ci_u"])
        out_gauss = {
            "ci_l": ci_gauss_out[0],
            "ci_u": ci_gauss_out[1],
            "method": "gauss",
            "converged": conv_res,
            "n": n,
            "d": d,
            "gam": Gamma[0, 0],
        }
    except:
        out_gauss = {
            "ci_l": np.nan,
            "ci_u": np.nan,
            "method": "gauss",
            "converged": np.nan,
            "n": n,
            "d": d,
            "gam": Gamma[0, 0],
        }
    return [out_la, out_ivx, out_gauss]


def main(argv):
    num_exp = int(argv[1])
    n = int(argv[2])
    d = int(argv[3])
    nsim = int(argv[4])
    name = (
        "./results/CI/n_" + str(n) + "/dim_" + str(d) + "/num_" + str(num_exp) + ".csv"
    )
    """with multiprocessing.Pool(processes=processes) as pool:
        # res = pool.map(boot_sim, a_seq * nsim)
        print("mapping...")
        results = tqdm.tqdm(pool.imap(get_ci, [(n, d)] * nsim), total=nsim)
        print("running...")
        res = list(results)
        print("done")"""
    res = [get_ci(n, d) for i in range(nsim)]
    res = [item for sublist in res for item in sublist]
    res_df = pd.DataFrame(res)
    res_df.to_csv(name, header=True)
    return None


if __name__ == "__main__":
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["OMP_NUM_THREADS"] = "1"
    main(sys.argv)
