import numpy as np
from scipy import stats
from scipy import linalg
from scipy import optimize
import GPy


# Function for finding least squares estimators
def ls_est(X):
    n, _ = X.shape
    y = X[1:, :]
    x = X[: (n - 1), :]
    gam_hat, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    eps_hat = y - x.dot(gam_hat)
    sig_hat = eps_hat.T.dot(eps_hat) / n
    return gam_hat, sig_hat


# Function for computing test using lag-augmentation
def test_la(X, alpha):
    n, d = X.shape
    y = X[2:, :]
    x = np.hstack([X[1 : (n - 1), :], X[: (n - 2), :]])
    gam_hat, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    _, sig_hat = ls_est(X)
    sig_hat_la = np.kron(linalg.inv(sig_hat), sig_hat)
    gam_hat_vec = gam_hat[:d, :].reshape((-1,))
    C = np.kron(
        np.hstack([np.zeros((d - 1, 1)), np.eye(d - 1)]), np.array([1] + [0] * (d - 1))
    )
    Cg = C.dot(gam_hat_vec)
    CsC = C.dot(sig_hat_la).dot(C.T)
    t_sq = n * linalg.solve(CsC, Cg).dot(Cg)
    c = stats.chi2.ppf(1 - alpha, d - 1)
    return {"t_sq": t_sq, "thr": c, "reject": t_sq > c}


# Function from sampling from \Theta (sort of uniformly)
def sample_theta(k, d):
    d_sq = d**2
    theta_out = np.zeros((k, d_sq))
    for i in range(k):
        lam = np.random.uniform(-0.9, 1, size=d)
        u = np.random.uniform(-1, 1, size=d_sq)
        u_mat = u.reshape((d, d))
        theta_mat = linalg.solve(u_mat, np.diag(lam).T).dot(u_mat)
        theta_out[i, :] = theta_mat.reshape((-1,))
    return theta_out


# Function for sampling parameters close to the least squares estimator
def sample_theta_close(k, gam, pert=1e-1):
    d = gam.shape[0]
    d_sq = d**2
    theta_out = np.zeros((k, d_sq))
    for i in range(k):
        u = np.random.uniform(-pert, pert, size=d_sq)
        theta_out[i, :] = gam.reshape((-1,)) + u
    return theta_out


# Function for choosing initial points for optimization problem
def draw_initial_theta(theta_hash, f, r_max, r_min=1e-10, num_points=5):
    r_current = 1 / 2
    points = 0
    d_sq = theta_hash.shape[0]
    d = int(np.sqrt(d_sq))
    theta_out = np.zeros((num_points, d_sq))
    while (r_current > r_min) & (points < num_points):
        for j in range(2):
            try:
                u = np.array(
                    [
                        np.random.uniform(0, np.abs(r_max)[i] * r_current)
                        for i in range(d_sq)
                    ]
                )
            except:
                print(np.abs(r_max) * r_current)
                print(r_max)
                print(theta_hash)
                u = 0
            u = np.sign(r_max) * u
            if (not np.isnan(u).any()) or (not np.isfinite(u).all()):
                u = 0
            theta_draw = theta_hash + u
            if -f(theta_draw) > 1e-10:
                theta_out[points, :] = theta_draw
                points += 1
                if points >= num_points:
                    break
        if points >= num_points:
            break
        r_current /= 2
    theta_rand = sample_theta_close(2, theta_hash.reshape((d, d)), 1e-4)
    theta_out = np.vstack([theta_out, theta_rand])
    return theta_out


# Function for computing the standard normal t^2-stat
def cov_map(a, b):
    n = a.shape[0]
    s_ez = a.T.dot(b) / n
    s_zz = b.T.dot(b) / n
    norm = np.abs(s_zz).max()
    s_zz = s_zz / norm
    s_ez = s_ez / np.sqrt(norm)
    try:
        out = n * np.trace(np.linalg.solve(s_zz, s_ez.T).T.dot(s_ez.T))
    except:
        out = 0
    return out


# Function for evaluating c(\theta)
def c_function(n, d, sigma, alpha, nsim=1000):
    eps = np.array(np.random.multivariate_normal(np.zeros(d), sigma, size=(nsim, n)))
    eps_flat = eps[:, : (n - 1), :].reshape((nsim, -1))
    eps_up = eps[:, 1:, :]

    def f_1(gam):
        gam_mat = gam.reshape((d, d))
        gam_powers = np.hstack(
            [np.linalg.matrix_power(gam_mat, i) for i in range(n - 1)]
        )
        gam_triang = np.vstack(
            [
                np.pad(gam_powers, [(0, 0), (d * i, 0)])[:, : d * (n - 1)]
                for i in range(n - 1)
            ]
        )
        Z = eps_flat.dot(gam_triang).reshape((nsim, n - 1, d))
        ts = [cov_map(eps_up[i, :, :], Z[i, :, :]) for i in range(nsim)]
        return np.quantile(ts, alpha)

    def f_2(gam):
        gam_mat = gam.reshape((d, d))
        ts = np.zeros((nsim,))
        for i in range(nsim):
            Z = np.zeros((n - 1, d))
            for t in range(1, n - 1):
                Z[t, :] = Z[t - 1, :].dot(gam_mat) + eps[i, t - 1, :]
            ts[i] = cov_map(eps_up[i, :, :], Z)
        return np.quantile(ts, alpha)

    return f_2


# Function for evaluating g(\theta)
def eval_g(X, gam):
    n, d = X.shape
    x = X[: (n - 1), :]
    s_xx = x.T.dot(x) / n
    gam_mat = gam.reshape((d, d))
    gam_hat, sig_hat = ls_est(X)
    sig_hat_sq = linalg.sqrtm(sig_hat)
    t_sq = (gam_hat - gam_mat).T.dot(s_xx).dot((gam_hat - gam_mat))
    t_sq = np.linalg.solve(sig_hat_sq, np.linalg.solve(sig_hat_sq, t_sq).T).T
    return n * np.trace(t_sq)


# Function for fitting approximating GP
def fit_gp(theta, c):
    k, d_sq = theta.shape
    rbf = GPy.kern.RBF(input_dim=d_sq, ARD=True)
    matern = GPy.kern.Matern32(input_dim=d_sq, ARD=True)
    gpr = GPy.models.GPRegression(theta, c, rbf, normalizer=True)
    gpr.optimize()
    return gpr


# Helper function for proper reshaping
def rs_idx(d):
    return np.arange(d**2).reshape((d, d)).T.reshape((-1,))


# Function for determning bonferroni test-statistic
def t_sq_bon(X):
    n, d = X.shape
    y = X[1:, 0].reshape((n - 1, 1))
    X_til = X[:, 1:]
    x_til = X_til[:-1, :]
    x_til_ = X_til[1:, :]
    s_xy = x_til.T.dot(y) / n
    s_xx = x_til.T.dot(x_til) / n
    s_xx_ = x_til.T.dot(x_til_) / n
    _, sig_hat = ls_est(X)
    sig_x = sig_hat[1:, 1:]
    sig_y = sig_hat[0, 0].reshape((1, 1))
    sig_xy = sig_hat[1:, 0].reshape((d - 1, 1))
    sig_sol = linalg.solve(sig_x, sig_xy)
    sig_hat_y = sig_y - sig_xy.T.dot(sig_sol)
    A = s_xy - s_xx_.dot(sig_sol)
    B = np.kron(sig_sol.T, s_xx)

    def t_sq(x):
        gam = x[rs_idx(d - 1)]
        gam_til = A + B.dot(gam).reshape((d - 1, 1))
        return (n * gam_til.T.dot(linalg.solve(s_xx, gam_til)) / sig_hat_y)[0, 0]

    def t_sq_grad(x):
        gam = gam = x[rs_idx(d - 1)]
        gam_til = A + B.dot(gam).reshape((d - 1, 1))
        out = 2 * n * B.T.dot(linalg.solve(s_xx, gam_til)) / sig_hat_y
        return out.reshape((d - 1, d - 1)).T.reshape((-1,))

    def t_sq_hess(x):
        out = 2 * n * B.T.dot(linalg.solve(s_xx, B)) / sig_hat_y
        return out[rs_idx(d - 1), :][:, rs_idx(d - 1)]

    return t_sq, t_sq_grad, t_sq_hess


# Function returning objective function and gradient
def obj(theta_hash, t_sq, t_sq_grad, sig_x, gam_hat, n, gpr):
    x_hat = gam_hat.reshape((-1,))
    d = gam_hat.shape[0]

    def f(x):
        g = n * (x - x_hat).dot(sig_x).dot(x - x_hat)
        c, s = gpr.predict(x.reshape((1, -1)))
        c = c[0, 0]
        s = np.sqrt(s[0, 0])
        # return -(t_sq(x) - t_sq(theta_hash)) * stats.norm.cdf(-(g - c) / s)
        if stats.norm.cdf(-(g - c) / s) < 1e-4:
            return 0.0
        return (t_sq(x) - t_sq(theta_hash)) * stats.norm.cdf(-(g - c) / s)

    def grad_f(x):
        g = n * (x - x_hat).dot(sig_x).dot(x - x_hat)
        c, s = gpr.predict(x.reshape((1, -1)))
        c = c[0, 0]
        s = np.sqrt(s[0, 0])
        grad_g = 2 * n * (x - x_hat).dot(sig_x)
        grad_c, grad_s = gpr.predictive_gradients(x.reshape((1, -1)))
        grad_c, grad_s = grad_c.reshape((-1,)), grad_s.reshape((-1,))
        a_1 = -t_sq_grad(x) * stats.norm.cdf(-(g - c) / s)
        a_2 = -(t_sq(x) - t_sq(theta_hash)) * stats.norm.pdf(-(g - c) / s)
        a_3 = -((grad_g - grad_c) * s - (g - c) * grad_s) / (s**2)
        # return a_1 + a_2 * a_3
        return -(a_1 + a_2 * a_3)

    return f, grad_f


# Get unique (within atol) rows of a 2D np.array A
def unique_rows(A, atol=10e-5):
    remove = np.zeros(A.shape[0], dtype=bool)  # Row indexes to be removed.
    for i in range(A.shape[0]):  # Not very optimized, but simple.
        equals = np.all(np.isclose(A[i, :], A[(i + 1) :, :], atol=atol), axis=1)
        remove[(i + 1) :] = np.logical_or(remove[(i + 1) :], equals)
    return np.logical_not(remove)


# EAM algo
def EAM(
    X,
    thr,
    alpha,
    k_init=10,
    eps=0.1,
    h_rate=2,
    obj_tol=1e-3,
    opt_tol=1e-3,
    viol_tol=1e-2,
    max_iter=100,
    min_iter=5,
    early_stop=5,
    mid_stop=10,
    verbose=True,
):
    # Initialize algo
    t_sq, t_sq_grad, t_sq_hess = t_sq_bon(X)
    y, X_ = X[:, 0], X[:, 1:]
    n, d = X_.shape
    gam_hat, sig_hat = ls_est(X_)
    x = X_[: (n - 1), :]
    s_xx = x.T.dot(x) / n
    sig_x = np.kron(s_xx, linalg.inv(sig_hat))
    eval_c = c_function(n, d, sig_hat, alpha, 1000)

    # Quick check to see if t_sq(gam_hat) is below threshold
    gam_hat_vec = gam_hat.reshape((-1,))
    if (t_sq(gam_hat_vec) <= thr) & (eval_g(X_, gam_hat_vec) <= eval_c(gam_hat_vec)):
        if verbose:
            print("Terminated early. Feasible point lower than threshold found.")
        return_dict = {
            "t_sq_min": t_sq(gam_hat.reshape((-1,))),
            "theta_min": gam_hat.reshape((-1,)),
            "thr": thr,
            "viol": np.NaN,
            "status": "threshold",
            "num_iter": 0,
        }
        return return_dict

    k = k_init * (d**2)
    theta_k = sample_theta(k, d)
    theta_k_close = sample_theta_close(d**2, gam_hat, 1e-4)
    theta_k = np.vstack([theta_k, theta_k_close, gam_hat.reshape((-1,))])
    c_k = np.array([eval_c(theta) for theta in theta_k])
    g_k = np.array([eval_g(X_, theta) for theta in theta_k])
    feas_idx = g_k <= c_k
    theta_feas = theta_k[feas_idx, :]
    t_feas = [t_sq(theta) for theta in theta_feas]
    g_feas = g_k[feas_idx]
    c_feas = c_k[feas_idx]
    theta_max = theta_feas[np.argmin(t_feas)]
    theta_max_old = theta_max
    num_iter = 0
    h_counter = 0
    num_feas = 0
    viol = np.abs((g_feas - c_feas)[np.argmin(t_feas)])
    flag_while = True
    status = "converged"

    if verbose:
        print(
            "Iteration     |     Opt Proj     | Change in EI proj        |     Change     |     Max Violation   | Feasible points    |  Multi. Start Num.  | Percent conv. EI>0  |  Contraction counter"
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ "
        )

    while flag_while:
        ub = 0.1 * (1 / (d**h_counter))

        # A-step
        unique_idx = unique_rows(theta_k, 1e-7)  # Make sure that no two theta are close
        theta_mod, c_mod = theta_k[unique_idx, :], c_k[unique_idx, np.newaxis]
        gpr = fit_gp(theta_mod, c_mod)

        # M-step
        f, f_grad = obj(theta_max, t_sq, t_sq_grad, sig_x, gam_hat, n, gpr)
        r_max = t_sq_grad(theta_max)
        r_max /= np.abs(r_max).max()
        theta_init = draw_initial_theta(theta_max, f, r_max)
        theta_init = np.vstack(
            [
                theta_init,
                theta_max,
                theta_max + np.random.choice([1, -1], d**2) * 1e-4,
            ]
        )
        init_size = theta_init.shape[0]
        lin_const = optimize.LinearConstraint(
            np.eye(d**2), theta_max - ub, theta_max + ub, keep_feasible=False
        )
        cons_hess = lambda x, v: v[0] * t_sq_hess(x)
        nonlin_const = optimize.NonlinearConstraint(
            t_sq, -np.inf, t_sq(theta_max), jac=t_sq_grad, hess=cons_hess
        )
        const = [lin_const, nonlin_const]
        theta_res = []
        obj_res = []
        for theta in theta_init:
            res = optimize.minimize(
                f,
                theta,
                method="trust-constr",
                constraints=const,
                jac=f_grad,
                options={"verbose": 0},
            )
            theta_res.append(res.x)
            obj_res.append(res.fun)
        EI_pos_per = len(np.array(obj_res) < 0) / theta_init.shape[0]
        t_sq_res = np.array([t_sq(theta) for theta in theta_res])
        fin_idx = t_sq_res < 1e5
        theta_res_fin = np.array(theta_res)[fin_idx, :]
        theta_new = theta_res_fin[
            np.argmin(np.array(obj_res)[fin_idx]), :
        ]  # Pick best optimum
        if t_sq(theta_new) > 1e5:
            print(theta_new.reshape((d, d)))
            print(t_sq(theta_new))
        EI = -f(theta_new)
        bound_diff_u = np.abs(theta_new - (theta_max + ub)).min()
        bound_diff_l = np.abs(theta_new - (theta_max - ub)).min()

        # E-step (+ update)
        if EI > 1e-8:
            theta_k = np.vstack([theta_k, theta_new])
            c_new = eval_c(theta_new)
            c_k = np.append(c_k, c_new)
            g_new = eval_g(X_, theta_new)
            if (g_new <= c_new) & (t_sq(theta_new) - t_sq(theta_max) < 0):
                theta_max_old = theta_max
                theta_max = theta_new
                viol = np.abs(g_new - c_new)
                num_feas += 1
            else:
                theta_max_old = theta_max
        else:
            theta_max_old = theta_max

        # Add random theta with prob. eps
        if np.random.uniform() <= eps:
            theta_rand = sample_theta(1, d)[0, :]
            theta_k = np.vstack([theta_k, theta_rand])
            c_rand = eval_c(theta_rand)
            c_k = np.append(c_k, c_rand)
            g_rand = eval_g(X_, theta_rand)
            if (g_rand <= c_rand) & (t_sq(theta_rand) - t_sq(theta_max) < 0):
                theta_max_old = theta_max
                theta_max = theta_rand
                viol = np.abs(g_rand - c_rand)
                num_feas += 1

        # Check convergence criteria
        obj_diff = np.abs(t_sq(theta_new) - t_sq(theta_max))
        opt_diff = np.abs(t_sq(theta_max) - t_sq(theta_max_old))
        bound_diff = min(bound_diff_u, bound_diff_l)
        flag_while = np.logical_not(
            np.all(
                [
                    (num_iter >= min_iter),
                    (obj_diff < obj_tol),
                    (opt_diff < opt_tol),
                    (num_feas >= 1),
                    (bound_diff > 1e-4 / h_rate**h_counter),
                    (viol < viol_tol),
                ]
            )
        )
        out = (
            num_iter,
            t_sq(theta_max),
            obj_diff,
            opt_diff,
            viol,
            num_feas,
            init_size,
            EI_pos_per,
            h_counter,
        )

        # Check if statistic is below threshold
        if t_sq(theta_max) <= thr:
            if verbose:
                print("Terminated early. Feasible point lower than threshold found.")
            status = "threshold"
            flag_while = False

        # Check if early stop criterion is reached
        if (t_sq(theta_max) - thr > 100) & (num_iter > early_stop):
            status = "early_stop"
            flag_while = False

        # Check if mid stop criterion is reached
        if (t_sq(theta_max) - thr > 10) & (num_iter > mid_stop):
            status = "eary_stop"
            flag_while = False

        # Check if max_iter reached
        if num_iter + 1 == max_iter:
            status = "max_iter"
            flag_while = False

        # Update rate counter
        if np.abs(t_sq(theta_max) - t_sq(theta_max_old)) < 1e-3 / (h_rate**h_counter):
            h_counter += 1
        elif bound_diff < 1e-4:
            h_counter += -1

        if verbose:
            print(
                "{:9d}     |   {:9.4f}      | {:9.4f}                |  {:9.4f}     |    {:9.4f}        | {:9d}          | {:9d}           | {:9.4f}           | {:9d}            ".format(
                    *out
                )
            )
            print("")
        num_iter += 1

    return_dict = {
        "t_sq_min": t_sq(theta_max),
        "theta_min": theta_max,
        "thr": thr,
        "viol": viol,
        "status": status,
        "num_iter": num_iter,
    }
    return return_dict


# Function for computing the Bonferroni test based on
# Gaussian approx. CRs.
def test_gauss_bonf(
    X,
    alpha_1,
    alpha_2,
    k=10,
    num_restart=1,
    eps=0.1,
    h_rate=2,
    obj_tol=1e-3,
    opt_tol=1e-3,
    viol_tol=1e-2,
    max_iter=100,
    min_iter=5,
    verbose=True,
):
    for i in range(num_restart):
        n, d = X.shape
        thr = stats.chi2.ppf(1 - alpha_1, d - 1)
        eam_res = EAM(
            X,
            thr,
            alpha_2,
            k,
            eps,
            h_rate,
            obj_tol,
            opt_tol,
            viol_tol,
            max_iter,
            min_iter,
            verbose,
        )
        if eam_res["status"] in [
            "converged",
            "threshold",
            "early_stop",
        ]:  # (eam_res["status"] == "converged") | (eam_res["status"] == "threshold") | :
            break
        if verbose:
            print("Algorithm did not converge - status: {}".format(eam_res["status"]))
            print("Trying again...")
    return eam_res


# IVX estimator
def ivx_est(X, beta=0.95):
    n, d = X.shape
    y = X[1:, :]
    x = X[: (n - 1), :]
    # Construct instrument
    z = np.zeros((n - 1, d))
    z[0, :] = np.zeros(d)
    for t in range(1, n - 1):
        z[t, :] = (1 - n ** (-beta)) * z[t - 1, :] + x[t, :] - x[t - 1, :]
    s_yz = y.T.dot(z) / n
    s_xz = x.T.dot(z) / n
    return linalg.solve(s_xz.T, s_yz.T).T


# IVX t^2-statistic
def ivx_stat(X, beta=0.95):
    n, d = X.shape
    y = X[1:, :]
    x = X[: (n - 1), :]
    # Construct instrument
    delx = y - x
    z = np.zeros((n - 1, d))
    z[0, :] = np.zeros(d)
    for t in range(1, n - 1):
        z[t, :] = (1 - n ** (-beta)) * z[t - 1, :] + delx[t - 1, :]
    # Compute sample covariances
    s_yz = y.T.dot(z) / n
    s_xz = x.T.dot(z) / n
    s_zz = z.T.dot(z) / n
    # s_xzx = s_xz.dot(linalg.inv(s_zz)).dot(s_xz.T)
    s_xzx = linalg.solve(s_zz.T, s_xz.T).T.dot(s_xz.T)
    # Compute estimator and t^2-statistic
    gam_hat = linalg.solve(s_xz.T, s_yz.T).T
    gam_hat = gam_hat.T.reshape((-1,))
    _, sig_hat = ls_est(X)
    sig_xzx = np.kron(s_xzx, linalg.inv(sig_hat))

    def ivx_t(x):
        return n * (x - gam_hat).dot(sig_xzx).dot((x - gam_hat).T)

    def ivx_t_grad(x):
        return 2 * n * sig_xzx.dot((x - gam_hat).T)

    def ivx_t_hess(x):
        return 2 * n * sig_xzx

    return ivx_t, ivx_t_grad, ivx_t_hess


# Function for computing the Bonferroni test based on
# IVX CRs.
def test_ivx_bonf(X, alpha_1, alpha_2, beta=0.9):
    t_sq, t_sq_grad, t_sq_hess = t_sq_bon(X)
    y, X_ = X[:, 0], X[:, 1:]
    n, d = X_.shape
    thr = stats.chi2.ppf(1 - alpha_1, d)
    c = stats.chi2.ppf(alpha_2, df=d**2)
    gam_init = ivx_est(X_, beta)
    gam_init = gam_init.reshape((-1,))
    # Constraint
    ivx_t, ivx_t_grad, ivx_t_hess = ivx_stat(X_, beta)
    ivx_t_hess_v = lambda x, v: v[0] * ivx_t_hess(x)
    nl_const = optimize.NonlinearConstraint(
        ivx_t, -np.inf, c, jac=ivx_t_grad, hess=ivx_t_hess_v
    )
    res = optimize.minimize(
        t_sq,
        gam_init,
        method="trust-constr",
        jac=t_sq_grad,
        hess=t_sq_hess,
        constraints=[nl_const],
    )
    theta_min = res.x
    return {
        "t_sq_min": t_sq(theta_min),
        "theta_min": theta_min,
        "thr": thr,
        "reject": t_sq(theta_min) > thr,
    }
