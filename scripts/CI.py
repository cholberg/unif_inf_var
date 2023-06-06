import numpy as np
from scipy import stats
from scipy import linalg
from scipy import optimize
import GPy


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
def draw_initial_theta(theta_hash, p, f, sign, r_max, r_min=1e-10, num_points=5):
    r_current = r_max
    points = 0
    p_hash = p.dot(theta_hash)
    d_sq = theta_hash.shape[0]
    theta_out = np.zeros((num_points, d_sq))
    while (r_current > r_min) & (points < num_points):
        for i in range(2):
            if sign > 0:
                u_1 = np.random.uniform(0, r_current)
            else:
                u_1 = np.random.uniform(-r_current, 0)
            u_d = np.random.uniform(-r_current, r_current, size=d_sq - 1)
            u = np.append(u_1, u_d)
            theta_draw = theta_hash + u
            if -f(theta_draw) > 1e-10:
                theta_out[points, :] = theta_draw
                points += 1
                if points >= num_points:
                    break
        r_current /= 2
    theta_rand = sample_theta(2, int(np.sqrt(d_sq)))
    theta_out = np.vstack([theta_out, theta_rand])
    return theta_out[np.abs(theta_out).max(axis=1) > 0, :]


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
def c_function(n, d, sigma, nsim=1000):
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
        return np.quantile(ts, 0.95)

    def f_2(gam):
        gam_mat = gam.reshape((d, d))
        ts = np.zeros((nsim,))
        for i in range(nsim):
            Z = np.zeros((n - 1, d))
            for t in range(1, n - 1):
                Z[t, :] = Z[t - 1, :].dot(gam_mat) + eps[i, t - 1, :]
            ts[i] = cov_map(eps_up[i, :, :], Z)
        return np.quantile(ts, 0.95)

    return f_2


# Function for finding least squares estimators
def ls_est(X):
    n, _ = X.shape
    y = X[1:, :]
    x = X[: (n - 1), :]
    gam_hat, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    eps_hat = y - x.dot(gam_hat)
    sig_hat = eps_hat.T.dot(eps_hat) / n
    return gam_hat, sig_hat


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


# Function returning objective function and gradient
def obj(theta_hash, p, sig_x, gam_hat, n, gpr, sign):
    x_hat = gam_hat.reshape((-1,))
    q = sign * p

    def f(x):
        g = n * (x - x_hat).dot(sig_x).dot(x - x_hat)
        c, s = gpr.predict(x.reshape((1, -1)))
        c = c[0, 0]
        s = np.sqrt(s[0, 0])
        if stats.norm.cdf(-(g - c) / s) < 1e-4:
            return 0.0
        return -q.dot(x - theta_hash) * stats.norm.cdf(-(g - c) / s)

    def grad_f(x):
        g = n * (x - x_hat).dot(sig_x).dot(x - x_hat)
        c, s = gpr.predict(x.reshape((1, -1)))
        c = c[0, 0]
        s = np.sqrt(s[0, 0])
        grad_g = 2 * n * (x - x_hat).dot(sig_x)
        grad_c, grad_s = gpr.predictive_gradients(x.reshape((1, -1)))
        grad_c, grad_s = grad_c.reshape((-1,)), grad_s.reshape((-1,))
        a_1 = -q.dot(stats.norm.cdf(-(g - c) / s))
        a_2 = -q.dot(x - theta_hash) * stats.norm.pdf(-(g - c) / s)
        a_3 = -((grad_g - grad_c) * s - (g - c) * grad_s) / (s**2)
        return a_1 + a_2 * a_3

    return f, grad_f


# Get unique (within atol) rows of a 2D np.array A
def unique_rows(A, atol=10e-5):
    remove = np.zeros(A.shape[0], dtype=bool)  # Row indexes to be removed.
    for i in range(A.shape[0]):  # Not very optimized, but simple.
        equals = np.all(np.isclose(A[i, :], A[(i + 1) :, :], atol=atol), axis=1)
        remove[(i + 1) :] = np.logical_or(remove[(i + 1) :], equals)
    return np.logical_not(remove)


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
def ivx_t(X, gam, beta=0.95):
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
    # Compute estimator and t^2-statistic
    gam_hat = linalg.solve(s_xz.T, s_yz.T).T
    _, sig_hat = ls_est(X)
    sig_hat_sq = linalg.sqrtm(sig_hat)
    t_sq = s_xz.dot(linalg.inv(s_zz)).dot(s_xz.T)
    t_sq = (gam_hat - gam).dot(t_sq).dot((gam_hat - gam).T)
    t_sq = np.linalg.solve(sig_hat_sq, np.linalg.solve(sig_hat_sq, t_sq).T).T
    return n * np.trace(t_sq)


# Left step for bisection algo
def ci_step_neg(stat, thr, start, step_length, gam_init):
    d = gam_init.shape[0]
    gam_init_vec = gam_init.reshape((-1,))[1:]
    inside = True
    current_pos = start
    A = np.array([1] + [0] * (d**2 - 1))
    while inside:
        current_pos -= step_length
        obj = lambda x: stat(np.hstack([current_pos, x]).reshape((d, -1)))
        res = optimize.minimize(obj, gam_init_vec)
        min_gam = np.hstack([current_pos, res.x]).reshape((d, -1))
        min_stat = stat(min_gam)
        inside = min_stat <= thr
    return current_pos + step_length


# Right step for bisection algo
def ci_step_pos(stat, thr, start, step_length, gam_init):
    d = gam_init.shape[0]
    gam_init_vec = gam_init.reshape((-1,))[1:]
    inside = True
    current_pos = start
    A = np.array([1] + [0] * (d**2 - 1))
    while inside:
        current_pos += step_length
        obj = lambda x: stat(np.hstack([current_pos, x]).reshape((d, -1)))
        res = optimize.minimize(obj, gam_init_vec)
        min_gam = np.hstack([current_pos, res.x]).reshape((d, -1))
        min_stat = stat(min_gam)
        inside = min_stat <= thr
    return current_pos - step_length


# Bisection algo
def ci_bisection(stat, thr, gam_init, init_length=0.25, iter=5):
    # Find upper end point
    start_u = gam_init[0, 0].copy()
    start_l = gam_init[0, 0].copy()
    step_length = init_length
    for i in range(iter):
        start_l = ci_step_neg(stat, thr, start_l, step_length, gam_init)
        start_u = ci_step_pos(stat, thr, start_u, step_length, gam_init)
        step_length = step_length / 4
    return [start_l - step_length, start_u + step_length]


# EAM algo
def EAM(
    X,
    p=None,
    k_init=10,
    eps=0.1,
    lb=-0.95,
    ub=1.2,
    h_rate=2,
    obj_tol=1e-3,
    opt_tol=1e-3,
    viol_tol=1e-2,
    max_iter=100,
    min_iter=5,
    verbose=True,
):
    # Initialize algo
    n, d = X.shape
    gam_hat, sig_hat = ls_est(X)
    x = X[: (n - 1), :]
    s_xx = x.T.dot(x) / n
    sig_x = np.kron(s_xx, linalg.inv(sig_hat))
    eval_c = c_function(n, d, sig_hat, 1000)
    if p is None:
        p = np.array([1] + [0] * (d**2 - 1))
    k = k_init * (d**2)
    theta_k = sample_theta(k, d)
    theta_k_close = sample_theta_close(d**2, gam_hat)
    theta_k = np.vstack([theta_k, theta_k_close, gam_hat.reshape((-1,))])
    theta_k_u = theta_k.copy()
    theta_k_l = theta_k.copy()
    c_k = np.array([eval_c(theta) for theta in theta_k])
    c_k_u = c_k.copy()
    c_k_l = c_k.copy()
    g_k = np.array([eval_g(X, theta) for theta in theta_k])
    feas_idx = g_k <= c_k
    theta_feas = theta_k[feas_idx, :]
    p_feas = [p.dot(theta) for theta in theta_feas]
    g_feas = g_k[feas_idx]
    c_feas = c_k[feas_idx]
    theta_min = theta_feas[np.argmin(p_feas)]
    theta_min_old = theta_min
    theta_max = theta_feas[np.argmax(p_feas)]
    theta_max_old = theta_max
    num_iter = 0
    h_counter_u = 0
    h_counter_l = 0
    num_feas_u = 0
    num_feas_l = 0
    viol_u = np.abs((g_feas - c_feas)[np.argmax(p_feas)])
    viol_l = np.abs((g_feas - c_feas)[np.argmin(p_feas)])
    flag_while_u = True
    flag_while_l = True
    status_u = "converged"
    status_l = "converged"

    if verbose:
        print(
            "Iteration     |     Opt Proj     | Change in EI proj        |     Change     |     Max Violation   | Feasible points    |  Multi. Start Num.  | Percent conv. EI>0  |  Contraction counter"
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ "
        )

    while (flag_while_u | flag_while_l) & (num_iter < max_iter):
        # ub_h = p.dot(theta_max) + (ub - p.dot(theta_max)) / (h_rate**h_counter_u)
        ub_h = p.dot(theta_max) + 0.2 / (h_rate**h_counter_u)
        # lb_h = p.dot(theta_min) - (p.dot(theta_min) - lb) / (h_rate**h_counter_l)
        lb_h = p.dot(theta_min) - 0.2 / (h_rate**h_counter_u)

        # A-step
        unique_idx = unique_rows(theta_k, 1e-7)  # Make sure that no two theta are close
        theta_mod, c_mod = theta_k[unique_idx, :], c_k[unique_idx, np.newaxis]
        gpr = fit_gp(theta_mod, c_mod)

        # Upper
        if flag_while_u:
            # M-step
            r_max = np.abs(p.dot(theta_max) - ub_h)
            f, f_grad = obj(theta_max, p, sig_x, gam_hat, n, gpr, 1.0)
            # Draw initilizations for multi start
            theta_init = draw_initial_theta(theta_max, p, f, 1.0, r_max)
            theta_init = np.vstack([theta_init, theta_max, theta_max + p * 1e-4])
            init_size_u = theta_init.shape[0]
            const = optimize.LinearConstraint(
                p, p.dot(theta_max), ub_h, keep_feasible=False
            )
            theta_res = []
            obj_res = []
            for theta in theta_init:
                res = optimize.minimize(f, theta, constraints=const, jac=f_grad)
                theta_res.append(res.x)
                obj_res.append(res.fun)
            EI_pos_per = len(np.array(obj_res) < 0) / theta_init.shape[0]
            theta_new_u = theta_res[np.argmin(obj_res)]  # Pick best optimum'
            EI_u = -f(theta_new_u)

            # E-step (+ update)
            if EI_u > 1e-5:
                theta_k = np.vstack([theta_k, theta_new_u])
                theta_k_u = np.vstack([theta_k_u, theta_new_u])
                c_new_u = eval_c(theta_new_u)
                c_k = np.append(c_k, c_new_u)
                c_k_u = np.append(c_k_u, c_new_u)
                g_new_u = eval_g(X, theta_new_u)
                if (g_new_u <= c_new_u) & (p.dot(theta_new_u - theta_max) > 0):
                    theta_max_old = theta_max
                    theta_max = theta_new_u
                    viol_u = np.abs(g_new_u - c_new_u)
                    num_feas_u += 1
                else:
                    theta_max_old = theta_max
            else:
                theta_max_old = theta_max

            # Add random theta with prob. eps
            if np.random.uniform() <= eps:
                theta_rand_u = sample_theta(1, d)[0, :]
                theta_k = np.vstack([theta_k, theta_rand_u])
                theta_k_u = np.vstack([theta_k_u, theta_rand_u])
                c_rand_u = eval_c(theta_rand_u)
                c_k = np.append(c_k, c_rand_u)
                c_k_u = np.append(c_k_u, c_rand_u)
                g_rand_u = eval_g(X, theta_rand_u)
                if (g_rand_u <= c_rand_u) & (p.dot(theta_rand_u - theta_max) > 0):
                    theta_max_old = theta_max
                    theta_max = theta_rand_u
                    viol_u = np.abs(g_rand_u - c_rand_u)
                    num_feas_u += 1

            # Check convergence criteria
            obj_diff_u = np.abs(p.dot(theta_new_u - theta_max_old))
            opt_diff_u = np.abs(p.dot(theta_max - theta_max_old))
            bound_diff_u = np.abs(p.dot(theta_max) - ub_h)
            flag_while_u = np.logical_not(
                np.all(
                    [
                        (num_iter >= min_iter),
                        (obj_diff_u < obj_tol),
                        (opt_diff_u < opt_tol),
                        (num_feas_u >= 1),
                        (bound_diff_u > 1e-4 / h_rate**h_counter_u),
                        (viol_u < viol_tol),
                    ]
                )
            )
            out_u = (
                num_iter,
                p.dot(theta_max),
                obj_diff_u,
                opt_diff_u,
                viol_u,
                num_feas_u,
                init_size_u,
                EI_pos_per,
                h_counter_u,
            )

            # Check if the new optimums are too close to the boundaries
            """if np.abs(p.dot(theta_max) - ub) < 1e-4:
                status_u = "boundary"
                flag_while_u = False"""

            # Check if max_iter reached
            if num_iter + 1 == max_iter:
                status_u = "max_iter"

            # Update rate counter
            if np.abs(p.dot(theta_max - theta_max_old)) < 1e-3 / (
                h_rate**h_counter_u
            ):
                h_counter_u += 1
            elif np.abs(p.dot(theta_max) - ub_h) < 1e-4:
                h_counter_u += -1
            # if (h_counter_u > 0) & (np.abs(p.dot(theta_max) - ub_h) < 1e-4):
            #    h_counter_u += -1

        # Lower
        if flag_while_l:
            # M-step
            r_max = np.abs(p.dot(theta_min) - lb_h)
            f, f_grad = obj(theta_min, p, sig_x, gam_hat, n, gpr, -1.0)
            # Draw initilizations for multi start
            theta_init = draw_initial_theta(theta_min, p, f, -1.0, r_max)
            theta_init = np.vstack([theta_init, theta_min, theta_min - p * 1e-4])
            init_size_l = theta_init.shape[0]
            const = optimize.LinearConstraint(
                p, lb_h, p.dot(theta_min), keep_feasible=False
            )
            theta_res = []
            obj_res = []
            for theta in theta_init:
                res = optimize.minimize(f, theta, constraints=const, jac=f_grad)
                theta_res.append(res.x)
                obj_res.append(res.fun)
            EI_pos_per = len(np.array(obj_res) < 0) / theta_init.shape[0]
            theta_new_l = theta_res[np.argmin(obj_res)]  # Pick best optimum
            EI_l = -f(theta_new_l)

            # E-step (+ update)
            if EI_l > 1e-5:
                theta_k = np.vstack([theta_k, theta_new_l])
                theta_k_l = np.vstack([theta_k_l, theta_new_l])
                c_new_l = eval_c(theta_new_l)
                c_k = np.append(c_k, c_new_l)
                c_k_l = np.append(c_k_l, c_new_l)
                g_new_l = eval_g(X, theta_new_l)
                if (g_new_l <= c_new_l) & (p.dot(theta_new_l - theta_min) < 0):
                    theta_min_old = theta_min
                    theta_min = theta_new_l
                    viol_l = np.abs(g_new_l - c_new_l)
                    num_feas_l += 1
                else:
                    theta_min_old = theta_min
            else:
                theta_min_old = theta_min

            # Add random theta with prob. eps
            if np.random.uniform() <= eps:
                theta_rand_l = sample_theta(2, d)[0, :]
                theta_k = np.vstack([theta_k, theta_rand_l])
                theta_k_l = np.vstack([theta_k_l, theta_rand_l])
                c_rand_l = eval_c(theta_rand_l)
                c_k = np.append(c_k, c_rand_l)
                c_k_l = np.append(c_k_l, c_rand_l)
                g_rand_l = eval_g(X, theta_rand_l)
                if (g_rand_l <= c_rand_l) & (p.dot(theta_rand_l - theta_min) < 0):
                    theta_min_old = theta_min
                    theta_min = theta_rand_l
                    viol_l = np.abs(g_rand_l - c_rand_l)
                    num_feas_l += 1

            # Check convergence criteria
            obj_diff_l = np.abs(p.dot(theta_new_l - theta_min_old))
            opt_diff_l = np.abs(p.dot(theta_min - theta_min_old))
            bound_diff_l = np.abs(p.dot(theta_min) - lb_h)
            flag_while_l = np.logical_not(
                np.all(
                    [
                        (num_iter >= min_iter),
                        (obj_diff_l < obj_tol),
                        (opt_diff_l < opt_tol),
                        (num_feas_l >= 1),
                        (bound_diff_l > 1e-4 / h_rate**h_counter_l),
                        (viol_l < viol_tol),
                    ]
                )
            )
            out_l = (
                num_iter,
                p.dot(theta_min),
                obj_diff_l,
                opt_diff_l,
                viol_l,
                num_feas_l,
                init_size_l,
                EI_pos_per,
                h_counter_l,
            )

            # Check if the new optimums are too close to the boundaries
            """if np.abs(p.dot(theta_min) - lb) < 1e-4:
                status_l = "boundary"
                flag_while_l = False"""

            # Check if max_iter reached
            if num_iter + 1 == max_iter:
                status_l = "max_iter"

            # Update rate counter
            if np.abs(p.dot(theta_min - theta_min_old)) < 1e-3 / (
                h_rate**h_counter_l
            ):
                h_counter_l += 1
            elif np.abs(p.dot(theta_min) - lb_h) < 1e-4:
                h_counter_l += -1
            # if (h_counter_l > 0) & (np.abs(p.dot(theta_min) - lb_h) < 1e-4):
            #    h_counter_l += -1

        if verbose:
            print(
                "{:9d}     |   {:9.4f}      | {:9.4f}                |  {:9.4f}     |    {:9.4f}        | {:9d}          | {:9d}           | {:9.4f}           | {:9d}            ".format(
                    *out_l
                )
            )
            print(
                "{:9d}     |   {:9.4f}      | {:9.4f}                |  {:9.4f}     |    {:9.4f}        | {:9d}          | {:9d}           | {:9.4f}           | {:9d}            ".format(
                    *out_u
                )
            )
            print("")
        num_iter += 1

    return_dict = {
        "ci_l": p.dot(theta_min),
        "ci_u": p.dot(theta_max),
        "theta_min": theta_min,
        "theta_max": theta_max,
        "viol_l": viol_l,
        "viol_u": viol_u,
        "status_l": status_l,
        "status_u": status_u,
    }
    return return_dict


# Function for computing lag-augmented CIs
def ci_la(X, p=None):
    n, d = X.shape
    if p is None:
        p = np.array([1] + [0] * (d**2 - 1))
    y = X[2:, :]
    x = np.hstack([X[1 : (n - 1), :], X[: (n - 2), :]])
    gam_hat, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    _, sig_hat = ls_est(X)
    sig_hat_inv = np.linalg.inv(sig_hat)
    sig_hat_11 = sig_hat_inv[0, 0] * sig_hat[0, 0]
    z = stats.norm.ppf(0.975)
    ci_l = gam_hat[0, 0] - z * sig_hat_11 / np.sqrt(n)
    ci_u = gam_hat[0, 0] + z * sig_hat_11 / np.sqrt(n)
    return [ci_l, ci_u]


# Function for copmuting IVX projected CIs
def ci_ivx_proj(X, p=None, init_length=0.25, iter=5, beta=0.9):
    n, d = X.shape
    if p is None:
        p = np.array([1] + [0] * (d**2 - 1))
    c = stats.chi2.ppf(0.95, df=d**2)
    stat = lambda x: ivx_t(X, x, beta)
    gam_init = ivx_est(X, beta)
    return ci_bisection(stat, c, gam_init, init_length, iter)


# Function for computing Projected CIs using Gaussian approx.
# Basically just a wrapper around EAM to allow for multiple restarts
# if the algo does not converge. (This can happen since c is computed
# from random samples and not exact.)
def ci_gauss_proj(
    X,
    p=None,
    k_init=10,
    eps=0.1,
    lb=-0.95,
    ub=1.2,
    h_rate=2,
    obj_tol=1e-3,
    opt_tol=1e-3,
    viol_tol=1e-2,
    max_iter=100,
    min_iter=5,
    num_restart=3,
    verbose=True,
):
    for i in range(num_restart):
        eam_res = EAM(
            X,
            p,
            k_init,
            eps,
            lb,
            ub,
            h_rate,
            obj_tol,
            opt_tol,
            viol_tol,
            max_iter,
            min_iter,
            verbose,
        )
        if (eam_res["status_u"] == "converged") & (eam_res["status_l"] == "converged"):
            break
        if verbose:
            print(
                "Algorithm did not converge - status_u: {}, status_l: {}".format(
                    eam_res["status_u"], eam_res["status_l"]
                )
            )
            print("Trying again...")
    return eam_res
