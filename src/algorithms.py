import numpy as np
from src.base import ProblemAbstract, OptimizationResult
import osqp
import scipy.sparse as sp
from src.colors_setting import eth_purple, eth_bronze, dark_blue, dark_orange

# ====================
# step-size schedulers
# ====================


# Cor. 3.3 "First-Order Methods for Nonsmooth Nonconvex Functional Constrained Optimization"
# Jia + Grimmer, 2024
def step_size_gen_PPM(mu: float, L1: float) -> callable:
    def _alpha_t_PPM(t) -> float:
        return 2 / (mu * (t + 2) + (L1**2) / (mu * (t + 1)))

    return _alpha_t_PPM


# Prop. 5.5 in "Oracle Complexity of Single-Loop Switching Subgradient Methods"
# by Huang and Lin, 2023, NeurIPS 2023
def step_size_gen_WCWC(
    epsilon: float,
    rho: float,
    nu: float,
    L1: float,
) -> callable:
    # precompute this value once
    _step_size_infeas = nu / (4 * L1**2) * min(epsilon**2 / L1, nu / (4 * rho))

    def _alpha_t_WCWC(
        is_feasible: bool, F2_val: float, partial_F2: np.ndarray
    ) -> float:
        if is_feasible:
            return _step_size_infeas
        return F2_val / np.linalg.norm(partial_F2, ord=2) ** 2

    return _alpha_t_WCWC


def project_with_osqp(
    xk: np.ndarray,
    grad_list: list[np.ndarray],
    rhs_list: list[np.ndarray],
    F_list: list[float],
):
    n = xk.shape[0]
    q = len(grad_list)
    F_mat = np.vstack([g.reshape(1, -1) for g in grad_list])
    rhs_vec = np.array([rhs_list[i] - F_list[i] + grad_list[i] @ xk for i in range(q)])
    # we scale the object by 1/2 -> does not matter
    P = sp.eye(n, format="csc")
    q_vec = -xk.copy()
    A = sp.csc_matrix(F_mat)
    l = -np.inf * np.ones(q)
    u = rhs_vec
    prob = osqp.OSQP()
    prob.setup(P=P, q=q_vec, A=A, l=l, u=u, verbose=False)
    res = prob.solve()
    if res.info.status != "solved":
        raise RuntimeError(f"OSQP failed: {res.info.status}")
    # print("inner")
    # print(res.x)
    return res.x


# ====================
# Solution Algorithms
# ====================


# --- HC-PMM Algorithm ---
def HC_PMM_osqp(
    x0: np.ndarray,
    max_iter: int,
    problem: ProblemAbstract,
    _max_iter_global: int,
    eta: float = None,
) -> OptimizationResult:
    x = x0.copy()

    history = [x.copy()]
    iter_history = [1]

    min_score = None
    t_star_min_score = None

    # CASE 1: assume F_1^* is given
    if eta is None:
        fstar_known = True
        eta = problem.F1(problem.optimum_XX)
        get_next_eta = lambda x_next, prev_eta: eta
        alpha = (
            problem.epsilon
            * (problem.mu_c**2)
            / ((problem.rho + problem.L) * problem.D_U**2)
        )
        alpha = 0.005
        beta = 1
        lam = 0
    else:
        # CASE 2: F_1^* is unknown, we only have a lower bound eta
        # eta = problem.F1(problem.optimum_XX) - 0.001
        fstar_known = False
        _eta_zero = eta  # copy to insert into the label
        lam = problem.lambdaOpt
        beta = 1 / 2
        A = (problem.rho + problem.L) * problem.D_U**2 / (2 * problem.mu_c**2)
        alpha = min(
            problem.epsilon / (16 * A),
            problem.epsilon / (8 * lam * A),
            1
            / np.log(
                8 * (problem.F1(x0) - problem.F1(problem.optimum_XX)) / problem.epsilon
            ),
        )
        alpha = 0.0005
        get_next_eta = lambda x_next, prev_eta: (1 - beta) * (
            prev_eta
            + problem.F1(x_next)
            + lam * max(0, problem.F2(x_next))
            - (1 - 2 * beta) * prev_eta
        )

    tau = problem.rho * alpha**2 * problem.D_U**2 / (2 * problem.mu_c**2)

    for t in range(max_iter):
        f1 = problem.F1(x)
        f2 = problem.F2(x)
        g1 = problem.partial_F1(x)
        g2 = problem.partial_F2(x)

        rhs1 = (
            (1 - alpha * beta) * f1
            + alpha * beta * eta
            + (1 - beta) * alpha * lam * max(0, f2)
            + tau
        )
        rhs2 = (1 - alpha) * f2 + tau

        x = project_with_osqp(
            xk=x, grad_list=[g1, g2], rhs_list=[rhs1, rhs2], F_list=[f1, f2]
        )
        # either constant, if F_1^* is known, or linesearch method
        eta = get_next_eta(x, eta)

        # compute the score
        _new_score = problem.F1(x) + lam * max(0, problem.F2(x))
        if min_score is None or _new_score <= min_score:
            min_score = _new_score
            t_star_min_score = t

        history.append(x.copy())
        iter_history.append(t)

    # Return last iterate and best according to augmented objective

    if fstar_known:
        x_final = history[-1]
    else:
        # scores = [problem.F1(x) + lam * max(0, problem.F2(x)) for x in history]
        # t_star = np.argmin(scores)
        x_final = history[t_star_min_score]

    # append the last iterate to match the plot
    diff = _max_iter_global - max_iter
    iter_history.extend(list(np.arange(max_iter, _max_iter_global)))
    history.extend([x.copy() for _ in range(diff)])

    if fstar_known:
        return OptimizationResult(
            x_final=x_final,
            iter_history=np.array(iter_history),
            history=np.array(history),
            label="S-StarBL",
            marker="*",
            linestyle=(0, (3, 5, 1, 5, 1, 5)),
            color=eth_bronze,
            _skip_iterates=50,
        )
    else:
        return OptimizationResult(
            x_final=x_final,
            iter_history=np.array(iter_history),
            history=np.array(history),
            label="S-BL+AdaLS",
            # label=rf"Ada-LS+PMM, $\eta_0={{{_eta_zero}}}$",
            marker="x",
            linestyle="dashdot",
            color=dark_orange,
            _skip_iterates=50,
        )


# first method of the paper, here with switching sub-gradient as an inner solve
def IPPM_switching_subgradient(
    x0: np.ndarray,
    step_size: float,
    max_iter: int,
    iter_PPM: int,
    constraint_viol_tol: float,
    problem: ProblemAbstract,
    const_step_size: bool = False,
) -> OptimizationResult:
    x = x0.copy()
    history = [x0.copy()]
    iter_history = [1]
    x_reg = x0.copy()

    if const_step_size:
        alpha_t = lambda t: step_size
    else:
        alpha_t = step_size_gen_PPM(problem.mu, problem.G_subgradients)

    run_sum_iterate = np.zeros_like(x0)
    run_sum_coeff = 1

    for k in range(max_iter):
        if problem.F2(x) > constraint_viol_tol:
            g = problem.partial_F2(x)
        else:
            g = problem.partial_F1(x)
            run_sum_iterate += (k % iter_PPM + 1) * x.copy()
            run_sum_coeff += k % iter_PPM + 1

        g += problem.rho_hat * (x - x_reg)

        # Update step
        x -= alpha_t(k % iter_PPM) * g
        # x -= step_size * g

        # history.append(run_sum_iterate / run_sum_coeff)

        if k % iter_PPM == 0 and k > 0:
            # print("update reg: {}".format(k))
            # history.append(run_sum_iterate / run_sum_coeff)
            iter_history.append(k)
            history.append(run_sum_iterate / run_sum_coeff)

            x_reg = run_sum_iterate / run_sum_coeff
            # reg_hist = []
            run_sum_iterate = np.zeros_like(x0)
            run_sum_coeff = 1

    return OptimizationResult(
        x_final=x,
        iter_history=np.array(iter_history),
        history=np.array(history),
        label="IPPM+SwSG",
        marker="v",
        linestyle="--",
        color=eth_purple,
    )


def IPPM_ACGD(
    x0: np.ndarray,
    # step_size: float,
    max_iter: int,
    iter_PPM: int,
    # constraint_viol_tol: float,
    problem: ProblemAbstract,
    # const_step_size: bool = False,
) -> OptimizationResult:
    # Step size buffers
    tau_vals = []
    theta_vals = []
    eta_vals = []
    omega_vals = []

    # Precompute tau values
    for t in range(1, iter_PPM + 1):
        tau_t = (t - 1) / 2  # min((t - 1) / 2, problem.sqrt_kappa)
        tau_vals.append(tau_t)

    # Precompute eta values
    for t in range(1, iter_PPM + 1):
        tau_tp1 = tau_vals[t] if t < iter_PPM else tau_vals[-1]  # tau_{t+1}
        # TODO
        eta_t = problem.L_Lambda_r / tau_tp1
        eta_vals.append(eta_t)

    # Precompute theta values
    theta_vals.append(0.0)  # dummy for t=1
    for t in range(1, iter_PPM):
        theta_t = tau_vals[t] / (tau_vals[t - 1] + 1)
        theta_vals.append(theta_t)

    # Precompute omega values
    omega_vals.append(1.0)
    for t in range(1, iter_PPM):
        omega_t = omega_vals[t - 1] / theta_vals[t]
        omega_vals.append(omega_t)

    history = [x0.copy()]
    iter_history = [1]

    run_sum_iterate = np.zeros_like(x0)
    run_sum_coeff = 0

    x_reg = x0.copy()
    x_tm1 = x0.copy()
    x_tm2 = x0.copy()

    def shifted_constraint(zT, x_reg):
        return (
            problem.F2(zT)
            + problem.rho_hat / 2 * np.linalg.norm(zT - x_reg) ** 2
            - problem.shift
        )

    # running value -> add to a list
    for k in range(max_iter):

        tauT = tau_vals[k % iter_PPM]
        thetaT = theta_vals[k % iter_PPM]
        etaT = eta_vals[k % iter_PPM]
        omegaT = omega_vals[k % iter_PPM]

        # Momentum step
        x_tilde = x_tm1 + thetaT * (x_tm1 - x_tm2)
        x_bar = (tauT * x_tm1 + x_tilde) / (1 + tauT)

        # Gradient updates
        pi = problem.partial_F1(x_bar) + problem.rho_hat * (x_bar - x_reg)
        nu = problem.partial_F2(x_bar) + problem.rho_hat * (x_bar - x_reg)

        # Constraint value
        g_val = shifted_constraint(x_bar, x_reg)

        # Unconstrained minimizer
        x_unconstr = x_bar - (1.0 / etaT) * pi

        # Constraint: <nu, x - x_bar> + g(x_bar) <= 0
        d = np.dot(nu, x_unconstr - x_bar) + g_val
        # save if/else call -> optional proj via maximum
        x_next = x_unconstr - np.maximum(d, 0.0) / (np.dot(nu, nu) + 1e-10) * nu

        run_sum_iterate += omegaT * x_next
        run_sum_coeff += omegaT

        # Update iterates
        x_tm2 = x_tm1.copy()
        x_tm1 = x_next.copy()

        if k % iter_PPM == 0 and k > 0:
            # print("update reg: {}".format(k))
            # history.append(run_sum_iterate / run_sum_coeff)
            iter_history.append(k)
            history.append(run_sum_iterate / run_sum_coeff)

            x_reg = run_sum_iterate.copy() / run_sum_coeff
            # reg_hist = []

            # reset the running ergodic mean
            run_sum_iterate = np.zeros_like(x0)
            run_sum_coeff = 0

            # reset the momentum
            x_tm1 = x_reg.copy()
            x_tm2 = x_reg.copy()

    # TODO(Florian): different color + label depending on Slater
    return OptimizationResult(
        x_final=x_next,
        iter_history=np.array(iter_history),
        history=np.array(history),
        label="IPPM+ACGD",
        marker="v",
        linestyle="--",
        color=eth_purple,
    )


# penalty-based approach NON-SMOOTH case
# second method of the paper, using sub-gradient descent as an inner solver
def IPPPM_subgradient(
    x0: np.ndarray,
    step_size: float,
    max_iter: int,
    iter_PPM: int,
    problem: ProblemAbstract,
    const_step_size: bool = False,
):
    x = x0.copy()
    history = [x0.copy()]
    iter_history = [1]
    x_reg = x0.copy()
    if const_step_size:
        alpha_t = lambda t: step_size
    else:
        alpha_t = step_size_gen_PPM(problem.mu, problem.G_subgradients)

    run_sum_iterate = np.zeros_like(x0)
    run_sum_coeff = 1

    for k in range(max_iter):
        g = np.zeros_like(x0)
        if problem.F2(x) > 0:
            # if constraint violated -> add penalty sub-gradient
            # g += beta * F2(x) * partial_F2(x)  # QUADRATIC PENALTY
            g += problem.beta_PPPM * problem.partial_F2(x)  # EXACT PENALTY
            # print("F2(x): ", F2(x))
            # print("g F2: ", g)
        # sub-gradient of F1, regularization below
        g += problem.partial_F1(x)

        run_sum_iterate += (k % iter_PPM + 1) * x.copy()
        run_sum_coeff += k % iter_PPM + 1

        g += problem.rho_hat * (x - x_reg)

        # Update step
        x -= alpha_t(k % iter_PPM) * g
        # x -= step_size * g

        # history.append(run_sum_iterate / run_sum_coeff)

        if k % iter_PPM == 0 and k > 0:
            # print("update reg: {}".format(k))
            # history.append(run_sum_iterate / run_sum_coeff)
            iter_history.append(k)
            history.append(run_sum_iterate / run_sum_coeff)

            x_reg = run_sum_iterate / run_sum_coeff
            # reg_hist = []
            run_sum_iterate = np.zeros_like(x0)
            run_sum_coeff = 0

    return OptimizationResult(
        x_final=x,
        iter_history=np.array(iter_history),
        history=np.array(history),
        label="IPPPM+SG",
        marker="1",
        linestyle="dotted",
        color=dark_orange,
    )


# penalty-based approach in SMOOTH case
# second method of the paper, using the gradient method as an inner solver
def IPPPM_GD(
    x0: np.ndarray,
    problem: ProblemAbstract,
    max_iter: int,
    iter_PPM: int,
    const_step_size: bool = False,
) -> OptimizationResult:
    # momentum scheme
    x = x0.copy()
    # 1/L step size
    grad_stepSize = 1 / (
        problem.G_subgradients
        + problem.rho_hat
        + problem.beta_PPPM
        * problem.G_subgradients
        * (problem.D_X * problem.G_subgradients + problem.G_subgradients)
    )

    # save results
    history = [x0.copy()]
    iter_history = [1]
    x_reg = x0.copy()
    if const_step_size:
        alpha_t = lambda t: grad_stepSize
    else:
        alpha_t = step_size_gen_PPM(problem.mu, problem.G_subgradients)

    g = np.zeros_like(x0)
    for k in range(max_iter):

        # gradient calculation
        g = problem.partial_F1(x)
        g += problem.beta_PPPM * (max(problem.F2(x), 0)) * problem.partial_F2(x)
        g += problem.rho_hat * (x - x_reg)

        # Update step
        x -= alpha_t(k % iter_PPM) * g

        if k % iter_PPM == 0 and k > 0:
            iter_history.append(k + 1)
            history.append(x)
            x_reg = x.copy()

    return OptimizationResult(
        x_final=x,
        iter_history=np.array(iter_history),
        history=np.array(history),
        label="IPPPM+DG",
        marker="1",
        linestyle="dotted",
        color=dark_orange,
    )


def ReLU(x):
    return x * (x > 0)


# implementation of the IPPM framework with the Constraint Extrapolation
# (ConEx) method by Boob, Deng, Lan; Mathematical Programming; 2023
def IPPM_ConEx(
    x0: np.ndarray,
    # step_size: float,
    max_iter: int,
    iter_PPM: int,
    problem: ProblemAbstract,
    # TODO: fix constants
    y0: float = 0,
    B: float = 1,
    const_step_size: bool = False,
):
    history = [x0.copy()]
    iter_history = [1]

    # notation of Boob et al.
    alpha0 = problem.rho_hat
    L0_ConEx = problem.rho_hat
    L1_ConEx = problem.rho_hat

    t0 = 4 * (L0_ConEx + B * L1_ConEx) / alpha0 + 2
    caligraphic_M = 2 * problem.G_subgradients

    # internal step-size schedules
    def theta_k(k):
        return (k + t0 + 1) / (k + t0 + 2)

    def eta_k(k):
        return 2 / (alpha0 * (k + t0 + 1))

    def tau_k(k):
        return (k + 1) * (alpha0) / (32 * caligraphic_M**2)

    def gamma_k(k):
        return k + t0 + 2

    # if const_step_size:
    #     alpha_t = lambda t: step_size
    # else:
    #     alpha_t = step_size_gen_PPM(mu, G_subgradients)

    run_sum_iterate = np.zeros_like(x0)
    run_sum_coeff = 0

    x_reg = x0.copy()
    x_prev = x0.copy()
    x_curr = x0.copy()

    # no copy since it is a float
    y_k = y0

    def conEx_step(x_curr, x_prev, x_reg, partial_F2_xPrev):
        return (
            # constraint
            problem.F2(x_prev)
            + problem.rho_hat / 2 * np.linalg.norm(x_prev - x_reg) ** 2
            # constraint extrapolation via sub-differential, reuse precomputed sub-gradient
            + np.inner(
                partial_F2_xPrev + problem.rho_hat * (x_prev - x_reg), x_curr - x_prev
            )
        )

    ell_prev = problem.F2(x_curr)
    ell_curr = problem.F2(x_curr)

    for k in range(max_iter):
        # dual update
        inner_t = k % iter_PPM
        s_k = (1 + theta_k(inner_t)) * ell_curr + theta_k(inner_t) * ell_prev
        y_k = ReLU(y_k + tau_k(inner_t) * s_k)

        g1 = problem.partial_F1(x_curr)
        g2 = problem.partial_F2(x_curr)

        z_k = (
            g1
            + problem.rho_hat * (x_curr - x_reg)
            + (g2 + problem.rho_hat * (x_curr - x_reg)) * y_k
        )
        # primal updates
        x_prev = x_curr.copy()
        x_curr -= eta_k(inner_t) * z_k
        # EM update
        run_sum_iterate += gamma_k(inner_t) * x_curr
        run_sum_coeff += gamma_k(inner_t)

        # PPM update
        if inner_t == 0 and k > 0:
            x_EM = run_sum_iterate / run_sum_coeff
            x_reg = x_EM.copy()
            iter_history.append(k)
            history.append(x_EM.copy())
            # restart online ergodic mean computation
            run_sum_iterate = np.zeros_like(x0)
            run_sum_coeff = 0

        # update constraint extrapolation
        ell_prev = ell_curr
        ell_curr = conEx_step(x_curr, x_prev, x_reg, g2)

    # return x_curr, np.array(iter_history), np.array(history)
    return OptimizationResult(
        x_final=x_curr,
        iter_history=np.array(iter_history),
        history=np.array(history),
        label="IPPM+ConEx",
        marker="*",
        linestyle=(0, (3, 5, 1, 5, 1, 5)),
        color=eth_purple,
    )


# method proposed by Huang and Lin, 2023, NeurIPS 2023
def switching_subgradient(
    x0: np.ndarray,
    max_iter: int,
    step_size: float,
    problem: ProblemAbstract,
    constraint_viol_tol: float,
    const_step_size: bool = False,
):
    x = x0.copy()
    history = [x0.copy()]
    iter_history = [0.01]
    if const_step_size:
        alpha_t = lambda is_feasible, F2_val, partial_F2: step_size
    else:
        alpha_t = step_size_gen_WCWC(
            epsilon=constraint_viol_tol,
            rho=problem.rho,
            nu=problem.nu,
            L1=problem.G_subgradients,
        )

    is_feasible = False
    for k in range(max_iter):
        # constraint violated -> subgradient step on F_2
        is_feasible = False
        if problem.F2(x) > constraint_viol_tol:
            g = problem.partial_F2(x)
        else:
            g = problem.partial_F1(x)
            is_feasible = True
        # g = subgradient_F(x)

        # Update step
        x -= alpha_t(is_feasible=is_feasible, F2_val=problem.F2(x), partial_F2=g) * g

        # Store the current point
        iter_history.append(k)
        history.append(x.copy())

    # return x, np.array(iter_history), np.array(history)
    return OptimizationResult(
        x_final=x,
        iter_history=np.array(iter_history),
        history=np.array(history),
        label="SwSG",
        marker="dotted",
        linestyle=(0, (3, 5, 1, 5, 1, 5)),
        color=eth_purple,
    )
