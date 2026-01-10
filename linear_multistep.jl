using LinearAlgebra
include("generation/multistep.jl")
include("generation/bdf.jl")
setprecision(BigFloat, 200)

#defines the finite difference Jacobian
function finite_diff_jac(fun, x, eps=1e-8)
    n = length(x)
    f0 = fun(x)
    J = zeros(n, n)
    for j in 1:n
        dx = zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    end
    return J
end

#solves the nonlinear system of equations
function newton_solve(residual, y0, jac=nothing, tol=1e-10, max_iter=12)
    y = copy(y0)
    for _ in 1:max_iter
        r = residual(y)
        if norm(r) < tol
            return y
        end
        J = jac !== nothing ? jac(y) : finite_diff_jac(residual, y)
        dy = J \ (-r)
        y += dy
        if norm(dy) < tol
            break
        end
    end
    return y
end

#defines the RK4 step
function _rk4_step(f, t, y, h)
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
end

#bootstraps the first steps with RK4
function _bootstrap_rk4(f, t_grid, Y, F, start_idx, steps, h)
    for i in 1:steps
        n = start_idx + i - 1
        Y[n + 1, :] = _rk4_step(f, t_grid[n], Y[n, :], h)
        F[n + 1, :] = f(t_grid[n + 1], Y[n + 1, :])
    end
end


#defines the AB solver
function solve_ab(f, t_span, y0, h, order=3, prec=500, adaptive=false)
    b = [Float64(x) for x in get_ab_coeffs(order, prec, adaptive)]
    k = length(b)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N + 1, length(y0))
    F = zeros(N + 1, length(y0))
    Y[1, :] = y0
    F[1, :] = f(t_grid[1], Y[1, :])

    #bootstraps the first steps with RK4
    bootstrap_steps = min(k - 1, N)
    _bootstrap_rk4(f, collect(t_grid), Y, F, 1, bootstrap_steps, h)
    if N <= k - 1
        return collect(t_grid), Y
    end

    #main solver loop for AB
    for n in k:N
        acc = zeros(length(y0))
        for j in 1:k
            acc += b[j] * F[n + 1 - j, :]
        end
        Y[n + 1, :] = Y[n, :] + h * acc
        F[n + 1, :] = f(t_grid[n + 1], Y[n + 1, :])
    end

    return collect(t_grid), Y
end

#defines the AM solver
function solve_am(f, t_span, y0, h, order=3, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8, prec=500, adaptive=false)
    b = [Float64(x) for x in get_am_coeffs(order, prec, adaptive)]
    k = length(b)
    b0 = b[1]
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N + 1, length(y0))
    F = zeros(N + 1, length(y0))

    Y[1, :] = y0
    F[1, :] = f(t_grid[1], Y[1, :])

    #bootstraps the first steps with RK4
    bootstrap_steps = min(k - 1, N)
    _bootstrap_rk4(f, collect(t_grid), Y, F, 1, bootstrap_steps, h)
    if N <= k - 1
        return collect(t_grid), Y
    end

    #main solver loop for AM
    for n in k:N
        t_next = t_grid[n + 1]
        known = sum([b[j] * F[n + 2 - j, :] for j in 2:k])

        function R(y_next)
            return y_next - Y[n, :] - h * (b0 * f(t_next, y_next) + known)
        end

        function J(y_next)
            Jf = jac !== nothing ? jac(t_next, y_next) : finite_diff_jac(z -> f(t_next, z), y_next, fd_eps)
            return I(length(y0)) - h * b0 * Jf
        end

        y_guess = Y[n, :]
        y_next = newton_solve(R, y_guess, J, tol, max_iter)
        Y[n + 1, :] = y_next
        F[n + 1, :] = f(t_next, y_next)
    end

    return collect(t_grid), Y
end

#defines the BDF solver (alpha[0]=1, sum alpha[j] y_{n+1-j} = beta0*h*f_{n+1})
function solve_bdf(f, t_span, y0, h, order=2, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8)
    a, beta0 = bdf_coeffs(order)
    # generator returns mpmath types; cast to float
    a = [Float64(val) for val in a]
    beta0 = Float64(beta0)
    t0, tf = t_span
    N = Int(ceil((tf - t0) / h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N + 1, length(y0))

    Y[1, :] = y0

    #bootstraps the first steps with RK4
    bootstrap_steps = min(order - 1, N)
    F_boot = zeros(N + 1, length(y0))
    F_boot[1, :] = f(t_grid[1], Y[1, :])
    _bootstrap_rk4(f, collect(t_grid), Y, F_boot, 1, bootstrap_steps, h)
    if N <= order - 1
        return collect(t_grid), Y
    end

    for n in order:N
        t_next = t_grid[n + 1]
        known = zeros(length(y0))
        for j in 2:(order+1)
            known += a[j] * Y[n + 2 - j, :]
        end

        function R(y_next)
            return a[1] * y_next + known - beta0 * h * f(t_next, y_next)
        end

        function J(y_next)
            Jf = jac !== nothing ? jac(t_next, y_next) : finite_diff_jac(z -> f(t_next, z), y_next, fd_eps)
            return a[1] * I(length(y0)) - beta0 * h * Jf
        end

        y_guess = Y[n, :]
        y_next = newton_solve(R, y_guess, J, tol, max_iter)
        Y[n + 1, :] = y_next
    end
    return collect(t_grid), Y
end