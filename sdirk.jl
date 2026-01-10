using LinearAlgebra

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

#defines the SDIRK step
function step_sdirk(f, t, y, h, A, b, c, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8)
    s = length(b)
    n = length(y)
    Y = repeat(reshape(y, 1, length(y)), s, 1)
    K = zeros(s, n)

    for i in 1:s
        t_i = t + c[i] * h
        g_known = i > 1 ? sum([A[i, j] * K[j, :] for j in 1:(i-1)]) : zeros(n)

        function R(z)
            return z - y - h * (g_known + A[i, i] * f(t_i, z))
        end

        function J(z)
            Jf = jac !== nothing ? jac(t_i, z) : finite_diff_jac(zz -> f(t_i, zz), z, fd_eps)
            return I(n) - h * A[i, i] * Jf
        end

        y_guess = Y[i, :]
        y_i = newton_solve(R, y_guess, J, tol, max_iter)
        Y[i, :] = y_i
        K[i, :] = f(t_i, y_i)
    end

    y_next = y + h * sum([b[i] * K[i, :] for i in 1:s])
    return y_next
end

#defines the SDIRK2 solver
function solve_sdirk2(f, t_span, y0, h, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8)
    gamma = 1.0 - 1.0/sqrt(2.0)
    A = [gamma 0.0; 1.0 - gamma gamma]
    b = [1.0 - gamma, gamma]
    c = [gamma, 1.0]
    t0, tf = t_span
    N = Int(ceil((tf - t0)/h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N+1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n+1, :] = step_sdirk(f, t_grid[n], Y[n, :], h, A, b, c, jac, tol, max_iter, fd_eps)
    end
    return collect(t_grid), Y
end

#defines the SDIRK3 solver
function solve_sdirk3(f, t_span, y0, h, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8)
    gamma = 0.435866521508459
    A = [
        gamma 0.0 0.0;
        0.2820667395 gamma 0.0;
        1.208496649 -0.644363171 gamma
    ]
    b = [1.208496649, -0.644363171, gamma]
    c = [gamma, 0.7179332605, 1.0]
    t0, tf = t_span
    N = Int(ceil((tf - t0)/h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N+1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n+1, :] = step_sdirk(f, t_grid[n], Y[n, :], h, A, b, c, jac, tol, max_iter, fd_eps)
    end
    return collect(t_grid), Y
end

#defines the SDIRK4 solver
function solve_sdirk4(f, t_span, y0, h, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8)
    gamma = 0.572816062482135
    a21 = 0.5 - gamma
    a31 = 2 * gamma
    a32 = 1 - 4 * gamma
    a41 = 2 * gamma
    a42 = 1 - 4 * gamma
    a43 = gamma
    A = [
        gamma 0.0 0.0 0.0;
        a21  gamma 0.0 0.0;
        a31  a32  gamma 0.0;
        a41  a42  a43 gamma
    ]
    b = [a41, a42, a43, gamma]
    c = [gamma, a21 + gamma, a31 + a32 + gamma, 1.0]

    t0, tf = t_span
    N = Int(ceil((tf - t0)/h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N+1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n+1, :] = step_sdirk(f, t_grid[n], Y[n, :], h, A, b, c, jac, tol, max_iter, fd_eps)
    end
    return collect(t_grid), Y
end