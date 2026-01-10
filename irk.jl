using LinearAlgebra
setprecision(BigFloat, 200)
include("generation/gauss_legendre.jl")
include("generation/radau.jl")
include("generation/lobatto.jl")

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

#loads the Butcher tableau from the generators
function get_tableau(family, s)
    family = lowercase(family)
    if family == "gauss"
        A, b, c = build_gauss_legendre_irk(s)
    elseif family == "radau"
        A, b, c = build_radau_irk(s)
    elseif family == "lobatto"
        A, b, c = build_lobatto_IIIC_irk(s)
    else
        error("Unknown family '$family', must be 'gauss', 'radau', or 'lobatto'.")
    end

    #converts the tableau to numpy arrays
    A = [[Float64(A[i, j]) for j in 1:s] for i in 1:s]
    b = [Float64(b[i]) for i in 1:s]
    c = [Float64(c[i]) for i in 1:s]
    return A, b, c
end


#defines the IRK collocation step
function step_collocation(f, t, y, h, A, b, c, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8)
    s = length(b)
    n = length(y)
    Y = repeat(reshape(y, 1, length(y)), s, 1)
    t_nodes = [t + c[i] * h for i in 1:s]

    #builds the residual
    function residual(z_flat)
        Z = reshape(z_flat, s, n)
        R = zeros(size(Z))
        for i in 1:s
            acc = zeros(n)
            for j in 1:s
                acc += A[i][j] * f(t_nodes[j], Z[j, :])
            end
            R[i, :] = Z[i, :] - y - h * acc
        end
        return vec(R)
    end

    #builds the Jacobian
    function jacobian(z_flat)
        Z = reshape(z_flat, s, n)
        J_full = zeros(s * n, s * n)
        for j in 1:s
            Jf_j = jac !== nothing ? jac(t_nodes[j], Z[j, :]) : finite_diff_jac(z -> f(t_nodes[j], z), Z[j, :], fd_eps)
            for i in 1:s
                block = -h * A[i][j] * Jf_j
                if i == j
                    block = block + I(n)
                end
                row = (i-1)*n+1:i*n
                col = (j-1)*n+1:j*n
                J_full[row, col] = block
            end
        end
        return J_full
    end

    z0 = vec(Y)
    z_star = newton_solve(residual, z0, jacobian, tol, max_iter)
    Y = reshape(z_star, s, n)
    K = zeros(s, n)
    for i in 1:s
        K[i, :] = f(t_nodes[i], Y[i, :])
    end
    y_next = y + h * sum([b[i] * K[i, :] for i in 1:s])
    return y_next
end


#main solver for any collocation method
function solve_collocation(f, t_span, y0, h, family="gauss", s=3, jac=nothing, tol=1e-10, max_iter=12, fd_eps=1e-8)
    A, b, c = get_tableau(family, s)
    t0, tf = t_span
    N = Int(ceil((tf - t0)/h))
    t_grid = range(t0, tf, length=N+1)
    Y = zeros(N+1, length(y0))
    Y[1, :] = y0
    for n in 1:N
        Y[n+1, :] = step_collocation(f, t_grid[n], Y[n, :], h, A, b, c, jac, tol, max_iter, fd_eps)
    end
    return collect(t_grid), Y
end