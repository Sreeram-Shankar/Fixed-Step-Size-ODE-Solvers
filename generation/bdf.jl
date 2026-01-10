using Printf
using LinearAlgebra

#computes bdf coefficients
function bdf_coeffs(k::Int, prec::Int=300)
    setprecision(BigFloat, prec)
    h = BigFloat(1)
    A = Matrix{BigFloat}(undef, k+1, k+1)
    for m in 0:k
        for j in 0:k
            A[m+1, j+1] = (-j*h)^m
        end
    end
    b = [m == 0 ? BigFloat(0) : h*m*(BigFloat(0)^(m-1)) for m in 0:k]
    alpha = A \ b
    alpha = [a/alpha[1] for a in alpha]
    
    #computes the rhs beta naught coefficient
    beta_0 = -sum(j * alpha[j+1] for j in 0:k)
    return alpha, beta_0
end

#verifies bdf order accuracy
function check_order_bdf(alpha::Vector{BigFloat}, beta_0::BigFloat)
    k = length(alpha) - 1
    errs = BigFloat[]
    h = BigFloat(1)
    for p in 0:k
        lhs = sum(alpha[j+1] * (-j*h)^p for j in 0:k)
        rhs = p > 0 ? h * beta_0 * p * (BigFloat(0)^(p-1)) : BigFloat(0)
        push!(errs, abs(lhs - rhs))
    end
    return maximum(errs)
end

#adaptive precision bdf computation
function adaptive_bdf(k::Int, max_prec::Int=2000, target_error::Float64=1e-80)
    prec = 200
    best_coeffs = nothing
    best_beta = nothing
    best_err = Inf
    while prec <= max_prec
        setprecision(BigFloat, prec)
        coeffs, beta_0 = bdf_coeffs(k, prec)
        err = check_order_bdf(coeffs, beta_0)
        if err < best_err
            best_err, best_coeffs, best_beta = err, coeffs, beta_0
        end
        if err < BigFloat(target_error)
            return best_coeffs, best_beta, prec, err
        end
        prec = Int(prec * 1.5)
    end
    return best_coeffs, best_beta, prec, best_err
end

#saves bdf coefficients to file
function save_bdf_to_file(coeffs::Vector{BigFloat}, beta_0::BigFloat, k::Int, prec::Int, filename::String="")
    if filename == ""
        filename = "Julia_BDF_s$(k)_high_precision.txt"
    end
    open(filename, "w") do f
        for c in coeffs
            println(f, nstr_fixed(c, 80))
        end
        println(f, nstr_fixed(beta_0, 80))
    end
end

#formats the number string
function nstr_fixed(x::BigFloat, digits::Int=80)
    x_float = Float64(x)
    multiplier = 10.0^digits
    rounded = round(x_float * multiplier) / multiplier
    s = string(rounded)
    if occursin(".", s)
        s = rstrip(s, '0')
        s = rstrip(s, '.')
    end
    return s
end