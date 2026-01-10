using Printf
using QuadGK
using LinearAlgebra

#computes adams-bashforth coefficients for order k
function adams_bashforth_coeffs(k::Int, prec::Int=500)
    if k >= 12
        prec = max(prec, 1000)
    end
    
    setprecision(BigFloat, prec)
    xs = [BigFloat(-j) for j in 0:(k-1)]
    coeffs = BigFloat[]
    
    for j in 1:k
        function ell(x::BigFloat)
            num, den = one(BigFloat), one(BigFloat)
            for m in 1:k
                if m != j
                    diff_num = x - xs[m]
                    diff_den = xs[j] - xs[m]
                    num *= diff_num
                    den *= diff_den
                end
            end
            return num / den
        end
        coeff, _ = quadgk(ell, BigFloat(0), BigFloat(1), rtol=eps(BigFloat))
        push!(coeffs, coeff)
    end
    
    return coeffs
end

#computes adams-moulton coefficients for order k
function adams_moulton_coeffs(k::Int, prec::Int=500)
    if k >= 12
        prec = max(prec, 1000)
    end
    
    setprecision(BigFloat, prec)
    xs = [BigFloat(1); [BigFloat(-j) for j in 0:(k-2)]]
    coeffs = BigFloat[]
    
    for j in 1:k
        function ell(x::BigFloat)
            num, den = one(BigFloat), one(BigFloat)
            for m in 1:k
                if m != j
                    diff_num = x - xs[m]
                    diff_den = xs[j] - xs[m]
                    num *= diff_num
                    den *= diff_den
                end
            end
            return num / den
        end
        coeff, _ = quadgk(ell, BigFloat(0), BigFloat(1), rtol=eps(BigFloat))
        push!(coeffs, coeff)
    end
    
    return coeffs
end

#verifies order conditions for multistep coefficients
function check_order(coeffs::Vector{BigFloat}, kind::String)
    k = length(coeffs)
    xs = kind == "AB" ? [BigFloat(-j) for j in 0:(k-1)] : [BigFloat(1); [BigFloat(-j) for j in 0:(k-2)]]
    errs = BigFloat[]
    
    #checks the order conditions for powers 0 to k-1
    for p in 0:(k-1)
        lhs = sum(coeffs[j] * xs[j]^p for j in 1:k)
        rhs = BigFloat(1) / (p + 1)
        error = abs(lhs - rhs)
        push!(errs, error)
    end
    
    max_error = maximum(errs)
    return max_error
end

#adaptive precision to achieve target accuracy
function adaptive_precision_coeffs(k::Int, kind::String="AB", max_prec::Int=2000, target_error::Float64=1e-100)
    prec = 200
    best_coeffs = nothing
    best_error = Inf
    
    while prec <= max_prec
        try
            #computes the coefficients with the current precision
            if kind == "AB"
                coeffs = adams_bashforth_coeffs(k, prec)
            else
                coeffs = adams_moulton_coeffs(k, prec)
            end
            
            #checks the accuracy of the computed coefficients
            current_error = check_order(coeffs, kind)
            
            if current_error < best_error
                best_coeffs = coeffs
                best_error = current_error
            end
            
            if current_error < BigFloat(target_error)
                return coeffs, prec, current_error
            end
            
            #increases the precision for the next iteration
            prec = Int(prec * 1.5)
        catch e
            break
        end
    end
    
    return best_coeffs, prec÷2, best_error
end

#saves computed coefficients to text file
function save_coefficients_to_file(coeffs::Vector{BigFloat}, filename::String, kind::String, k::Int, precision_used::Int)
    open(filename, "w") do f
        for c in coeffs
            println(f, nstr_fixed(c, 100))
        end
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

#returns the Adams-Bashforth coefficients
function get_ab_coeffs(k::Int, prec::Int=500, adaptive::Bool=false, max_prec::Int=2000, target_error::Float64=1e-100)
    if adaptive
        coeffs, _, _ = adaptive_precision_coeffs(k, "AB", max_prec, target_error)
        return coeffs
    end
    return adams_bashforth_coeffs(k, prec)
end

#returns the Adams-Moulton coefficients
function get_am_coeffs(k::Int, prec::Int=500, adaptive::Bool=false, max_prec::Int=2000, target_error::Float64=1e-100)
    if adaptive
        coeffs, _, _ = adaptive_precision_coeffs(k, "AM", max_prec, target_error)
        return coeffs
    end
    return adams_moulton_coeffs(k, prec)
end