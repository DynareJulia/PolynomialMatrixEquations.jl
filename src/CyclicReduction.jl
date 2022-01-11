using FastLapackInterface: LinSolveWs, linsolve_core!

using LinearAlgebra
using LinearAlgebra.BLAS: gemm!
export CyclicReductionWs, cyclic_reduction!, cyclic_reduction_check

mutable struct CyclicReductionWs
    linsolve_ws::LinSolveWs
    ahat1::Matrix{Float64}
    a1copy::Matrix{Float64}
    m::Matrix{Float64,}
    m00::SubArray{Float64}
    m02::SubArray{Float64}
    m20::SubArray{Float64}
    m22::SubArray{Float64}
    m1::Matrix{Float64}
    m1_a0::SubArray{Float64}
    m1_a2::SubArray{Float64}
    m2::Matrix{Float64}
    m2_a0::SubArray{Float64}
    m2_a2::SubArray{Float64}
    info::Int64

    function CyclicReductionWs(n)
        linsolve_ws = LinSolveWs(n)
        ahat1 = Matrix{Float64}(undef, n,n)
	a1copy = Matrix{Float64}(undef, n,n)
        m = Matrix{Float64}(undef, 2*n,2*n)
        m00 = view(m,1:n,1:n)
        m02 = view(m,1:n,n .+ (1:n))
        m20 = view(m,n .+ (1:n), 1:n)
        m22 = view(m,n .+ (1:n),n .+(1:n))
        m1 = Matrix{Float64}(undef, n, 2*n)
        m1_a0 = view(m1,1:n,1:n)
        m1_a2 = view(m1,1:n,n .+ (1:n))
        m2 = Matrix{Float64}(undef, 2*n, n)
        m2_a0 = view(m2, 1:n, 1:n)
        m2_a2 = view(m2, n .+ (1:n), 1:n)
        info = 0
        new(linsolve_ws, ahat1, a1copy, m, m00, m02, m20, m22,
            m1, m1_a0, m1_a2, m2, m2_a0, m2_a2, info) 
    end
end

"""
    cyclic_reduction!(x::Array{Float64},a0::Array{Float64},a1::Array{Float64},a2::Array{Float64},ws::CyclicReductionWs, cvg_tol::Float64, max_it::Int64)

Solve the quadratic matrix equation a0 + a1*x + a2*x*x = 0, using the cyclic reduction method from Bini et al. (???).

The solution is returned in matrix x. In case of nonconvergency, x is set to NaN and 
UndeterminateSystemExcpetion or UnstableSystemException is thrown

# Example
```meta
DocTestSetup = quote
     using CyclicReduction
     n = 3
     ws = CyclicReductionWs(n)
     a0 = [0.5 0 0; 0 0.5 0; 0 0 0];
     a1 = eye(n)
     a2 = [0 0 0; 0 0 0; 0 0 0.8]
     x = zeros(n,n)
end
```

```jldoctest
julia> display(names(CyclicReduction))
```

```jldoctest
julia> cyclic_reduction!(x,a0,a1,a2,ws,1e-8,50)
```
"""
function cyclic_reduction!(x::AbstractMatrix{Float64},
                           a0::AbstractMatrix{Float64},
                           a1::AbstractMatrix{Float64},
                           a2::AbstractMatrix{Float64},
                           ws::CyclicReductionWs,
                           cvg_tol::Float64,
                           max_it::Int64)
    copyto!(x,a0)
    copyto!(ws.ahat1,1,a1,1,length(a1))
    @inbounds copyto!(ws.m1_a0, a0)
    @inbounds copyto!(ws.m1_a2, a2)
    @inbounds copyto!(ws.m2_a0, a0)
    @inbounds copyto!(ws.m2_a2, a2)
    it = 0
    @inbounds while it < max_it
        #        ws.m = [a0; a2]*(a1\[a0 a2])
	copyto!(ws.a1copy,a1)
        linsolve_core!(ws.a1copy, ws.m1, ws.linsolve_ws)
        gemm!('N','N',-1.0,ws.m2,ws.m1,0.0,ws.m)
        @simd for i in eachindex(a1)
            a1[i] += ws.m02[i] + ws.m20[i]
        end
        copyto!(ws.m1_a0, ws.m00)
        copyto!(ws.m1_a2, ws.m22)
        copyto!(ws.m2_a0, ws.m00)
        copyto!(ws.m2_a2, ws.m22)
        if any(isinf.(ws.m)) || any(isnan.(ws.m))
            fill!(x,NaN)
            if norm(ws.m1_a0) < Inf
                throw(UndeterminateSystemException())
            else
                throw(UnstableSystemException())
            end
        end
        ws.ahat1 += ws.m20
        crit = norm(ws.m1_a0,1)
        if crit < cvg_tol
	    # keep iterating until condition on a2 is met
            if norm(ws.m1_a2,1) < cvg_tol
                break
            end
        end
        it += 1
    end
    if it == max_it
        println("max_it")
        if norm(ws.m1_a0) < cvg_tol
            throw(UnstableSystemException())
        else
            throw(UndeterminateSystemException())
        end
        fill!(x,NaN)
        return
    else
        linsolve_core!(ws.ahat1, x, ws.linsolve_ws)
        @inbounds lmul!(-1.0,x)
        ws.info = 0
    end
end

function cyclic_reduction_check(x::Array{Float64,2},a0::Array{Float64,2}, a1::Array{Float64,2}, a2::Array{Float64,2},cvg_tol::Float64)
    res = a0 + a1*x + a2*x*x
    if (sum(sum(abs.(res))) > cvg_tol)
        print("the norm of the residuals, ", res, ", compared to the tolerance criterion ",cvg_tol)
    end
    nothing
end

