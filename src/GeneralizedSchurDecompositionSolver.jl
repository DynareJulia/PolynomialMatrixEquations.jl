using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK: gges!
struct GsSolverWs
    tmp1::Matrix{Float64}
    tmp2::Matrix{Float64}
    g1::Matrix{Float64}
    g2::Matrix{Float64}
    luws1::LUWs
    luws2::LUWs
    schurws::GeneralizedSchurWs{Float64}
    
    function GsSolverWs(d,n1)
        n = size(d,1)
        n2 = n - n1
        tmp1 = similar(d, n1, n1)
        tmp2 = similar(d, n1, n1)
        g1   = similar(d, n1, n1)
        g2   = similar(d, n2, n1)
        luws1 = LUWs(tmp1)
        luws2 = LUWs(n2)
        schurws = GeneralizedSchurWs(d)
        new(tmp1,tmp2,g1,g2, luws1, luws2, schurws)
    end
end

#"""
#    gs_solver!(ws::GsSolverWs, d::Matrix{Float64}, e::Matrix{Float64}, n1::Int64, qz_criterium)
#
#finds the unique stable solution for the following system:
#
#```
#d \left[\begin{array}{c}I\\g_2\end{array}\right]g_1 = e \left[\begin{array}{c}I\\g_2\end{array}\right]
#```
#The solution is returned in ``ws.g1`` and ``ws.g2``
#"""
function gs_solver!(ws::GsSolverWs, d::Matrix{Float64}, e::Matrix{Float64}, n1::Int64, qz_criterium::Union{Float64, FastLapackInterface.SCHURORDER} = 1 + 1e-6)
    gges_select!(ws, d, e,  qz_criterium)
    nstable = ws.schurws.sdim[]::Int
    n = size(d, 1)
    if nstable < n1
        throw(UnstableSystemException())
    elseif nstable > n1
        throw(UndeterminateSystemException())
    end
    
    transpose!(ws.g2, view(ws.schurws.vsr, 1:nstable, nstable+1:n))
    lu_t = LU(factorize!(ws.luws2, view(ws.schurws.vsr,nstable+1:n, nstable+1:n))...)
    ldiv!(lu_t', ws.g2)
    lmul!(-1.0,ws.g2)
    
    transpose!(ws.tmp1, view(ws.schurws.vsr, 1:nstable, 1:nstable))
    lu_t = LU(factorize!(ws.luws1, view(d, 1:nstable,1:nstable))...)
    ldiv!(lu_t', ws.tmp1)

    transpose!(ws.tmp2, view(e,1:nstable,1:nstable))
    lu_t = LU(factorize!(ws.luws1, view(ws.schurws.vsr,1:nstable, 1:nstable))...)
    ldiv!(lu_t', ws.tmp2)
    mul!(ws.g1, ws.tmp1', ws.tmp2', 1.0, 0.0)
end

function gges_select!(ws::GsSolverWs, d, e, qz_criterium::Number)
    # This is a closure
    return gges!(ws.schurws, 'N', 'V', e, d, select = (αr, αi, β) -> αr^2 + αi^2 < qz_criterium * β^2)
end

function gges_select!(ws::GsSolverWs, d, e, qz_criterium::FastLapackInterface.SCHURORDER)
    # This is not a closure
    return gges!(ws.schurws, 'N', 'V', e, d, select = qz_criterium)
end

