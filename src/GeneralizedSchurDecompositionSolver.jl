using FastLapackInterface.SchurAlgo: DggesWs, dgges!
using FastLapackInterface.LinSolveAlgo: LinSolveWs, linsolve_core!
using LinearAlgebra
using LinearAlgebra.BLAS

struct GsSolverWs
    dgges_ws::DggesWs
    linsolve_ws_11::LinSolveWs
    linsolve_ws_22::LinSolveWs
    D11::SubArray{Float64}
    E11::SubArray{Float64}
    vsr::Matrix{Float64}
    Z11::SubArray{Float64}
    Z12::SubArray{Float64}
    Z21::SubArray{Float64}
    Z22::SubArray{Float64}
    tmp1::Matrix{Float64}
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}
    g1::Matrix{Float64}
    g2::Matrix{Float64}
    eigval::Vector{ComplexF64}
    
    function GsSolverWs(d,e,n1)
        dgges_ws = DggesWs(Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{UInt8}('N'), e, d)
        n = size(d,1)
        n2 = n - n1
        linsolve_ws_11 = LinSolveWs(n1)
        linsolve_ws_22 = LinSolveWs(n2)
        D11 = view(d,1:n1,1:n1)
        E11 = view(e,1:n1,1:n1)
        vsr = Matrix{Float64}(undef, n, n)
        Z11 = view(vsr,1:n1,1:n1)
        Z12 = view(vsr,1:n1,n1+1:n)
        Z21 = view(vsr,n1+1:n,1:n1)
        Z22 = view(vsr,n1+1:n,n1+1:n)
        tmp1 = Matrix{Float64}(undef, n2, n1)
        tmp2 = Matrix{Float64}(undef, n1, n1)
        tmp3 = Matrix{Float64}(undef, n1, n1)
        g1 = Matrix{Float64}(undef, n1, n1)
        g2 = Matrix{Float64}(undef, n2, n1)
        eigval = Vector{ComplexF64}(undef, n)
        new(dgges_ws,linsolve_ws_11, linsolve_ws_22, D11,E11,vsr,Z11,Z12,Z21,Z22,tmp1,tmp2,tmp3,g1,g2, eigval)
    end
end

#"""
#    gs_solver!(ws::GsSolverWs,d::Matrix{Float64},e::Matrix{Float64},n1::Int64,qz_criterium)
#
#finds the unique stable solution for the following system:
#
#```
#d \left[\begin{array}{c}I\\g_2\end{array}\right]g_1 = e \left[\begin{array}{c}I\\g_2\end{array}\right]
#```
#The solution is returned in ``ws.g1`` and ``ws.g2``
#"""
function gs_solver!(ws::GsSolverWs,d::Matrix{Float64},e::Matrix{Float64},n1::Int64,qz_criterium::Float64)

    try
        dgges!('N', 'V', e, d, zeros(1,1), ws.vsr, ws.eigval, ws.dgges_ws)
    catch err
        if err.error_nbr == size(e,1) + 2
            println("Warning: DGGES reports error $(e.error_nbr)")
        else
            rethrow(err)
        end
    end
    nstable = ws.dgges_ws.sdim[]
    
    if nstable < n1
        throw(UnstableSystemException())
    elseif nstable > n1
        throw(UndeterminateSystemException())
    end
    ws.g2 .= ws.Z12'
    linsolve_core!(ws.Z22', ws.g2, ws.linsolve_ws_22)
    lmul!(-1.0,ws.g2)
    ws.tmp2 .= ws.Z11'
    ws.D11 .= view(d,1:nstable,1:nstable)
    linsolve_core!(ws.D11', ws.tmp2, ws.linsolve_ws_11)
    ws.E11 .= view(e,1:nstable,1:nstable)
    ws.tmp3 .= ws.E11'
    linsolve_core!(ws.Z11', ws.tmp3, ws.linsolve_ws_11)
    gemm!('T','T',1.0,ws.tmp2,ws.tmp3,0.0,ws.g1)
end

