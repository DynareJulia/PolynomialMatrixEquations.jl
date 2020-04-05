using PolynomialMatrixEquations
using LinearAlgebra
using Random
using Test

Random.seed!(132)
n = 3
d_org = randn(n,n)
e_org = randn(n,n)

n1 = count(abs.(eigen(e, d).values) .< 1+1e-6)

ws = GsSolverWs(d, e, n1)

qz_criterium = 1.0 + 1e-6
d = copy(d_org)
e = copy(e_org)
gs_solver!(ws, d , e, n1 , qz_criterium)
#@test ws.info == 0
@test d_org*[I(n1); ws.g2]*ws.g1 ≈ e_org*[I(n1); ws.g2]

#=
cyclic_reduction_check(x,a0,a1,a2,1e-8)
a0 = [0.5 0 0; 0 1.1 0; 0 0 0];
a1 .= I(n) .+ zeros(n, n)
a2 = [0 0 0; 0 0 0; 0 0 0.8]
cyclic_reduction!(x,a0,a1,a2,ws,1e-8,50)
@test ws.info == 1

a0 = [0.5 0 0; 0 0.5 0; 0 0 0];
a1 = I(n) .+ zeros(n, n)
a2 = [0 0 0; 0 0 0; 0 0 1.2]
cyclic_reduction!(x,a0,a1,a2,ws,1e-8,50)
@test ws.info == 2
=#
