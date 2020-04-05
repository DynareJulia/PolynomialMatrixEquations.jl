using PolynomialMatrixEquations
using LinearAlgebra
using Test

n = 3
ws = CyclicReductionWs(n)
a0 = [0.5 0 0; 0 0.5 0; 0 0 0];
a1 = I(n) .+ zeros(n, n)
a2 = [0 0 0; 0 0 0; 0 0 0.8]
x = zeros(n,n)
cyclic_reduction!(x,a0,a1,a2,ws,1e-8,50)
@test ws.info == 0
@test a0 + a1*x + a2*x*x ≈ zeros(n, n)
display(x)

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
