using LinearAlgebra
using PolynomialMatrixEquations
using Random
using Test

undeterminatcase = false
unstablecas = false
numberundeterminate = 0
numberunstable = 0

qz_criterium = 1 + 1e-6

Random.seed!(123)
ncases = 20
n=10

for i = 1:ncases
    println("Test $i")
    d_orig = randn(n, n)
    e_orig = randn(n, n)
    F = schur(e_orig, d_orig)
    eigenvalues = F.α ./ F.β
    nstable = count(abs.(eigenvalues) .< 1+1e-6)

    d = copy(d_orig)
    e = copy(e_orig)
    ws1 = GsSolverWs(d, e, nstable)
    gs_solver!(ws1, d, e, nstable, qz_criterium)
    @test d_orig*[I(nstable); ws1.g2]*ws1.g1 ≈ e_orig*[I(nstable); ws1.g2]

    a0 = Matrix([-e[:, 1:nstable] zeros(n, n-nstable)])
    a1 = Matrix([d[:, 1:nstable] -e[:, (nstable+1):n]])
    a2 = Matrix([zeros(n, nstable) d[:, (nstable+1):n]])

    x = zeros(n, n)
    ws2 = CyclicReductionWs(n)
    cyclic_reduction!(x, a0, a1, a2, ws2, 1e-8, 50)
    @test isapprox(a0 + a1*x + a2*x*x, zeros(n, n); atol = 1e-12)

    nstable1 = nstable + 1
    @test_throws UnstableSystemException gs_solver!(ws1, d, e, nstable + 1, qz_criterium)
    
    a0 = Matrix([-e[:, 1:nstable1] zeros(n, n-nstable1)])
    a1 = Matrix([d[:, 1:nstable1] -e[:, (nstable1+1):n]])
    a2 = Matrix([zeros(n, nstable1) d[:, (nstable1+1):n]])

    x = zeros(n, n)
    ws2 = CyclicReductionWs(n)

    @test_throws UnstableSystemException cyclic_reduction!(x, a0, a1, a2, ws2, 1e-8, 50)

    nstable1 = nstable - 1
    @test_throws UndeterminateSystemException gs_solver!(ws1, d, e, nstable1, qz_criterium)

    a0 = Matrix([-e[:, 1:nstable1] zeros(n, n-nstable1)])
    a1 = Matrix([d[:, 1:nstable1] -e[:, (nstable1+1):n]])
    a2 = Matrix([zeros(n, nstable1) d[:, (nstable1+1):n]])

    x = zeros(n, n)
    ws2 = CyclicReductionWs(n)
    @test_throws UndeterminateSystemException cyclic_reduction!(x, a0, a1, a2, ws2, 1e-8, 50)
end
