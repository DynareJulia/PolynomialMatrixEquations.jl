WORK IN PROGRESS

This package solves matrix polynomial equations in the form
```
A0 + A1*X + A2*X*X = 0
```
or
```
 -    -       -    -
 | I  |       | I  |
D| -  | X1 = E| -  |
 | X2 |       | X2 |
 -    -       -    -
```
 where matrices `X1` and `X2` contain columns from the solution matrix
`X`.

Two algorithms are provided

## Cyclic reduction

    cyclic_reduction!(x::Array{Float64},a0::Array{Float64},a1::Array{Float64},a2::Array{Float64},ws::CyclicReductionWs, cvg_tol::Float64, max_it::Int64)

Solve the quadratic matrix equation a0 + a1*x + a2*x*x = 0, using the cyclic reduction method from Bini et al. (???).

The solution is returned in matrix x. In case of nonconvergency, x is set to NaN and 
`UndeterminateSystemExcpetion` or `UnstableSystemException` is thrown

# Example
```
using CyclicReduction
n = 3
ws = CyclicReductionWs(n)
a0 = [0.5 0 0; 0 0.5 0; 0 0 0];
a1 = eye(n)
a2 = [0 0 0; 0 0 0; 0 0 0.8]
x = zeros(n,n)


cyclic_reduction!(x,a0,a1,a2,ws,1e-8,50)



## General Schur Decomposition

```

    gs_solver!(ws::GsSolverWs,d::Matrix{Float64},e::Matrix{Float64},n1::Int64,qz_criterium)

```

The solution is returned in ``ws.g1`` and ``ws.g2``
