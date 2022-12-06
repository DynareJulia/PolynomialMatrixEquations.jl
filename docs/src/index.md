# PolynomialMatrixEquations.jl

This package contains algorithm to solve various polynomial matrix equations.

# Cyclic Reduction
This algorithm adapted from Bini et al. (??) solves the second order matrix equation iteratively:
```math
A_0 + A_1 X + A_2 X^2 = \bm{0}
```
for $X$. It is implemented both for $A_0$, $A_1$ and $A_2$ dense or sparse (i.e. `SparseMatrixCSC`).
In the sparse version, a special packing is performed that maximally exploits the column and row sparsity
of the $A$ matrices.

```@docs
CyclicReductionWs
PolynomialMatrixEquations.solve!(::CyclicReductionWs{T}, ::Matrix{T}, ::Matrix{T}, ::Matrix{T}, ::Matrix{T}) where {T<:AbstractFloat}
```

# Generalized Schur Decomposition Solver
The polynomial matrix equation
```math
A_0 + A_1 X + A_2 X^2 = \bm{0}
```
can be rewritten in the form
```math
\begin{bmatrix}
0 & A_2 \\
I & 0
\end{bmatrix} 
\begin{bmatrix}
I \\ X
\end{bmatrix} X
=
-\begin{bmatrix}
A_0 & A_1 \\
0 & I
\end{bmatrix} 
\begin{bmatrix}
I \\ X
\end{bmatrix}
```
In practive, it is possible to eliminate empty columns in $A_0$, $A_1$
and $A_2$ (See Klein, 2000, and . More generally, this solver finds a solution to the following equation:
```math
D \begin{bmatrix}
I \\ X_2
\end{bmatrix} X_1
=
E \begin{bmatrix}
I \\ X_2
\end{bmatrix}
```
 where matrices $X_1$ and $X_2$ contain columns from the solution matrix $X$.

 ```@docs
GSSolverWs
PolynomialMatrixEquations.solve!(::GSSolverWs{T}, ::Matrix{T}, ::Matrix{T}, ::Int, ::Number) where {T<:AbstractFloat}
```

# References

- D.A. Bini, B. Iannazzo, B. Meini, *Numerical Solution of Algebraic
Riccati Equations*, SIAM Book Series Fundamentals of Algorithms, 2012.
- P. Klein *Using the generalized Schur form to solve a linear
  rational expectations model*, Journal of Economci Dynamics and
  Control, 2000.
- C. Sims *Solving rational expectations models*, Computational
  Economics, 2002.
  
