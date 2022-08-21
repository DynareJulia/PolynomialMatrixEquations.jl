module PolynomialMatrixEquations

struct UnstableSystemException <: Exception end
struct UndeterminateSystemException <: Exception end
export UnstableSystemException, UndeterminateSystemException
using FastLapackInterface
# include("CyclicReduction.jl")
# export CyclicReductionWs, cyclic_reduction!, cyclic_reduction_check

include("GeneralizedSchurDecompositionSolver.jl")
export GsSolverWs, gs_solver!



end # module
