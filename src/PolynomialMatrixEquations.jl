module PolynomialMatrixEquations

include("CyclicReduction.jl")
export CyclicReductionWs, cyclic_reduction!, cyclic_reduction_check

include("GeneralizedSchurDecompositionSolver.jl")
export GsSolverWs, gs_solver!



end # module
