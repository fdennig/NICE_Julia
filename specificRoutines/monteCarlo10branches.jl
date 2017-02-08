using HDF5, JLD, DataFrames, DataArrays, StatsBase, NLopt, Roots
folder = pwd()
include("$folder/Function_definitions.jl")
include("$folder/createPrandom.jl")
include("$folder/optimisations.jl")
eeMv = [1]
regimes = [9 16 18 19]
lmv = [0 3 tm]
runs = 2
resArray = Array{Results}(length(eeMv),length(regimes),length(lmv),runs)
WELF = DataFrame(Elasticity = Int[], Regime = Int[], Learnperiod = Int[], Welfare = Float64[])
for (g,eeM) in enumerate(eeMv)
for (i,lm) in enumerate(lmv)
for (h,regime) in enumerate(regimes)
PP = createP(regime)
for j=1:runs
resArray[g,h,i,j] = optimiseNICER10(PP,regime, lm=lm))
lp = 2015+lm*10
WELFA = DataFrame(Elasticity = eeM, Regime = regime, Learnperiod = lp, Welfare = resArray[g,h,i,j].EWelfare)
WELF = append!(WELF,WELFA)
writetable("$(pwd())/Outputs/valueOfLearning/welfares10branches.csv",WELF)
println("lp=$(lp)regime=$(regime)run=$j")
end
end
end
end
