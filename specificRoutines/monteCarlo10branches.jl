using HDF5, JLD, DataFrames, DataArrays, StatsBase, NLopt, Roots, Gadfly
f(A,names) = DataFrame(Any[A[:,i] for i = 1:size(A,2)], map(Symbol,names))
folder = pwd()
include("$folder/Function_definitions.jl")
include("$folder/createPrandom.jl")
include("$folder/optimisations.jl")
eeMv = [1]
regimes = [19 9 16 18] #[19 9 16 18]
Tm=32
tm=19
lmv = [0 3]#[3] #[0 3 tm]
runs = 1 #2
resArray = Array{Results10}(length(eeMv),length(regimes),length(lmv),runs)
WELF = DataFrame(Elasticity = Int[], Regime = Int[], Learnperiod = Int[], Welfare = Float64[])
for (g,eeM) in enumerate(eeMv)
for (i,lm) in enumerate(lmv)
for (h,regime) in enumerate(regimes)
PP = createP(regime)
for j=1:runs
resArray[g,h,i,j] = optimiseNICER10(PP,regime,lm=lm, tm=tm)
lp = 2015+lm*10
WELFA = DataFrame(Elasticity = eeM, Regime = regime, Learnperiod = lp, Welfare = resArray[g,h,i,j].EWelfare)
WELF = append!(WELF,WELFA)
writetable("$(pwd())/Outputs/valueOfLearning/welfares10branches.csv",WELF)
println("lp=$(lp)regime=$(regime)run=$j")
end
end
end
end
# save("$(pwd())/Outputs/valueOfLearning/resArray10branches.jld", "resArray", resArray)

taxs = resArray[1,1,1,1].taxes[1:tm+1,:]
legend = ["x", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10"]
df = f([convert(Array,10*(0:tm)+2005) taxs], legend)
plot(melt(df, :x), x = :x, y = :value, color = :variable, Geom.line, Geom.point )
