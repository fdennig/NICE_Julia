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
lmv = [0 3 tm] #[0 3 tm]#[3] #[0 3 tm]
runs = 1 #2
# resArray = Array{Results10}(length(1),length(regimes),3,2)
WELF = DataFrame(Elasticity = Int[], Regime = Int[], Learnperiod = Int[], Welfare = Float64[])
for (g,eeM) in enumerate(eeMv)
for (i,lm) in enumerate(lmv)
for (h,regime) in enumerate(regimes)
PP = createP(regime)
nsample = length(PP)
for j=1:runs
pred1 = resA[1,h,3,1].taxes[2:lm+1,1]*0.5
pred2 = zeros(tm-lm,nsample)
for jj = 1:nsample
  pred2[:,jj] = resA[1,h,3,1].taxes[lm+2:tm+1,1]*0.5
end
pre = [pred1; pred2[:]]-0.1
resArray[g,h,i,j] = optimiseNICER10(PP,regime,lm=lm,tm=tm,inite=pre)
lp = 2015+lm*10
WELFA = DataFrame(Elasticity = eeM, Regime = regime, Learnperiod = lp, Welfare = resArray[g,h,i,j].EWelfare)
WELF = append!(WELF,WELFA)
writetable("$(pwd())/Outputs/valueOfLearning/welfares10branches.csv",WELF)
println("lp=$(lp)regime=$(regime)run=$j")
end
end
end
end
# save("$(pwd())/Outputs/valueOfLearning/resArray10branchesPartial.jld", "resArray", resArray)
legend = ["x", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10"]
REGIME = ["Damage" "TFP" "Climate sensitivity" "Convergence rate"]
p = Array{Any}(4,3)
for i=1:4
  for j=1:3
    taxs = resArray[1,i,j,1].taxes[1:tm+1,:]
    df = f([convert(Array,10*(0:tm)+2005) taxs], legend)
    p[i,j] = plot(melt(df, :x), x = :x, y = :value, color = :variable, Geom.line, Geom.point,Guide.title("Regime $(REGIME[i])"))
  end
end
draw(PDF("$(pwd())/Outputs/valueOfLearning/tenBranches3.pdf", 20inch, 12inch), vstack(hstack(p[1,1],p[2,1],p[3,1],p[4,1]),hstack(p[1,2],p[2,2],p[3,2],p[4,2]),hstack(p[1,3],p[2,3],p[3,3],p[4,3])))
