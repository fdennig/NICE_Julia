using HDF5, JLD, DataFrames, DataArrays, StatsBase, NLopt, Roots, Gadfly, Distributions
f(A,names) = DataFrame(Any[A[:,i] for i = 1:size(A,2)], map(Symbol,names))
folder = pwd()
folderLearning = "$(pwd())/Outputs/valueOfLearning"
include("$(pwd())/Function_definitions.jl")
include("$(pwd())/createPrandom.jl")
include("$(pwd())/optimisations.jl")
resAL = load("$folderLearning/resArray10branchesmarch15.jld")
resA = resAL["resArray"]
eeMv = [-1 0 1]
regimes = [19 9 16 18]
Tm=32
tm=22
lmv = [0 3 tm]
runs = 8 #15 #2
ND = Normal()
                                    # resArray = Array{Results10}(length(eeMv),length(regimes),length(lmv),runs)
                                  # initials = Array{Vector{Float64}}(length(eeMv),length(regimes),length(lmv),runs)
                                    # WELF = DataFrame(Elasticity = Int[], Regime = Int[], Learnperiod = Int[], Welfare = Float64[])
for j=1:runs
for (g,eeM) in enumerate(eeMv)
for (i,lm) in enumerate(lmv)
for (h,regime) in enumerate(regimes)
PP = createP(regime)
nsample = length(PP)

  if j==1
    pred1 = resA[1,h,i,1].taxes[2:lm+1,1]*0.9
    pred2 = zeros(tm-lm,nsample)
    for jj = 1:nsample
      pred2[:,jj] = resA[1,h,i,1].taxes[lm+2:tm+1,jj]*0.9+0.01
    end
  else
    pred1 = resArray[g,h,i,j-1].taxes[2:lm+1,1]*0.9+rand(ND,lm)
    pred2 = zeros(tm-lm,nsample)
    for jj = 1:nsample
      pred2[:,jj] = (resArray[g,h,i,j-1].taxes[lm+2:tm+1,jj]+0.01)*0.99 + max(0,rand(ND,tm-lm))
    end
  end
  pre = [pred1; pred2[:]]
  resArray[g,h,i,j] = optimiseNICER10(PP,regime,lm=lm,tm=tm,inite=pre)
  lp = 2015+lm*10
  WELFA = DataFrame(Elasticity = eeM, Regime = regime, Learnperiod = lp, Welfare = resArray[g,h,i,j].EWelfare)
  WELF = append!(WELF,WELFA)
  writetable("$(pwd())/Outputs/valueOfLearning/tenBranches/welfares10branches.csv",WELF)
  initials[g,h,i,j] = pre
  save("$(pwd())/Outputs/valueOfLearning/tenBranches/initials.jld","initials",initials)
  save("$(pwd())/Outputs/valueOfLearning/tenBranches/results.jld","resArray",resArray)
  println("e=$(eeM)lp=$(lp)regime=$(regime)run=$j")
end
end
end
end

df = readtable("$(pwd())/Outputs/valueOfLearning/tenbranches/welfarestenbranches.csv")  #"$(pwd())/Outputs/valueOfLearning/welfaresxi10m1.csv"
rNames = ["TFP", "CS", "CR", "Damage"]
rCodes = [9, 16, 18, 19]
dfd=DataFrame(Elasticity = Int[], Regime = String[], WLearn2015 = Float64[], WLearn2045 = Float64[], WLearn2325 = Float64[])
for i=[1,0,-1]
  for j=1:4
    x = maximum(df[(df[:Elasticity].==i)&(df[:Regime].==rCodes[j])&(df[:Learnperiod].==2015),:Welfare])
    y = maximum(df[(df[:Elasticity].==i)&(df[:Regime].==rCodes[j])&(df[:Learnperiod].==2045),:Welfare])
    z = maximum(df[(df[:Elasticity].==i)&(df[:Regime].==rCodes[j])&(df[:Learnperiod].==2325),:Welfare])
    push!(dfd, [i, rNames[j], x, y, z])
  end
end
function learnValue(dW, res, eta)
vl = ((1+(1-eta)*dW/sum(repmat(res.PP[1].L[1,:]/5,1,5).*res.c[1,:,:,1].^(1-eta))).^(1/(1-eta))-1)*100
end
folder = pwd()
include("$folder/Function_definitions.jl")
resA = load("$(pwd())/Outputs/valueOfLearning/tenbranches/results.jld","results") #load("$(pwd())/Outputs/valueOfLearning/resArrayNICE.jld","resArray")
res = resA["resArray"][1,1,1,1]
dfd[:Value2015to2045] = learnValue(dfd[:WLearn2015] - dfd[:WLearn2045], res,2)
dfd[:Value2045to2325] = learnValue(dfd[:WLearn2045] - dfd[:WLearn2325], res,2)
output = DataFrame(Elasticity=dfd[:Elasticity], Uncertainty = dfd[:Regime], V2015to2045=dfd[:Value2015to2045], V2045to2325=dfd[:Value2045to2325])

writetable("$(pwd())/Outputs/valueOfLearning/tenbranchesVoLtable.csv",output)

using YTables
println(latex(output))


# p = Array{Any}(4,3)
# for i=1:4
#   for j=1:3
#     taxs = resArray[1,i,j,1].taxes[1:tm+1,:]
#     df = f([convert(Array,10*(0:tm)+2005) taxs], legend)
#     p[i,j] = plot(melt(df, :x), x = :x, y = :value, color = :variable, Geom.line, Geom.point,Guide.title("Regime $(REGIME[i])"))
#   end
# end
# draw(PDF("$(pwd())/Outputs/valueOfLearning/tenBranches5.pdf", 20inch, 12inch),
#         vstack(hstack(p[1,1],p[2,1],p[3,1],p[4,1]),hstack(p[1,2],p[2,2],p[3,2],p[4,2]),hstack(p[1,3],p[2,3],p[3,3],p[4,3])))
#
# JLD.save("$(pwd())/Outputs/valueOfLearning/resArray10branchesmarch15.jld", "resArray", resArray)
