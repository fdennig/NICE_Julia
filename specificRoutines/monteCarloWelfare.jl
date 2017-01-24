###########################################################################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Preliminaries
using HDF5, JLD, DataFrames, DataArrays, StatsBase, NLopt, Distributions, Gadfly, Cairo

folder = pwd()
include("$folder/Function_definitions.jl")
include("$folder/createPrandom.jl")
#Regimes
  #20: Normally distributed damage elasticity with mean=eeM and sd=eesd
Tm = 32
tm = 31
rho = 0.015 # discount rate
eta = 2 # inequality aversion/time smoothing parameter (applied to per capita consumption by region, quintile and time period)
nu = 2 # risk aversion parameter (applied over random draws)
eeM = 1 # elasticity of damage with respect to income
model="NICE"
regime = 9
lmv = [3 tm]
#compute optima
# pre allocate the dataframes and results array
resArray = Array{Results}(2,100)
initials = Array{Vector{Float64}}(length(lmv),100)
WELF = DataFrame(Learnperiod = -1, Welfare = 0.2)
# optimisation preliminaries
PP = createP(regime) # createP(regime_select; backstop_same = "Y", gy0M = dparam_i["gy0"][2]', gy0sd = ones(12)*0.0059, sighisTM = dparam_i["sighisT"][2]', sighisTsd = ones(12)*0.0064, xi1M = 3, xi1sd = 1.4, eeM = 1, eesd = 0.3)
nsample=length(PP)
pb = zeros(nsample,12,60)
for ii=1:nsample
  pb[ii,:,:] = PP[ii].pb'
end
idims = Int(max(round(nsample/2),1)) # bifurcates the random draws into two subsets
for (i, lm) in enumerate(lmv)
  for j=1:100
    tax_length = 2*tm - lm
    global count = 0 # keep track of number of iterations
    # Define function to be maximized (requires special format for NLopt package)
    function welfaremax(x,grad) # x is the tax vector, grad is the gradient (unspecified here since we have no way of computing it)
      WW = tax2expectedwelfare(x,PP,rho,eta,nu,Tm,tm,lm,idims,model="$(model)")[1] #change to model="RICE" or "DICE" for different levels of aggregation
      count += 1
      if count%100==0
        println("f_$count($x)")
      end
      return WW
    end
    n = tax_length
    opt = Opt(:LN_BOBYQA, n) # algorithm and dimension of tax vector, possible (derivative-free) algorithms: LN_COBYLA, LN_BOBYQA
    ub_lm = maximum(squeeze(maximum(pb,2),2),1)[2:lm+1]
    ub_1 = maximum(squeeze(maximum(pb,2),2)[1:idims,:],1)[lm+2:tm+1]
    ub_2 = maximum(squeeze(maximum(pb,2),2)[(idims+1):nsample,:],1)[lm+2:tm+1]
    # lower bound is zero
    lower_bounds!(opt, zeros(n))
    upper_bounds!(opt, [ub_lm; ub_1; ub_2])
    # Set maximization
    max_objective!(opt, welfaremax)
    # Set relative tolerance for the tax vector choice - vs. ftol for function value?
    ftol_rel!(opt,0.000000000005)
    # Optimize! Initial guess defined above

    init = Array{Float64}(n)
    # init = [ub_lm; ub_1; ub_2]*rand(Uniform(0,1),1)[1]
    for k=1:n
      init[k]=rand(Uniform(0,[ub_lm; ub_1; ub_2][k]),1)[1]
    end
    (expected_welfare,tax_vector,ret) = optimize(opt,init)
    # Extract the two tax vectors from the optimization object
    taxes_a = tax_vector[1:tm]
    taxes_b = [tax_vector[1:lm];tax_vector[tm+1:end]]
    # get all endogenous variables
    c, K, T, E, M, mu, lam, D, AD, Y, Q = VarsFromTaxes(taxes_a, taxes_b, PP, nsample, model="$(model)")
    # get taxes for both learning branches
    taxes_1 = maximum(PP[1].pb[1:Tm,:],2)[:,1]
    taxes_1[1] = 0
    taxes_1[2:(tm+1)] = taxes_a
    taxes_2 = maximum(PP[1].pb[1:Tm,:],2)[:,1]
    taxes_2[1] = 0
    taxes_2[2:(tm+1)] = taxes_b

    # create Results variable
    initials[i,j] = init
    res = Results(regime,nsample,Tm,tm,lm,Regions,taxes_1,taxes_2,expected_welfare,c,K,T,E,M,mu,lam,D,AD,Y,Q,rho,eta,nu,PP)
    resArray[i,j] = res
    # create dataframe of welfares
  end
end

WELF = DataFrame(Learnperiod = Int[], Welfare = Float64[])
for i=1:2
  for j=1:100
    WELFA = DataFrame(Learnperiod = resArray[i,j].lm, Welfare = resArray[i,j].EWelfare)
    WELF = append!(WELF, WELFA)
  end
end
JLD.save("$(pwd())/Outputs/valueOfLearning/resArrayHistogram2.jld", "resArray", resArray)
monteWelfare = plot(WELF, x="Welfare", color="Learnperiod",Geom.histogram)
xa = minimum(WELF[:Welfare])
WE = WELF[WELF[:Welfare].!=xa]
monteWelfare2 = plot(WE, x="Welfare", color="Learnperiod",Geom.histogram)
saveFolder = "$(pwd())/Outputs/valueOfLearning"
draw(PDF("$(saveFolder)/histogram2.pdf", 6inch, 4inch), monteWelfare)
draw(PDF("$(saveFolder)/histogram2b.pdf", 6inch, 4inch), monteWelfare2)
