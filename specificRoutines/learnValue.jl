###########################################################################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Preliminaries
using HDF5, JLD, DataFrames, DataArrays, StatsBase, NLopt, Roots

folder = pwd()
include("$folder/Function_definitions.jl")
include("$folder/createPrandom.jl")
#Regimes
  # 9 = Deciles of TFP (all else fixed at means) (nsample = 10)
  # 10 = Deciles of decarbonization rates (all else fixed at means) (nsample = 10)
  # 16 = DECILES of climate sensitivity
  # 17 = uncertainty (100 random draws) on the quadratic damage coefficient (full correlation)
# Define exogenous parameters
Tm = 32
tm = 18
rho = 0.015 # discount rate
eta = 2 # inequality aversion/time smoothing parameter (applied to per capita consumption by region, quintile and time period)
nu = 2 # risk aversion parameter (applied over random draws)
eeM = 1 # elasticity of damage with respect to income
models = ["NICE" "DICE"]
regimes = [9, 10, 16, 17]

#compute optima
# pre allocate the dataframes
consN = DataFrame(State = Int[], Region = String[], Year = Int[], tax = Float64[], cq1 = Float64[], cq2 = Float64[], cq3 = Float64[], cq4 = Float64[], cq5 = Float64[], K = Float64[], E = Float64[], mu = Float64[], lam = Float64[], D = Float64[], Y = Float64[], L = Float64[], Regime = Float64[], LearnP = Float64[])
consD = DataFrame(State = Int[], Region = String[], Year = Int[], tax = Float64[], c = Float64[], K = Float64[], E = Float64[], mu = Float64[], lam = Float64[], D = Float64[], Y = Float64[], L = Float64[], Regime = Float64[], LearnP = Float64[])
for k=1:2 # loop over models NICE and DICE
  for j = 1:4 # loop over regimes 9, 10, 16, 17
  # optimisation preliminaries
    PP = createP(regimes[j]) # createP(regime_select; backstop_same = "Y", gy0M = dparam_i["gy0"][2]', gy0sd = ones(12)*0.0059, sighisTM = dparam_i["sighisT"][2]', sighisTsd = ones(12)*0.0064, xi1M = 3, xi1sd = 1.4, eeM = 1, eesd = 0.3)
    nsample=length(PP)
    pb = zeros(nsample,12,60)
    for ii=1:nsample
      pb[ii,:,:] = PP[ii].pb'
    end
    idims = Int(max(round(nsample/2),1)) # bifurcates the random draws into two subsets
    for i = [0 3 tm] # loop over learning period
      lm = i
      tax_length = 2*tm - lm
      global count = 0 # keep track of number of iterations
  # Define function to be maximized (requires special format for NLopt package)
      function welfaremax(x,grad) # x is the tax vector, grad is the gradient (unspecified here since we have no way of computing it)
        WW = tax2expectedwelfare(x,PP,rho,eta,nu,Tm,tm,lm,idims,model="$(models[k])")[1] #change to model="RICE" or "DICE" for different levels of aggregation
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
      (expected_welfare,tax_vector,ret) = optimize(opt, [ub_lm; ub_1; ub_2].*0.5)

      # Extract the two tax vectors from the optimization object
      taxes_a = tax_vector[1:tm]
      taxes_b = [tax_vector[1:lm];tax_vector[tm+1:end]]
      # get all endogenous variables
      c, K, T, E, M, mu, lam, D, AD, Y, Q = VarsFromTaxes(taxes_a, taxes_b, PP, nsample, model="$(models[k])")
      # get taxes for both learning branches
      taxes_1 = maximum(PP[1].pb[1:Tm,:],2)[:,1]
      taxes_1[1] = 0
      taxes_1[2:(tm+1)] = taxes_a
      taxes_2 = maximum(PP[1].pb[1:Tm,:],2)[:,1]
      taxes_2[1] = 0
      taxes_2[2:(tm+1)] = taxes_b

      # create Results variable
      res = Results(regimes[j],nsample,Tm,tm,lm,Regions,taxes_1,taxes_2,expected_welfare,c,K,T,E,M,mu,lam,D,AD,Y,Q,rho,eta,nu,PP)
      # create dataframe of period by region by state data
      dataP = FrameFromResults(res, Tm, nsample, Regions, idims)
      dataP[:Regime] = ones(size(dataP)[1])*regimes[j]
      dataP[:LearnP] = ones(size(dataP)[1])*(2015+10*lm)
      delete!(dataP, [:ID, :A, :sigma, :th1, :pb, :EL])
      if models[k] == "NICE"
        consN = append!(consN, dataP)
      elseif models[k] == "DICE"
        consD = append!(consD, dataP)
      end
    end
  end
end

#Compute value of learning: load csvs below if optima were not just calculated
    # consN = readtable("$(pwd())/Outputs/valueOfLearning/consN.csv")
    # consD = readtable("$(pwd())/Outputs/valueOfLearning/consD.csv")
#preallocate dataframe
learn = DataFrame(Regime = repmat(regimes,4), Model = [repmat(["NICE"],8);repmat(["DICE"],8)], Learn1=repmat(repmat([2015 2045],4)[:],2), Learn2 = repmat(repmat([2045 2015+10*tm],4)[:],2))

#the NICE runs
resulN = zeros(4,2)
consN[:R] = (1/(1+rho)).^((consN[:Year]-2005)/10)
indi = 1
byRegime = groupby(consN,:Regime)
for byR in byRegime
  # byR = byRegime[1]
  Z = groupby(byR,:LearnP)
  Y1 = sum(
          mean([Z[2][x].^(1-eta)./(1-eta) for x in [:cq1,:cq2,:cq3,:cq4,:cq5]])
          .*Z[2][:L].*Z[2][:R]
          )
  Y2 = sum(
          mean([Z[3][x].^(1-eta)./(1-eta) for x in [:cq1,:cq2,:cq3,:cq4,:cq5]])
          .*Z[3][:L].*Z[3][:R]
          )
  function Dwel1(lambda)
    Lam = ones(size(Z[1])[1])
    Lam[Z[1][:Year].==2005] = lambda
    X = sum(
            mean([(Z[1][x].*Lam).^(1-eta)./(1-eta) for x in [:cq1,:cq2,:cq3,:cq4,:cq5]])
            .*Z[1][:L].*Z[1][:R]
            )
    X-Y1
  end
  function Dwel2(lambda)
    Lam = ones(size(Z[1])[1])
    Lam[Z[1][:Year].==2005] = lambda
    X = sum(
            mean([(Z[2][x].*Lam).^(1-eta)./(1-eta) for x in [:cq1,:cq2,:cq3,:cq4,:cq5]])
            .*Z[2][:L].*Z[2][:R]
            )
    X-Y2
  end
  resulN[indi,1] = (fzero(Dwel1,1)-1)*100
  resulN[indi,2] = (fzero(Dwel2,1)-1)*100
  indi+=1
end

#the DICE runs
resulD = zeros(4,2)
consD[:R] = (1/(1+rho)).^((consD[:Year]-2005)/10)
indi=1
byRegime = groupby(consD,:Regime)
for byR in byRegime
  # byR = byRegime[1]
  Z = groupby(byR,:LearnP)
  Y1 =  sum(
            by(Z[2],[:Year,:State],
              d->mean(d[:c].*d[:L]).^(1-eta)./(1-eta).*d[:R]
              )[:x1]
            )
  Y2 = sum(
            by(Z[3],[:Year,:State],
              d->mean(d[:c].*d[:L]).^(1-eta)./(1-eta).*d[:R]
              )[:x1]
            )
  function Dwel1(lambda)
    Xdis =  by(Z[1],[:Year,:State],
              d->mean(d[:c].*d[:L]).^(1-eta)./(1-eta).*d[:R]
              )
    Xdis[Xdis[:Year].==2005,:x1]*=lambda
    X = sum(Xdis[:x1])
    X-Y1
  end
  function Dwel2(lambda)
    Xdis =  by(Z[2],[:Year,:State],
              d->mean(d[:c].*d[:L]).^(1-eta)./(1-eta).*d[:R]
              )
    Xdis[Xdis[:Year].==2005,:x1]*=lambda
    X = sum(Xdis[:x1])
    X-Y2
  end
  resulD[indi,1] = (fzero(Dwel1,1.001)-1)*100
  resulD[indi,2] = (fzero(Dwel2,1.001)-1)*100
  indi+=1
end

learn[:Value] = [resulN[:];resulD[:]]

writetable("$(pwd())/Outputs/valueOfLearning/consD.csv", consD)
writetable("$(pwd())/Outputs/valueOfLearning/consN.csv", consN)
writetable("$(pwd())/Outputs/valueOfLearning/valueOfLearning.csv",learn)
