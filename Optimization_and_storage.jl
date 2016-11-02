###########################################################################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Preliminaries
using HDF5, JLD
# Set user name to run auxiliary files
user = "francis" # or "francis" or "marc" as the case may be
sd="small"

# Select the regime for parameter randomization
regime_select = 0

# Define exogenous parameters

Tm = 32
# Time period we want to consider, Tm <= 60
tm = 15 # (must be an integer) length of the tax vector we want to consider - THIS WILL DIFFER FROM THE DIMENSION OF THE ACTUAL TAX VECTOR OBJECT, tax_length!
  # note that the tax vector contains 2 sets of taxes -
  # the first lm elements are common to both sets, the remainder of the vector is then split in two, one for each set of taxes (must of equal length, (tm - lm)/2)

backstop_same = "Y" # "N" is default - choose "Y" if we want all the countries to face the same backstop prices over time

rho = 0.015 # PP[1].para[1] # discount rate
eta = 1.5 # PP[1].para[3] # inequality aversion/time smoothing parameter (applied to per capita consumption by region, quintile and time period)
nu = 1.5 # risk aversion parameter (applied over random draws)

# Now execute the whole code: select all and evaluate
###########################################################################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if user == "francis"
  folder = "/Users/francis/Dropbox/ARBEIT/aRESEARCH/NICE_Julia"
elseif user == "joshua"
  folder = "/Users/joshuabernstein/Dropbox"
elseif user == "marc"
  folder = "/Users/mfleur/Dropbox/RICE_for_Simon (1)/Julia"
else error("wrong user")
end


# Run Function Definitions File
include("$folder/NICE_Julia/Function_definitions.jl")

# 3. Run Function_defitions_for_createPrandom_and_parameters2consumption.jl AND createPrandom_and_parameters2consumption.jl first to build necessary parameters!
include("$folder/NICE_Julia/createPrandom.jl") # quick way to run this!

# Optimization of welfare function using NLopt package
using NLopt
idims = Int(max(nsample/2,1)) # bifurcates the random draws into two subsets
lm = 0
count = 0 # keep track of number of iterations

# Define function to be maximized (requires special format for NLopt package)
model = "RICE"
# backstop or zero vector for first three periods
        tax15to35 = maximum(squeeze(maximum(pb,2),2),1)[2:4] # THESE ARE THE TWO OPTIONS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        tax15to35 = zeros(3) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
function welfaremax(x,grad) # x is the tax vector, grad is the gradient (unspecified here since we have no way of computing it)
  tex = [tax15to35;x]
  WW = tax2welfare(tex, PP[1], rho, eta, Tm, model="RICE")[1] #change to model="RICE" or "DICE" for different levels of aggregation
  global count
  count::Int += 1
  println("f_$count($x)")
  return WW
end

n=tm-3 # length of the maximand
# Choose algorithm (gradient free method) and dimension of tax vector, tm+1 <= n <= Tm
opt = Opt(:LN_BOBYQA, n) # algorithm and dimension of tax vector, possible (derivative-free) algorithms: LN_COBYLA, LN_BOBYQA

ub = maximum(squeeze(maximum(pb,2),2)[1:idims,:],1)[5:tm+1]

# lower bound is zero
lower_bounds!(opt, zeros(n))
upper_bounds!(opt, ub)

# Set maximization
max_objective!(opt, welfaremax)

# Set relative tolerance for the tax vector choice - vs. ftol for function value?
ftol_rel!(opt,0.00000000000005)

# Optimize! Initial guess defined above
(expected_welfare,tax_vector,ret) = optimize(opt, ub.*0.5)

# tax vectors from the 2015
taxes_1 = [tax15to35; tax_vector]
taxes_2 = taxes_1

# Create storage objects
if (model == "RICE") | (model == "DICE")
  c = Array(Float64, Tm, 12, nsample)
else
  c = Array(Float64, Tm, 12, 5, nsample)
end
K = Array(Float64, Tm, 12, nsample)
T = Array(Float64, Tm, 2, nsample)
E = Array(Float64, Tm, 12, nsample)
M = Array(Float64, Tm, 3, nsample)
mu = Array(Float64, Tm, 12, nsample)
lam = Array(Float64, Tm, 12, nsample)
D = Array(Float64, Tm, 12, nsample)
AD = Array(Float64, Tm, 12, nsample)
Y = Array(Float64, Tm, 12, nsample)
Q = Array(Float64, Tm, 12, nsample)

# Store data
for i = 1:convert(Int,nsample/2)
  if (model == "RICE") | (model == "DICE")
    c[:,:,i] = fromtax(taxes_1,PP[i],Tm)[12]
  else
    c[:,:,:,i] = fromtax(taxes_1,PP[i],Tm)[1]
  end
  K[:,:,i] = fromtax(taxes_1,PP[i],Tm)[2]
  T[:,:,i] = fromtax(taxes_1,PP[i],Tm)[3]
  E[:,:,i] = fromtax(taxes_1,PP[i],Tm)[4]
  M[:,:,i] = fromtax(taxes_1,PP[i],Tm)[5]
  mu[:,:,i] = fromtax(taxes_1,PP[i],Tm)[6]
  lam[:,:,i] = fromtax(taxes_1,PP[i],Tm)[7]
  D[:,:,i] = fromtax(taxes_1,PP[i],Tm)[8]
  AD[:,:,i] = fromtax(taxes_1,PP[i],Tm)[9]
  Y[:,:,i] = fromtax(taxes_1,PP[i],Tm)[10]
  Q[:,:,i] = fromtax(taxes_1,PP[i],Tm)[11]
end
for i = convert(Int,(nsample/2 + 1)):nsample
  if (model == "RICE") | (model == "DICE")
    c[:,:,i] = fromtax(taxes_2,PP[i],Tm)[12]
  else
    c[:,:,:,i] = fromtax(taxes_2,PP[i],Tm)[1]
  end
  K[:,:,i] = fromtax(taxes_2,PP[i],Tm)[2]
  T[:,:,i] = fromtax(taxes_2,PP[i],Tm)[3]
  E[:,:,i] = fromtax(taxes_2,PP[i],Tm)[4]
  M[:,:,i] = fromtax(taxes_2,PP[i],Tm)[5]
  mu[:,:,i] = fromtax(taxes_2,PP[i],Tm)[6]
  lam[:,:,i] = fromtax(taxes_2,PP[i],Tm)[7]
  D[:,:,i] = fromtax(taxes_2,PP[i],Tm)[8]
  AD[:,:,i] = fromtax(taxes_2,PP[i],Tm)[9]
  Y[:,:,i] = fromtax(taxes_2,PP[i],Tm)[10]
  Q[:,:,i] = fromtax(taxes_2,PP[i],Tm)[11]
end

taxes_1 = [0; taxes_1; zeros(Tm-tm-1)]
taxes_2 = taxes_1
# Region Labels
Regions = ["USA", "OECD Europe", "Japan", "Russia", "Non-Russia Eurasia", "China", "India", "Middle East", "Africa", "Latin America", "OHI", "Other non-OECD Asia"]'

# Create storage object
type Results
  regime
  nsample
  Tm
  tm
  lm
  Regions
  taxes_1
  taxes_2
  EWelfare
  c
  K
  T
  E
  M
  mu
  lam
  D
  AD
  Y
  Q
  rho
  eta
  nu
  PP
end

# create .jld of S_lm
res = Results(regime_select,nsample,Tm,tm,lm,Regions,taxes_1,taxes_2,expected_welfare,c,K,T,E,M,mu,lam,D,AD,Y,Q,rho,eta,nu,PP)
filenm = string(regime_select)
# create dataframe of period by region by state data
  using DataFrames
  # set up dataframe with periods, regions, State
  if length(size(res.c)) > 2
    dataP = DataFrame(ID = 1:(Tm*12*nsample), State = reshape(repmat(collect(1:nsample)',Tm*12),Tm*12*nsample,1)[:,1], Region = repmat(reshape(repmat(Regions,Tm),Tm*12,1),nsample)[:,1], Year = repmat(repmat(10*(0:Tm-1)+2005,12),nsample))
    # add taxes (to the correct states)
    dataP[:tax] = [repmat(repmat(res.taxes_1,12),idims);repmat(repmat(res.taxes_2,12),nsample-idims)]

    # add consumption quintiles
    if length(size(res.c)) == 4
      confield = [:cq1, :cq2, :cq3, :cq4, :cq5]
      cquintiles=reshape(permutedims(res.c,[1 2 4 3]),Tm*12*nsample,5)
      m=1
      for field in confield
        dataP[Symbol(field)] = cquintiles[:,m]
        m+=1
      end
    elseif length(size(res.c)) == 3
      cons = reshape(res.c,Tm*12*nsample,1)[:,1]
      dataP[:c] = cons
    end

    # add remaining endogenous variables
    for field in [:K,:E,:mu,:lam,:D,:Y]
      dataP[Symbol(field)] = reshape(getfield(res,field),Tm*12*nsample)
    end
    # add exogenous variables
    y = Array(Float64,Tm*12*nsample,6)
    x = Array(Float64,Tm*12,nsample)
    k=1
    for field in [:L,:A,:sigma,:th1,:pb,:EL]
      for m in 1:nsample
        x[:,m] = reshape(getfield(res.PP[m],field)[1:Tm,:],Tm*12)
      end
      y[:,k] = reshape(x,Tm*12*nsample)
      dataP[Symbol(field)] = y[:,k]
      k+=1
    end
  end
