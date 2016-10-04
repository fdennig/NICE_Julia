###########################################################################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Preliminaries

# Set user name to run auxiliary files
user = "francis" # or "francis" or "marc" as the case may be

# Select the regime for parameter randomization
regime_select = 2
  # 0 = no randomization (just uses means)
  # 1 = total randomization

  # 2 = High initial TFP growth vs. Low initial TFP growth
  # 3 = High initial decarbonization rate vs. Low initial decarbonization rate
  # 4 = High elasticity of income wrt damage vs. Low elasticity of income wrt damage
  # 5 = High climate sensitivity vs. Low climate sensitivity
  # 6 = High atmosphere to upper ocean transfer coefficient vs. Low atmosphere to upper ocean transfer coefficient
  # 7 = High initial world backstop price vs. Low initial world backstop price
  # 8 = High T7 coefficient vs. Low T7 coefficient

  # 9 = Deciles of TFP (all else fixed at means) (nsample = 10)
  # 10 = Deciles of decarbonization rates (all else fixed at means) (nsample = 10)
  # 11 = Deciles - High TFP and High decarb vs. Low TFP and Low decarb (technology spillovers?)
  # 12 = Deciles - High TFP and Low decarb vs. Low TFP and High decarb (substitutable tech?)
  # 13 = Deciles of elasticity of income wrt damage (ee)
  # 14 = Deciles - High TFP and High ee vs. Low TFP and Low ee
  # 15 = Deciles - High TFP and Low ee vs. Low TFP and High ee

# Define exogenous parameters

Tm::Int = 32 # Time period we want to consider, Tm <= 60
tm::Int = round(15) # (must be an integer) length of the tax vector we want to consider - THIS WILL DIFFER FROM THE DIMENSION OF THE ACTUAL TAX VECTOR OBJECT, tax_length!
  # note that the tax vector contains 2 sets of taxes -
  # the first lm elements are common to both sets, the remainder of the vector is then split in two, one for each set of taxes (must of equal length, (tm - lm)/2)

backstop_same = "Y" # "N" is default - choose "Y" if we want all the countries to face the same backstop prices over time

rho = 0.015 # PP[1].para[1] # discount rate
eta = 1.5 # PP[1].para[3] # inequality aversion/time smoothing parameter (applied to per capita consumption by region, quintile and time period)
nu = 2 # risk aversion parameter (applied over random draws)

# Now execute the whole code: select all and evaluate

###########################################################################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Set directory based on User
if user == "francis"
  folder = "/Users/francis/Dropbox/ARBEIT/aRESEARCH/NICE_Julia"
elseif user == "joshua"
  folder = "/Users/joshuabernstein/Dropbox"
elseif user == "marc"
  folder = "\Users\mfleur\Dropbox\RICE_for_Simon (1)\Julia"
else error("wrong user")
end

# Run Function Definitions File
include("$folder/NICE_Julia/Function_definitions.jl")

# 3. Run Function_defitions_for_createPrandom_and_parameters2consumption.jl AND createPrandom_and_parameters2consumption.jl first to build necessary parameters!
include("$folder/NICE Julia/createPrandom.jl") # quick way to run this!

# Optimization of welfare function using NLopt package
using NLopt
idims = max(nsample/2,1) # bifurcates the random draws into two subsets

# 1. lm = 0
lm::Int = 0
tax_length::Int = round(2*tm - lm)

count::Int = 0 # keep track of number of iterations

# Define function to be maximized (requires special format for NLopt package)

function welfaremax(x,grad) # x is the tax vector, grad is the gradient (unspecified here since we have no way of computing it)
  WW = tax2expectedwelfare(x,PP,rho,eta,nu,Tm,tm,lm,idims)[1]
  global count
  count::Int += 1
  println("f_$count($x)")
  return WW
end

# Choose algorithm (gradient free method) and dimension of tax vector, tm+1 <= n <= Tm
n = tax_length
opt = Opt(:LN_BOBYQA, n) # algorithm and dimension of tax vector, possible (derivative-free) algorithms: LN_COBYLA, LN_BOBYQA

ub_lm = maximum(squeeze(maximum(pb,2),2),1)[2:lm+1]
ub_1 = maximum(squeeze(maximum(pb,2),2)[1:idims,:],1)[lm+2:tm+1]
ub_2 = maximum(squeeze(maximum(pb,2),2)[(idims+1):nsample,:],1)[lm+2:tm+1]
# lower bound is zero
lower_bounds!(opt, zeros(n))
upper_bounds!(opt, [ub_lm, ub_1, ub_2])

# Set maximization
max_objective!(opt, welfaremax)

# Set relative tolerance for the tax vector choice - vs. ftol for function value?
ftol_rel!(opt,0.00000000000005)

# Optimize! Initial guess defined above
(expected_welfare,tax_vector,ret) = optimize(opt, [ub_lm, ub_1, ub_2].*0.5)

# Extract the two tax vectors from the optimization object
taxes_1 = tax_vector[1:tm]
taxes_2 = [tax_vector[1:lm],tax_vector[tm+1:end]]

# Create storage objects
c = Array(Float64, Tm, 12, 5, nsample)
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
for i = 1:nsample/2
  c[:,:,:,i] = fromtax(taxes_1,PP[i],Tm)[1]
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
for i = (nsample/2 + 1):nsample
  c[:,:,:,i] = fromtax(taxes_2,PP[i],Tm)[1]
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
S_0 = Results(regime_select,nsample,Tm,tm,lm,Regions,taxes_1,taxes_2,expected_welfare,c,K,T,E,M,mu,lam,D,AD,Y,Q,rho,eta,nu,PP)
filenm = string(regime_select)

# 2. lm = 4
lm::Int = 4
tax_length::Int = round(2*tm - lm)

count = 0 # keep track of number of iterations

# Define function to be maximized (requires special format for NLopt package)

function welfaremax(x,grad) # x is the tax vector, grad is the gradient (unspecified here since we have no way of computing it)
  WW = tax2expectedwelfare(x,PP,rho,eta,nu,Tm,tm,lm,idims)[1]
  global count
  count::Int += 1
  println("f_$count($x)")
  return WW
end

# Choose algorithm (gradient free method) and dimension of tax vector, tm+1 <= n <= Tm
n = tax_length
opt = Opt(:LN_BOBYQA, n) # algorithm and dimension of tax vector, possible (derivative-free) algorithms: LN_COBYLA, LN_BOBYQA

ub_lm = maximum(squeeze(maximum(pb,2),2),1)[2:lm+1]
ub_1 = maximum(squeeze(maximum(pb,2),2)[1:idims,:],1)[lm+2:tm+1]
ub_2 = maximum(squeeze(maximum(pb,2),2)[(idims+1):nsample,:],1)[lm+2:tm+1]
# lower bound is zero
lower_bounds!(opt, zeros(n))
upper_bounds!(opt, [ub_lm, ub_1, ub_2])

# Set maximization
max_objective!(opt, welfaremax)

# Set relative tolerance for the tax vector choice - vs. ftol for function value?
ftol_rel!(opt,0.00000000000005)

# Optimize! Initial guess defined above
(expected_welfare,tax_vector,ret) = optimize(opt, [ub_lm, ub_1, ub_2].*0.5)

# Extract the two tax vectors from the optimization object
taxes_1 = tax_vector[1:tm]
taxes_2 = [tax_vector[1:lm],tax_vector[tm+1:end]]

# # Create storage objects
# c = Array(Float64, Tm, 12, 5, nsample)
# K = Array(Float64, Tm, 12, nsample)
# T = Array(Float64, Tm, 2, nsample)
# E = Array(Float64, Tm, 12, nsample)
# M = Array(Float64, Tm, 3, nsample)
# mu = Array(Float64, Tm, 12, nsample)
# lam = Array(Float64, Tm, 12, nsample)
# D = Array(Float64, Tm, 12, nsample)
# AD = Array(Float64, Tm, 12, nsample)
# Y = Array(Float64, Tm, 12, nsample)
# Q = Array(Float64, Tm, 12, nsample)

# Store data
for i = 1:nsample/2
  c[:,:,:,i] = fromtax(taxes_1,PP[i],Tm)[1]
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
for i = (nsample/2 + 1):nsample
  c[:,:,:,i] = fromtax(taxes_2,PP[i],Tm)[1]
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

# create .jld of S_lm
S_4 = Results(regime_select,nsample,Tm,tm,lm,Regions,taxes_1,taxes_2,expected_welfare,c,K,T,E,M,mu,lam,D,AD,Y,Q,rho,eta,nu,PP)

# 3. lm = tm
lm::Int = tm
tax_length::Int = round(2*tm - lm)

count = 0 # keep track of number of iterations

# Define function to be maximized (requires special format for NLopt package)

function welfaremax(x,grad) # x is the tax vector, grad is the gradient (unspecified here since we have no way of computing it)
  WW = tax2expectedwelfare(x,PP,rho,eta,nu,Tm,tm,lm,idims)[1]
  global count
  count::Int += 1
  println("f_$count($x)")
  return WW
end

# Choose algorithm (gradient free method) and dimension of tax vector, tm+1 <= n <= Tm
n = tax_length
opt = Opt(:LN_BOBYQA, n) # algorithm and dimension of tax vector, possible (derivative-free) algorithms: LN_COBYLA, LN_BOBYQA

ub_lm = maximum(squeeze(maximum(pb,2),2),1)[2:lm+1]
ub_1 = maximum(squeeze(maximum(pb,2),2)[1:idims,:],1)[lm+2:tm+1]
ub_2 = maximum(squeeze(maximum(pb,2),2)[(idims+1):nsample,:],1)[lm+2:tm+1]
# lower bound is zero
lower_bounds!(opt, zeros(n))
upper_bounds!(opt, [ub_lm, ub_1, ub_2])

# Set maximization
max_objective!(opt, welfaremax)

# Set relative tolerance for the tax vector choice - vs. ftol for function value?
ftol_rel!(opt,0.00000000000005)

# Optimize! Initial guess defined above
(expected_welfare,tax_vector,ret) = optimize(opt, [ub_lm, ub_1, ub_2].*0.5)

# Extract the two tax vectors from the optimization object
taxes_1 = tax_vector[1:tm]

# # Create storage objects
# c = Array(Float64, Tm, 12, 5, nsample)
# K = Array(Float64, Tm, 12, nsample)
# T = Array(Float64, Tm, 2, nsample)
# E = Array(Float64, Tm, 12, nsample)
# M = Array(Float64, Tm, 3, nsample)
# mu = Array(Float64, Tm, 12, nsample)
# lam = Array(Float64, Tm, 12, nsample)
# D = Array(Float64, Tm, 12, nsample)
# AD = Array(Float64, Tm, 12, nsample)
# Y = Array(Float64, Tm, 12, nsample)
# Q = Array(Float64, Tm, 12, nsample)

# Store data
for i = 1:nsample/2
  c[:,:,:,i] = fromtax(taxes_1,PP[i],Tm)[1]
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
for i = (nsample/2 + 1):nsample
  c[:,:,:,i] = fromtax(taxes_2,PP[i],Tm)[1]
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

# create .jld of S_lm
S_tm = Results(regime_select,nsample,Tm,tm,lm,Regions,taxes_1,taxes_2,expected_welfare,c,K,T,E,M,mu,lam,D,AD,Y,Q,rho,eta,nu,PP)

SS = Array(Results,3)
SS = [S_0 S_4 S_tm]

using JLD
save("$folder/NICE_Julia/Outputs/SS_$filenm.jld", "SS", SS)
