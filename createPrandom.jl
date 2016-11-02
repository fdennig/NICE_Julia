######################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Run Function_defitions.jl first!

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
######################################################################################################################################################

# After running that, we can begin this file...

######################################################################################################################################################
# Generate random draws of chosen parameters
#######################################################################################################################################################
# Creates nsample random draws of the parameters that we allow to vary randomly:

# Initial TFP growth rate (12) - gy0
# Initial decarbonization rate (12) - sighisT
# Atmosphere to upper ocean transfer coefficient (1) - TrM12
# Climate Sensitivity (1) - xi1
# Coefficient on T^7 in damage function (1) - psi7
# Initial world backstop price (1) - pw
# Elasticity of income wrt. damage (1) - ee

# POTENTIAL OTHER PARAMETERS (all scalar):
# du - backstop rate of decline before tau
# dd - backstop rate of decline after tau
# delsig - growth decline rate
# delA - decline in TFP growth in USA
# Crate - rate of convergence per decade
# Cratio - convergence ratio (in all regions except USA)

# Load the dparam_i.mat file to obtain the raw data

if user == "francis"
  folderData = "/Users/francis/Dropbox/ARBEIT/aRESEARCH/NICE_Julia"
elseif user == "joshua"
  folderData = "/Users/joshuabernstein/Dropbox"
elseif user == "marc"
  folderData = "/Users/mfleur/Dropbox/RICE_for_Simon (1)/Julia"
else error("wrong user")
end

dparam_i = load("$folderData/NICE_Julia/Data/dparam_i.jld")
# to call parameter values, use the syntax variable = dparam_i["variable"][2], except for q and tol where the '[2]' should be dropped
# First, get means and standard deviations for random draws.
# M for mean, sd for standard deviation
# sd small or large
if sd == "small"
  # TFP
  gy0M = dparam_i["gy0"][2]'
  gy0sd = ones(12)*0.004
  # Decarbonization
  sighisTM = dparam_i["sighisT"][2]'
  sighisTsd = ones(12)*0.004
  # World Backstop Price
  pwM = dparam_i["pw"][2]*1000
  pwsd = 68
  # Atmosphere to upper ocean transfer coefficient
  TrM12M = dparam_i["TrM"][2][1,2]/100
  TrM12sd = 0.01079
  # Climate sensitivity
  xi1M = 3.2 # hardcoded originally, changed from 3.8 to 3.2 to correspond to Nordhaus
  xi1sd = 0.3912
  # Coefficient on T^7 in damage function
  psi7M = 0.082 # hardcoded originally
  psi7sd = 0.028
  # Elasticity of income wrt. damage
  eeM = 0
elseif sd == "large" # 10 times larger than "small"
  # TFP
  gy0M = dparam_i["gy0"][2]'
  gy0sd = 10*ones(12)*0.004
  # Decarbonization
  sighisTM = dparam_i["sighisT"][2]'
  sighisTsd = 10*ones(12)*0.004
  # World Backstop Price
  pwM = dparam_i["pw"][2]*1000
  pwsd = 68
  # Atmosphere to upper ocean transfer coefficient
  TrM12M = dparam_i["TrM"][2][1,2]/100
  TrM12sd = 10*0.01079
  # Climate sensitivity
  xi1M = 3.2 # hardcoded originally, changed from 3.8 to 3.2 to correspond to Nordhaus
  xi1sd = 10*0.3912
  # Coefficient on T^7 in damage function
  psi7M = 0.082 # hardcoded originally
  psi7sd = 10*0.028
  # Elasticity of income wrt. damage
  eeM = 0
else
  # TFP
  gy0M = dparam_i["gy0"][2]'
  gy0sd = 10*ones(12)*0.004
  # Decarbonization
  sighisTM = dparam_i["sighisT"][2]'
  sighisTsd = 10*ones(12)*0.004
  # World Backstop Price
  pwM = dparam_i["pw"][2]*1000
  pwsd = 68
  # Atmosphere to upper ocean transfer coefficient
  TrM12M = dparam_i["TrM"][2][1,2]/100
  TrM12sd = 10*0.01079
  # Climate sensitivity
  xi1M = 3.2 # hardcoded originally, changed from 3.8 to 3.2 to correspond to Nordhaus
  xi1sd = 10*0.3912
  # Coefficient on T^7 in damage function
  psi7M = 0.082 # hardcoded originally
  psi7sd = 10*0.028
  # Elasticity of income wrt. damage
  eeM = 0
end


# Define Deep as the type object that will hold all the random parameter draws
type Deep
  gy0
  sighisT
  TrM12
  xi1
  psi7
  pw
  ee
end

# GENERAL RANDOMIZATION CODE FOR THE 29(?) VARIABLES WE WANT TO RANDOMIZE

using Distributions # required for some regime selections

if regime_select == 0
  # we just use the means for each sample draw
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M zeros(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM]

elseif regime_select == 1
    # total randomization
    z = zeros(nsample,29) # temp object to hold the random draws

    # Initial TFP growth rate (12, by region) - gy0 - NO RANDOMIZATION
    d_gy0 = MvNormal(vec(gy0M),diagm(gy0sd))
    z[:,1:12] = repmat(gy0M',nsample) # min(max(rand(d_gy0,nsample)',0),0.1)

    # Initial decarbonization rate (12) - sighisT
    d_sighisT = MvNormal(vec(sighisTM),diagm(sighisTsd))
    z[:,13:24] = rand(d_sighisT,nsample)'

    # Atmosphere to upper ocean transfer coefficient - TrM12 - normal distribution truncated on [0,1] - perhaps Beta is better, need to calculate parameters
    d_TrM12 = Beta((TrM12M/(1-TrM12M)),1) # Normal(TrM12M,TrM12sd)
    z[:,25] = min(max(rand(d_TrM12,nsample)',0),1)

    # Climate Sensitivity - xi1 - normal distribution, truncated at 0 !!!!!!!!!!!!!!!
    d_xi1 = Normal(xi1M,xi1sd)
    z[:,26] = max(rand(d_xi1,nsample),0)

    # Coefficient on T^7 in damage function - psi7 - normal distribution, truncated on [0,∞)
    d_psi7 = Normal(psi7M,psi7sd)
    z[:,27] = max(rand(d_psi7,nsample),0)

    # Initial world backstop price - pw - normal distribution truncated on [0.01,∞)
    d_pw = Normal(pwM,pwsd)
    z[:,28] = max(rand(d_pw,nsample),10)./1000

    # Elasticity of income wrt. damage - ee - uniform distribution on [-1.1]
    d_ee = Uniform(-1,1)
    z[:,29] = rand(d_ee,nsample)

elseif regime_select == 2
  # define preset regime: High initial TFP growth vs. Low initial TFP growth
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes

  # Initial TFP growth rate (12, by region) - gy0 - plus/minus 3 sd
  z[:,1:12] = [gy0M' + gy0sd'.*3; gy0M' - gy0sd'.*3]

elseif regime_select == 3
  # define preset regime: High initial decarbonization rate vs. Low initial decarbonization rate
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes

  # Initial decarbonization rate (12) - sighisT - plus/minus 3 sd
  z[:,13:24] = [sighisTM' + sighisTsd'.*3; sighisTM' - sighisTsd'.*3]

elseif regime_select == 4
  # define preset regime: High elasticity of income wrt damage vs. Low elasticity of income wrt damage
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes

  # Elasticity of income wrt damage - ee - 0.8 and -0.8
  z[:,29] = [0.8,-0.8]

elseif regime_select == 5
  # define preset regime: High climate sensitivity vs. Low climate sensitivity
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes

  # Climate sensitivity - xi1 - plus/minus 3 sd
  z[:,26] = [xi1M + xi1sd*3; xi1M - xi1sd*3]

elseif regime_select == 6
  # define preset regime: High atmosphere to upper ocean transfer coefficient vs. Low atmosphere to upper ocean transfer coefficient
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes

  # Atmosphere to upper ocean transfer coefficient - plus/minus 3 sd
  z[:,25] = [TrM12M + TrM12sd*3; TrM12M - TrM12sd*3]

elseif regime_select == 7
  # define preset regime: High initial world backstop price vs. Low initial world backstop price
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes

  # Initial world backstop price - pw - plus/minus 3 sd
  z[:,28] = [pwM + pwsd*3; pwM - pwsd*3]./1000

elseif regime_select == 8
  # define preset regime: High T7 coefficient vs. Low T7 coefficient
  nsample = 2
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes

  # T7 coefficient - psi7 - plus 3sd/0
  z[:,27] = [0.318; 0] # HARDCODING REPLACES: [psi7M + psi7sd*6,max(psi7M - psi7sd*3,0)]

elseif regime_select == 9
  # define preset regime: Deciles of TFP (all else fixed at means)
  nsample = 10 # for deciles
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes
  # Change gy0 to go from high to low in deciles
  decs = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05] # midpoints of each decile
  # d_gy0 = MvNormal(vec(gy0M),diagm(gy0sd)) # generates normal distribution
  QQ = zeros(12,10)
  for i = 1:12
    QQ[i,:] = quantile(Normal(gy0M[i],gy0sd[i]),decs)
  end
  z[:,1:12] = QQ'

elseif regime_select == 10
  # define preset regime: Deciles of decarb rates (all else fixed at means)
  nsample = 10 # for deciles
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes
  # Change sighisT to go from high to low in deciles
  decs = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05] # midpoints of each decile
  # d_sighisT = MvNormal(vec(sighisTM),diagm(sighisTsd)) # generates normal distribution
  QQ = zeros(12,10)
  for i = 1:12
    QQ[i,:] = quantile(Normal(sighisTM[i],sighisTsd[i]),decs)
  end
  z[:,13:24] = QQ'

elseif regime_select == 11
  # define preset regime: Deciles - High TFP and High decarb vs Low TFP and Low decarb
  nsample = 10 # for deciles
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes
  # Change gy0 and sighisT to go from high to low in deciles
  decs = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05] # midpoints of each decile
  QQ = zeros(24,10)
  for i = 1:12
    QQ[i,:] = quantile(Normal(gy0M[i],gy0sd[i]),decs)
  end
  for i = 13:24
    QQ[i,:] = quantile(Normal(sighisTM[i-12],sighisTsd[i-12]),decs)
  end
  z[:,1:24] = QQ'

elseif regime_select == 12
  # define preset regime: Deciles - High TFP and Low decarb vs. Low TFP and High decarb
  nsample = 10 # for deciles
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes
  # Change gy0 and sighisT to go from high to low in deciles
  decs_gy = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05] # midpoints of each decile for gy0 (high to low)
  decs_sig = [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95] # midpoints of each decile for gy0 (low to high)
  QQ = zeros(24,10)
  for i = 1:12
    QQ[i,:] = quantile(Normal(gy0M[i],gy0sd[i]),decs_gy)
  end
  for i = 13:24
    QQ[i,:] = quantile(Normal(sighisTM[i-12],sighisTsd[i-12]),decs_sig)
  end
  z[:,1:24] = QQ'

elseif regime_select == 13
  # define preset regime: Deciles of elasticity of income wrt damage
  nsample = 10 # for deciles
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes
  # Change ee to go from high to low in deciles
  decs = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05] # midpoints of each decile
  QQ = zeros(1,10)
  QQ[1,:] = [0.9 0.7 0.5 0.3 0.1 -0.1 -0.3 -0.5 -0.7 -0.9] # quantile(Uniform(-1,1),decs)
  z[:,29] = QQ'

elseif regime_select == 14
  # define preset regime: Deciles - High TFP and High ee vs. Low TFP and Low ee
  nsample = 10 # for deciles
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes
  # Change gy0 and sighisT to go from high to low in deciles
  decs = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05] # midpoints of each decile (high to low)
  QQ = zeros(13,10)
  for i = 1:12
    QQ[i,:] = quantile(Normal(gy0M[i],gy0sd[i]),decs)
  end
  for i = 13
    QQ[i,:] = [0.9 0.7 0.5 0.3 0.1 -0.1 -0.3 -0.5 -0.7 -0.9] # quantile(Uniform(-1,1),decs)
  end
  z[:,1:12] = QQ[1:12,:]'
  z[:,29] = QQ[13,:]'

elseif regime_select == 15
  # define preset regime: Deciles - High TFP and Low ee vs. Low TFP and High ee
  nsample = 10 # for deciles
  z = zeros(nsample,29) # temp object to hold the random draws
  z = [repmat(gy0M',nsample) repmat(sighisTM',nsample) ones(nsample).*TrM12M ones(nsample).*xi1M ones(nsample).*psi7M ones(nsample).*(pwM/1000) ones(nsample).*eeM] # fills up with means before making changes
  # Change gy0 and sighisT to go from high to low in deciles
  decs_gy = [0.95 0.85 0.75 0.65 0.55 0.45 0.35 0.25 0.15 0.05] # midpoints of each decile for gy0 (high to low)
  decs_ee = [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95] # midpoints of each decile for gy0 (low to high)
  QQ = zeros(13,10)
  for i = 1:12
    QQ[i,:] = quantile(Normal(gy0M[i],gy0sd[i]),decs_gy)
  end
  for i = 13
    QQ[i,:] = [-0.9 -0.7 -0.5 -0.3 -0.1 0.1 0.3 0.5 0.7 0.9] # quantile(Uniform(-1,1),decs_ee)
  end
  z[:,1:12] = QQ[1:12,:]'
  z[:,29] = QQ[13,:]'


else
  z = zeros(nsample,29)
  error("Select a regime!")

end

DEEPrandP = Deep(z[:,1:12],z[:,13:24],z[:,25].*100,z[:,26],z[:,27],z[:,28],z[:,29])

# END OF SECTION

# ------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------

##################################################################################################################################################################
# DEEPrandP can now be used along with certainPARAMETERS to generate the full arrays of exogenous parameters.
##################################################################################################################################################################

# Build the full array of exogenous parameters.

# Load the certainPARAMETERS.mat file

certainPARAMETERS = load("$folderData/NICE_Julia/Data/certainPARAMETERS.jld") # Adjust for User: JOSHUA

# to call parameter values, use the syntax 'variable = certainPARAMETERS["variable"][2]'

# Backstop price, pb
#  requires following parameters:
#  Th            inital to final backstop price ratio
#  RL            region to world backstop price ratio
#  du            rate of decline before tau
#  dd            rate of decline after tau
#  tau           period in which rate of decline changes
#  p0 = RL*pw    initial vector of backstop prices    1000 2005 USD per tC

# (for now, manually pull the certain parameter values from certainPARAMETERS.mat file)
Th = certainPARAMETERS["Th"][2]
if backstop_same == "Y"
  RL = ones(1,12)
elseif isdefined(:backstop_same) == false
  RL = certainPARAMETERS["RL"][2]
else RL = certainPARAMETERS["RL"][2]
end
du = certainPARAMETERS["du"][2] # could be randomized
dd = certainPARAMETERS["dd"][2] # could be randomized
tau = certainPARAMETERS["tau"][2]
# get pw from DEEPrandP
pw = DEEPrandP.pw
# build time series for backstop using assumed functional form:
pb = backstop(Th,RL,pw,du,dd,tau).*1000 # multiply by 1000 to get correct units

# Emissions to output ratio, sigma
# requires following certain parameters
    #   gT      trend growth rate
    #   delsig  growth decline rate
    #   adj15   2015 adjustment factor
    #   Y0      2005 output         trillions 2005 USD  1xI
    #   E0      2005 emissions      Mtons of CO2 equivalent 1xI
# (for now, manually pull the certain parameter values from certainPARAMETERS.mat file)
gT = certainPARAMETERS["gT"][2]
delsig = certainPARAMETERS["delsig"][2] # could be randomized
adj15 = certainPARAMETERS["adj15"][2]
Y0 = certainPARAMETERS["Y0"][2]
E0 = certainPARAMETERS["E0"][2]

# requires following random parameters
sighisT = DEEPrandP.sighisT

# build time series for sigma using assumed functional form
sigma = sig(gT,delsig,sighisT,adj15,Y0,E0)

# Multiplicative parameter in the abatement cost function, th1
# requires following certain parameters
# th2 - exponent in the abatement cost function
th2 = certainPARAMETERS["th2"][2]

# build time series for th1 using th2, sigma and pb
th1 = (pb.*sigma)/(1000*th2) # NOTE I AM NOW DIVIDING BY 1000 TO UNDO THE EARLIER RESCALING OF pb!

# Population path for all 12 regions, L
# requires following certain parameters
# Pop0 - population in 2005
# poprates - exogenous population growth rates for the first 30 periods (30 x 12 array)
Pop0 = certainPARAMETERS["Pop0"][2]
poprates = certainPARAMETERS["poprates"][2]

# build time series for L using Pop0 and poprates
L = population(Pop0,poprates)./1000 # divide by 1000 to get correct units

# Exogenous forcing (from other GHGs), Fex
# requires certain parameters
# Fex2000 - forcings in 2000
# Fex 2100 - forcings in 2100
Fex2000 = certainPARAMETERS["Fex2000"][2]
Fex2100 = certainPARAMETERS["Fex2100"][2]

# build time series for Fex using Fex2000 and Fex2100
Fex = forcingEx(Fex2000,Fex2100)

# TFP
# requires following certain parameters
# A0 - initial TFP in each region
# tgl - long run TFP growth in USA
# delA - decline in TFP growth in USA
# gamma - elasticity of capital in production
# Crate - rate of convergence per decade
# Cratio - convergence ratio (in all regions except USA)
# y0 - initial per capital consumption in each region
A0 = certainPARAMETERS["A0"][2]
tgl = certainPARAMETERS["tgl"][2]
delA = certainPARAMETERS["delA"][2] # could be randomized
gamma = certainPARAMETERS["gamm"][2]
Crate = certainPARAMETERS["Crate"][2] # could be randomized
Cratio = certainPARAMETERS["Cratio"][2] # could be randomized
y0 = certainPARAMETERS["y0"][2]

# requires the following random parameters
gy0 = DEEPrandP.gy0

# build time series for TFP
tfp = tfactorp(A0,gy0,tgl,delA,gamma,Crate,Cratio,y0)

# Emissions from land use change, EL
# requires following certain parameters
# EL0 - initial emissions due to land use change in each region
# delL - rate of decline of these
EL0 = certainPARAMETERS["EL0"][2]
delL = certainPARAMETERS["delL"][2]

# build time series for EL
EL = landuse(EL0,delL)

# Temperature forcing and temperature flow parameters, xi (nsample of them!)
# requires following certain parameters
# xi2 - 6 right elements of the vector
xi2 = certainPARAMETERS["xi"][2]

# requires the random parameter xi1 (climate sensitivity)
xi1 = DEEPrandP.xi1

# build
xi = [repmat(xi2[:,1:2],nsample) xi1 repmat(xi2[:,4:7],nsample)]

# Transition matrix for temperature flow, TrT - there are nsample of them!
# function of xi
TrT = zeros(nsample,2,2)
for i = 1:nsample
  TrT[i,:,:] = [-xi[i,2]*(xi[i,1]/xi[i,3]+xi[i,4]) xi[i,2]*xi[i,4] ; xi[i,5] -xi[i,5]]' + eye(2)
end

# Transition matrix for carbon flow, TrM (there will be nsample of them!)
# requires the following certain parameters
# TrML - lower two thirds of matrix
TrML = [0.0470 0.9480 0.0050;0 0.0008 0.9992].*100
# certainPARAMETERS["TrML"][2]


# requires the random parameter TrM12
TrM12 = DEEPrandP.TrM12'

# build TrM
TrM_ = zeros(3,3,nsample) # we have nsample 3x3 matrices (wrong order to check that TrM_ fills correctly)
for i = 1:nsample
  TrM_[:,:,i] = [100 - TrM12[i] TrM12[i] 0;TrML]
end

TrM = permutedims(TrM_,[3 1 2])./100 # divide by 100 to get in correct units

# Damage parameters, psi (nsample sets now)
# requires the following certain parameters
# psi1 - from Nordhaus (5x12 matrix)
psi1 = certainPARAMETERS["psi1"][2]

# requires the random parameter psi7
psi7 = DEEPrandP.psi7

# build psi
psi_ = zeros(3,12,nsample) # note order of dimensions to check easily
for i = 1:nsample
  psi_[1:2,:,i] = psi1[1:2,:]
  psi_[3,:,i] = repmat(psi7,1,12)[i,:]
end

psi = permutedims(psi_,[3 1 2])

# END OF SECTION

# --------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------

#######################################################################################################################################################################################
# We now have the full set of parameter arrays over time, regions, and randomization where applicable.
# We have avoided using a loop to construct the arrays for each randomization; instead we have 3 dimensional arrays, where the first dimension
# indexes the number of the random draw.
# Now build a final object which contains all the arrays together
######################################################################################################################################################################################

# Define remaining parameters
rho = certainPARAMETERS["rho"][2]
delta = certainPARAMETERS["delta"][2]
eta = certainPARAMETERS["eta"][2]
T0 = certainPARAMETERS["T0"][2]
T1 = certainPARAMETERS["T1"][2]
M0 = certainPARAMETERS["M0"][2]
M1 = certainPARAMETERS["M1"][2]
K0 = certainPARAMETERS["K0"][2]
R = certainPARAMETERS["R"][2]
# q = quintile distributions
q = dparam_i["q"]
# tol = minimum consumption
tol = dparam_i["tol"]
# d = damage distribution
d = zeros(nsample,5,12)
for i = 1:nsample
  d[i,:,:] = elasticity2attribution(DEEPrandP.ee[i],q)
end

# NEED TO FIGURE OUT HOW TO CLOSE THE .jld FILES WE HAVE OPEN!!

# Define para as the first four parameters in P - [rho, delta, eta, gamma]
para = [rho,delta,eta,gamma]'
# To assist with coding elsewhere, now build an object, PP, that contains a P object for each random draw

immutable PP_
  para # 1x4 vector, constant across nsample, regions, time
  L # TxI array
  A # TxI array
  sigma # TxI array
  th1 # nTxI array
  th2 # scalar (constant)
  pb # TxI array
  EL # TxI array
  Fex # 1xT array
  TrM # 3x3 array
  xi # 1x7 array
  TrT # 2x2 array
  psi # 3xI array
  T0 # 1x2 array (constant)
  T1 # 1x2 array (constant)
  M0 # 1x3 array (constant)
  M1 #1x3 array (constant)
  K0 # 1xI array
  E0 # 1x12 vector with 2005 emissions
  R # 1xT array
  q # 5x12 array (constant)
  d # 5x12 array
  tol # scalar (constant)
end

PP = Array(PP_,nsample)

for i=1:nsample
  PP[i] = PP_(para,L[i,:,:]',tfp[i,:,:]',sigma[i,:,:]',th1[i,:,:]',th2,pb[i,:,:]',EL',Fex,TrM[i,:,:],xi[i,:]',TrT[i,:,:],psi[i,:,:],T0,T1,M0,M1,K0,E0,R,q,d[i,:,:],tol)
  # note the transposes so that the relevant matrices are now in TxI format for easy transfer to following functions/code
end

# So calling PP[i]."variable" will call the relevant variable array/scalar for the ith random draw
# END OF SECTION

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
