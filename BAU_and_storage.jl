###########################################################################################################################################################################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

using HDF5, JLD
# Set user name to run auxiliary files
user = "francis" # or "francis" or "marc" as the case may be

regime_select = 0
idims = 1
Tm = 32
tm = Tm-1
lm=0
backstop_same = "Y" # "N"
model = "RICE"
rho=0.015
eta = 2
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

include("$folder/NICE_Julia/Function_definitions.jl")

include("$folder/NICE_Julia/createPrandom.jl")



# Extract the two tax vectors from the optimization object
taxes_1 = zeros(tm)
taxes_2 = taxes_1
expected_welfare = tax2welfare(taxes_1, PP[1], rho, eta, Tm, model="RICE")
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
for i = 1:2
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

taxes_1 = [0;taxes_1;zeros(Tm-tm-1)]
taxes_2 = [0;taxes_2;zeros(Tm-tm-1)]
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
  jldopen("$(pwd())/Outputs/BAU/BAUparameters.jld", "w") do file
      write(file, "PP", PP[1])
  end
  jldopen("$(pwd())/Outputs/BAU/BAU.jld", "w") do file
      write(file, "BAU", res)
  end

  writetable("$(pwd())/Outputs/BAU/BAU.csv", dataP)
