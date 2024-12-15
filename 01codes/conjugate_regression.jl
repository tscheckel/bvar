cd("/home/tobias/Dropbox/julia/")

# packages
using Plots
using Random
using Statistics
using LinearAlgebra
using Distributions
using StatsPlots

Random.seed!(5020)
closeall()

# SIMULATE DATA
K = 5
N = 600

sigma_true = 1

X = transpose(rand(MvNormal(fill(0,K), Diagonal(fill(10,K))),N))
eps_true = transpose(rand(MvNormal(fill(0,1), Diagonal(fill(sigma_true,1))),N))
beta_true = rand(MvNormal(fill(0,K), Diagonal(fill(1,K))))

y = X*beta_true + eps_true

histogram(X)
histogram(y)

# ---- ESTIMATION ----
# SET UP SAMPLER
nsave = 2500
nburn = 2500
ntot = nburn + nsave

# PRIORS
Beta_mean0 = fill(0, K)
Beta_V0 = fill(100,K)
nu0 = 1.0
s0 = 1.0

# STARTING VALUES
beta_OLS = beta_draw = inv(transpose(X)*X)*transpose(X)*y

# STORAGE ARRAYS
global beta_store = Array{Float64}(undef,K,1,nsave)
global Σ_store = Array{Float64}(undef,1,1,nsave)

# COMPUTE OUT-OF-LOOP
nu = nu0 + N/2
s = s0 .+ (transpose(y'y) .+ transpose(Beta_mean0)*Diagonal(1 ./Beta_V0)*Beta_mean0 .- 
transpose(Beta_mean)*inv(Beta_V)*Beta_mean)/2


# RUN GIBBS SAMPLER
for irep in 1:ntot
    # draw Σ
    global Σ_draw = 1 ./ rand(Gamma(nu,1/s[1,1]),1)

    
    # sample beta
    global Beta_V = inv(Symmetric(Diagonal(1 ./ Beta_V0) + transpose(X)*X))
    global Beta_mean = Beta_V*(Diagonal(1 ./ Beta_V0)*Beta_mean0 + transpose(X)*y)

    global beta_draw = rand(MvNormal(vec(Beta_mean), Beta_V),1)


    if irep > nburn
        beta_store[:,:,irep-nburn] = beta_draw
        Σ_store[:,:,irep-nburn] = Σ_draw
    end

    if irep%100==0
        println("Iteration $irep/$ntot")
    end
end

# EVALUATE OUTPUTS
pquant = [0.025, 0.5, 0.975]
beta_post = mapslices(x -> quantile(x, pquant), beta_store; dims=(3,))
Σ_post = mapslices(x -> quantile(x, pquant), Σ_store; dims=(3,))

