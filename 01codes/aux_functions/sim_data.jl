
# cd("/home/tobias/Dropbox/julia/lecture-julia.notebooks/")

# packages
using Distributions
using LinearAlgebra
using Missings
using Plots
using Random

# source files
include("get_companion.jl")

### for debugging purposes only:
# M = 50
# capT = 200
# burn = 100
# cons = false
# SV = false
# p = 1
# rseed = 5020

function sim_data(;
    M=10, capT=200, burn = 100,
    p=1, cons = true, SV = false, rseed = 5020,
    doplots = true
    )
    # set pseudo-random seed
    if @isdefined rseed 
        Random.seed!(rseed)
    end
    
    # get dimensions
    k = M*p
    K = k + cons
    
    # GENERATE PARAMETERS
    # VAR parameters
    A = fill(0.0, K, M)
    for i in 1:M
        A[i,i] = .95
    end
    # A[diagind(A)] .= 0.95
    global unstable = true
    global counter = 0
    global sparseshare = 0.1
    
    # stochastic component of A
    while unstable && counter < 1000
        counter += 1 
        # draw elements of A from MvNormal centered around random walk
        Astoch = A[1:k,:]
        for i in 1:M
            Astoch[:,i] = rand(MvNormal(Astoch[:,i],Diagonal(fill(0.1,k))),1)
        end
        if counter > 50
            # find number of off-diagonal elements to sparsify
            sparsify = Int(round(sparseshare*(M*k-k)))
            # Randomly select linear indices of non-diagonal elements to set to zero in Astoch
            id_offdiag = filter(i -> i % M != i ÷ M + 1, randperm(k * M))[1:sparsify]
            # Convert linear indices to 2D indices
            rows = (id_offdiag .- 1) .÷ M .+ 1
            cols = (id_offdiag .- 1) .% M .+ 1
            # Set the selected non-diagonal elements to zero in Astoch
            for i in 1:length(id_offdiag)
                Astoch[rows[i], cols[i]] = 0
            end 
            # increase sparseshare by 1% 
            if sparseshare<0.99
                global sparseshare *= 1.01
            end
        end
    
        # gen_compMat(transpose(A), M, p)
        temp = get_companion(Astoch, M, p)
        global Acmp = temp["Acm"]
    
        # continue here: check stability of VAR COEFFS
        maxeigen_A = maximum(abs.(real(eigen(Acmp).values)))
        global unstable = maxeigen_A > 0.9
        if !unstable
            A[1:k,:] = Astoch
        end
    end
    # set intercept
    if cons
        A[k+1,:] = rand(Normal(0.0, 2.0), M)
    end 

    # VC matrix of errors
    nu = M*10
    S = Matrix(Diagonal(fill(0.1,M)))
    # long-run mean
    Σ = rand(InverseWishart(nu, S*nu))
    # time-varying part
    H = fill(0.0, capT+burn, M)
    Σ_t = fill(0.0,M,M,capT+burn)
    # only define state equation parameters when specified with SV
    if SV
        rhoSV = fill(.95,M)
        volvol = fill(0.2,M)
    end
    
    # GENERATE DATA
    Y = fill(0.0, capT+burn, M)
    ε = copy(Y)
    
    for tt in p+1:capT+burn
        if SV
            H[tt,:] = rhoSV.*H[tt-1,:] + randn(M).*sqrt.(volvol)
        end
        Σ_t[:,:,tt] = Diagonal(exp.(H[tt,:]./2))*Σ*Diagonal(exp.(H[tt,:]./2))
        ε[tt,:] = rand(MvNormal(Symmetric(Σ_t[:,:,tt])))
        if cons
            X = vcat(vec(Y[[tt-i for i in 1:p],:]),1)
        else
            X = vec(Y[[tt-i for i in 1:p],:])
        end
        Y[tt,:] = X'*A + ε[tt,:]'
    end
    
    # drop burnin-period
    Y = Y[burn+1:end,:]
    ε = ε[burn+1:end,:]
    H = H[burn+1:end,:]
    Σ_t[:,:,burn+1:end]
    
    # plot simulated data
    if doplots
        Ypl = plot()
        for i in 1:M
            plot!(Ypl, Y[:,i], label="Series $i")
        end
        display(Ypl)
    end

    # store outputs in dictionary
    retdict = Dict(
        "A" => A,
        "Σ" => Σ,
        "H" => H,
        "Σ_t" => Σ_t,
        "Y" => Y,
        "ε" => ε,
        # Conditional inclusion of SV_param based on SV
        "SV_param" => SV ? Dict(
            "rhoSV" => rhoSV,
            "volvol" => volvol
        ) : Dict()  # Use an empty Dict() if SV is false
    )
end