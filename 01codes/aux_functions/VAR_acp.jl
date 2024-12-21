# function to estimate a Bayesian Vector Autoregressive Model using the 
# asymmetric conjugate prior put forward in Chan (2022)

# INPUTS:
# - data_train:        capTxM matrix of training data
# - p:                 scalar, number of lags 
# - cons:              boolean, indicate regression intercept?
# - nsave:             scalar, number of saved MCMC interations
# - nburn:             scalar, number of MCMC iterations discarded as burnin
# - FC:                boolean, whether to do forecast evaluation (requires data_test)
# - data_test:         FHORxM matrix-of realized observations

function VAR_acp(;
    data_train,
    p = 1, cons = true,
    nsave, nburn,
    FC = false, data_test = nothing)

    Y = data_train[p+1:end,:]
    X = mlag(data_train, p)[p+1:end,:]
    if cons
        X = hcat(X, ones(size(X,1)))
    end

    # get dimensions
    global capT,M = size(Y)
    global k = M*p
    global K = size(X,2)

    # set up sampler
    ntot = nburn + nsave

    # SPECIFY PRIORS
    # run AR(p) model of Y
    sig2 = ones(M)
    for i in 1:M
        # p = 2
        Ylag = DataFrame(
            Y = Y[(p+1):capT, i],
            Ylag1 = Y[p:(capT-1), i],
            # Ylag2 = Y[p-1:(capT-2), i],
        )
        # Fit the linear model
        local model = lm(@formula(Y ~ Ylag1), Ylag)

        # Extract the residual standard error
        sig2[i] = 1/(capT-p-1)*sum(residuals(model).^2)
    end

    S0 = sig2./2
    nu0 = Float64.((1:M)./2 .+1)

    # prior mean on VAR coeffs
    A0own = fill(.95, M)

    kappa_sh0 = M/2
    kappa_sc0 = log(M)

    # SPECIFY STARTING VALUES
    global kappa_draw = fill(10.0, 2)

    ML_draw = fill(0.0, M)
    ML_prop = fill(0.0, M)

    for i in 1:M
        acp_mom = acp_moments_eq(;Y=Y, X=X, i= i, sig2 = sig2, A0own = A0own,
                kappa_draw = kappa_draw, nu0 = nu0, S0 = S0)
        
        ML_draw[i] = acp_mom["ML_i"]
    end

    kappa_step = fill(1/M, 2)
    global kappa_count = 0

    B0_draw = LowerTriangular(Matrix{Float64}(undef, M, M))
    B0_draw .= Diagonal(ones(M))
    B_draw = Matrix{Float64}(undef, K, M)

    D_draw = ones(M)

    # INITIALIZE STORAGE OBJECTS
    A_store = Array{Union{Missing, Float64}}(missing, K, M, nsave)
    Lik_store = Array{Union{Missing, Float64}}(missing, ntot)
    kappa_store = Array{Union{Missing, Float64}}(missing, 2, nsave)
    Sigma_store = Array{Union{Missing, Float64}}(missing, M, M, nsave)
    Yhat_store = Array{Union{Missing, Float64}}(missing, capT, M, nsave)

    # RUN GIBBS SAMPLER
    for irep in 1:ntot
        # SAMPLE PRIOR VARIANCE MARGINALLY
        # propose new value
        global kappa_prop = rand.(LogNormal.(log.(kappa_draw)-.5*kappa_step.^2, kappa_step))
        # offsetting.(kappa_prop)
        # evaluate likelihood
        global ML_prop = fill(0.0, M)
        for i in 1:M
            ML_prop[i] = acp_moments_eq(;Y=Y, X=X, i= i, sig2 = sig2, A0own = A0own,
                kappa_draw = kappa_prop, nu0 = nu0, S0 = S0)["ML_i"]
        end
        # compute posterior ratio

        # proposal
        num_kappa = sum(ML_prop) +
            sum(logpdf.(InverseGamma(kappa_sh0, kappa_sc0), kappa_prop)) +
            sum(logpdf.(LogNormal.(log.(kappa_prop)-.5*kappa_step.^2, kappa_step), kappa_draw))
        # curent draw
        denom_kappa = sum(ML_draw) +
            sum(logpdf.(InverseGamma(kappa_sh0, kappa_sc0), kappa_draw)) +
            sum(logpdf.(LogNormal.(log.(kappa_draw)-.5*kappa_step.^2, kappa_step), kappa_prop))
        if num_kappa - denom_kappa > log(rand(Uniform(0,1)))
            global kappa_draw = kappa_prop
            global kappa_count += 1
        end
        # if(irep < (nburn*.5)){
        #     if(kappa.count/irep > .4) kappa.step <- kappa.step*1.01
        #     if(kappa.count/irep < .2) kappa.step <- kappa.step*.99
        #   }
        if irep < .5*nburn 
            if kappa_count/irep > .4
                kappa_step .*= 1.01
            end
            if kappa_count/irep < .2
                kappa_step .*= .99
            end
        end
        
        # Draw VAR coeffs, error variance
        for i in 1:M
            acp_mom = acp_moments_eq(;Y=Y, X=X, i= i, sig2 = sig2, A0own = A0own,
                kappa_draw = kappa_draw, nu0 = nu0, S0 = S0)
            # extract posterior moments
            B_M = acp_mom["B_M"]
            B_V = acp_mom["B_V"]
            S = acp_mom["S"]
            nu = acp_mom["nu"]
            ML_draw[i] = acp_mom["ML_i"]

            # DRAW ERROR VARIANCE FROM IG DISTRIBUTION
            D_draw[i] = rand(InverseGamma(nu, S[1])) # Julia
        
            # DRAW VAR COEFFS
            theta_i = B_M + transpose(cholesky(B_V).U)*rand(Normal(0,sqrt(D_draw[i])),K+i-1)
            # update recursive form coeffs
            B0_draw[(i),(1:(i-1))] = theta_i[1:(i-1)]
            B_draw[:,i] = theta_i[i:(K+i-1)]
        end
        
        # POST-PROCESSING
        # BRING INTO REDUCED FORM
        B0inv = inv(B0_draw)
        A_draw = B_draw*transpose(B0inv)
        
        Acm = get_companion(A_draw, M, p)["Acm"]
        stable = maximum(abs.(real.(eigen(Acm).values))) < 1
        
        global Sigma_draw = Symmetric(B0inv*Diagonal(D_draw)*transpose(B0inv))
        
        # Initialize likelihood storage
        Lik = zeros(capT)
        
        Yhat = X*A_draw
        
        # Loop through each time step
        Lik_t = zeros(M)
        for t in 1:capT
            try
                # Compute log-density of multivariate normal
                Lik_t = logpdf(MvNormal(Yhat[t,:], Sigma_draw), Y[t, :])
            catch e
                if isa(e, PosDefException)
                    # Add small variance to Sigma_draw if it is not positive definite
                    Sigma_draw += I * 1e-8  # Diagonal matrix adjustment
                    Lik_t = logpdf(MvNormal(Yhat[t,:], Sigma_draw), Y[t,:])
                else
                    rethrow(e)  # Rethrow unexpected errors
                end
            end
            Lik[t] = Lik_t
        end
        
        # Store the total log-likelihood for the current iteration
        Lik_store[irep] = sum(Lik)
        
        if irep > nburn
            A_store[:,:,irep-nburn] = A_draw
            kappa_store[:,irep-nburn] = kappa_draw
            Sigma_store[:,:,irep-nburn] = Sigma_draw
            Yhat_store[:,:,irep-nburn] = Yhat
        end
        if irep%100 == 0
            println("Iteration $irep / $ntot")
        end
    end

    # EVALUATE CONVERGENCE
    display(plot(Lik_store))

    # EVALUATE MODEL FIT
    pquant = [.025, .5, .975]
    Yhat_post = mapslices(x -> quantile(x, pquant), Yhat_store; dims=(3,))
    
    plot(Y[:,1])
    plot!(Yhat_post[:,1,:2], col = "red")
    display(plot!(Shape(vcat(1:capT,reverse(1:capT)),
        vcat(Yhat_post[:,1,3],reverse(Yhat_post[:,1,1]))),
        color=:red, alpha=.33, legend=false, border=:none, xlabel="Time", ylabel="Value"))

    
    # PLOT POSTERIOR DISTRIBUTIONS
    # plot posterior distributions
for i in 1:2
    histogram(kappa_store[i,:])
    display(title!("Posterior of κ_$i"))
end

    retdict = Dict(
        "A_store" => A_store,
        "Sigma_store" => Sigma_store, 
        "kappa_store" => kappa_store,
        "Yhat_store" => Yhat_store
        )
end