function acp_moments_eq(;
    # Y::Matrix, X::Matrix, i::Int, sig2::Vector{Float64},
    # A0own::Vector{Float64}, kappa_draw::Vector{Float64}, 
    # nu0::Vector{Float64}, S0::Vector{Float64}
    Y::Matrix, X::Matrix, i::Int, sig2::Vector{Float64},
    A0own::Vector{Float64}, kappa_draw::Vector{Float64}, 
    nu0::Vector{Float64}, S0::Vector{Float64}
    )
    # START FUNCTION ABOUT HERE
    Y_i = Y[:,i:i]
    X_i = hcat(-Y[:,1:(i-1)],X)
    
    # get dimensions
    K = size(X)[2]

    # set prior moments for eq i
    B_V0 = zeros(K+(i-1))
    B_M0 = zeros(K+(i-1))
    B_V0[1:(i-1)] = 1 ./sig2[1:(i-1)] # prior variance on contemp coeffs
    if cons
        B_V0[i-1+K] = 100 # prior variance on intercept
    end

    for plag in 1:p
        for j in 1:M
            if plag == 1 && j == i
                # print("$j")
                # print("$A0own[i]")
                B_M0[i-1+i] = A0own[i] # prior mean on own first lag
            end
            if i == j 
                B_V0[(i-1)+(plag-1)*M+i] = kappa_draw[1]/(plag^2*sig2[i]) # prior variance for own lags
            end
            if i != j 
                B_V0[(i-1)+(plag-1)*M+j] = kappa_draw[2]/(plag^2*sig2[j]) # prior variance for other lags
            end 
        end
    end
    # offset prior variance
    offsetting(B_V0)
    # prior precision
    B_P0 = Diagonal(1 ./B_V0)
    # posterior precision
    B_P = B_P0 + transpose(X_i)*X_i
    # posterior variance
    B_V = inv(Symmetric(B_P))

    if all(B_M0 .== 0)
        B_M = B_V*transpose(X)*Y
        S = S0[i] .+ transpose(Y_i)*Y_i - transpose(B_M)*B_P*B_M
    else
        B_M = B_V*(B_P0*B_M0 + transpose(X_i)*Y_i)
        S = S0[i] .+ transpose(Y_i)*Y_i .+ transpose(B_M0)*B_P0*B_M0 .- transpose(B_M)*B_P*B_M
    end
    nu = nu0[i] + capT/2

    B_V0_ldet = log(prod(B_V0)) # prod(B.V0) is faster than det(diag(B.V0))
    B_P_ldet = 2*sum(log.(diag(cholesky(B_P).U))) # see last paragraph of section 2 Chan (2022)

    termA = -(capT)/2*log(2*pi)
    termB = (-1/2)*(B_V0_ldet+B_P_ldet)
    termC = loggamma(nu) + nu0[i]*log(S0[i]) - lgamma(nu0[i]) .- nu*log(S)

    ML_i = termA + termB + termC[1,1]

    # store outputs in dictionary
    retdict = Dict(
        "B_M" => B_M,
        "B_P" => B_P,
        "B_V" => B_V,
        "S" => S,
        "nu" => nu,
        "ML_i" => ML_i
    )
end