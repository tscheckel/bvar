# function to draw horsehoe prior parameters
# author: tobias scheckel

# INPUTS:
# - bdraw:      vector, regression parameters
# - λ:  vector, local variance component
# - τ:     scalar, global variance component
# - ν:      vector, local auxiliary variables
# - ζ:    scalar, global auxiliary variables

# OUTPUTS:
# - updated values of lambda, tau, nu, zeta
# - psi = lambda.*tau: prior variance

using Distributions

function get_hs(;
    bdraw::Vector{Float64},
    λ::Vector{Float64}, τ::Float64,
    ν::Vector{Float64}, ζ::Float64
    )
    
    k = length(bdraw)
    
    # global & local prior variance components
    for j in 1:k
        λ[j] = rand(InverseGamma(1.0, 1.0/ν[j] + bdraw[j]^2/(2.0 * τ)))
    end
    τ = rand(InverseGamma((k+1.0)/2.0, 1.0/ζ + sum(bdraw.^2 ./ λ)/2.0))
    
    # auxiliary variables
    for j in 1:k
        ν[j] = rand(InverseGamma(1.0, 1.0 + 1.0/λ[j]))
    end
    ζ = rand(InverseGamma(1.0, 1.0 + 1.0 / τ))
    
    ret = Dict(
        "psi" => (λ * τ),
        "lambda" => λ,
        "tau" => τ,
        "nu" => ν,
        "zeta" => ζ
    )
    return ret
end
