offsetting = function(x;
    thrshld1 = 1e-8, thrshld2 = 1e8)

    x[x .< thrshld1] .= thrshld1
    x[x .> thrshld2] .= thrshld2

end