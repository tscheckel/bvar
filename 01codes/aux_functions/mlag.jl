function mlag(X::AbstractMatrix, plag::Int)
    X = convert(Matrix{Float64}, X)  # Ensure X is a Float64 matrix
    Traw, N = size(X)
    Xlag = fill(NaN, Traw, plag * N)  # Initialize Xlag with NaNs
    
    for ii in 1:plag
        Xlag[(plag + 1):Traw, (N * (ii - 1) + 1):(N * ii)] = X[(plag + 1 - ii):(Traw - ii), 1:N]
    end
    
    return Xlag
end