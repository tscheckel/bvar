function gen_compMat(A::AbstractMatrix, n::Int, plag::Int)
    # Initialize Jm matrix
    Jm = zeros(n * plag, n)
    Jm[1:n, 1:n] .= I(n)  # Set the first n x n block to an identity matrix

    # Adjust A to match the dimensions needed
    A = A[1:(n * plag), :]
    
    # Initialize Cm matrix
    Cm = zeros(n * plag, n * plag)
    
    if plag == 1
        Cm .= A'
    else
        for jj in 1:(plag - 1)
            Cm[(jj * n + 1):(n * (jj + 1)), (n * (jj - 1) + 1):(jj * n)] .= I(n)
        end
    end
    
    bbtemp = A[1:(n * plag), :]
    splace = 0
    
    for pp in 1:plag
        for nn in 1:n
            Cm[nn, ((pp - 1) * n + 1):(pp * n)] .= bbtemp[(splace + 1):(splace + n), nn]'
        end
        splace += n
    end
    
    return Dict("Cm" => Cm, "Jm" => Jm)
end