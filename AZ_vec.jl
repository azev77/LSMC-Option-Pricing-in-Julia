using Statistics, LinearAlgebra, Random, BenchmarkTools, Strided
#merge scale()  & Skip first_one() & put advance() into main loop!
@inline function chebyshev_basis(x, order, n)  #chebyshev_basis(S[:, t_next], order, n)
    #first scale(x)
    xmin = minimum(x)
    xmax = maximum(x)
    a = 2. / (xmax - xmin)
    b = -0.5 * a * (xmin + xmax)
    x = @.  a * x + b
    #Second get Cheby basis.
    B = ones(n, order)
    B[:,2] = x
    for j in range(3, stop = order)
        @. @views B[:,j] = 2 * x * B[:,j - 1] - B[:,j - 2]
    end
    return B
end
#
function ridge_regression(X, Y, λ)  ##mul!(out, X, β) #AZ: don't know how
    β = (X'X + λ * I) \ (X'Y)
    return X * β
end
#
function w(a, b, c)
    a ? b : c
end
#
@inline function compute_price(Spot, σ, K, r, n, m, Δt, order)
    Random.seed!(0)
    #FILL IN S
    S = zeros(n, m + 1)
    S[:, 1] .= Spot
    for t = 1:m
        dB = sqrt(Δt) * randn(Float64, (n))
        @. @views S[:, t + 1] = S[:, t] *(1 + r * Δt + σ * dB)
    end
    #FILL IN CFL
    CFL = @.  max(0, K - S)
    #CFL = max.(0, K .- S)
    #FILL IN VALUE & CV
    discount = exp(-r * Δt)
    value = zeros(n, m)
    @strided value[:, m] = CFL[:, (m + 1)] * discount
    CV = zeros(n, m)
    @inbounds for k = 1:m -1  #1-9
        t = m - k   #9-1
        t_next = t + 1 #10-2
        @views XX = chebyshev_basis(S[:, t_next], order, n)
        @views YY = value[:, t_next]
        CV[:, t] = ridge_regression(XX, YY, 100)
        @strided value[:, t] = discount * w.(CFL[:, t_next] .> CV[:, t], CFL[:, t_next], value[:, t_next])
    end
    #Compute Price. Exercise if Today Payoff> Ridge predicted value of waiting!
    PRICE = 0.
    @inbounds for i=1:n
        for j=1:m
            if CV[i, j] < CFL[i, j+1] && CFL[i, j+1] > 0.
                PRICE += CFL[i, j+1]*exp(-r*(j-1)*Δt)
                break
            end
        end
    end
    PRICE /= n
end
######



compute_price(36., .2, 40., .06, 100000, 10, 1/10, 5)
#
Spot = 36.0
σ = 0.2
n = 100000
m = 10
K = 40.0
r = 0.06
T = 1
Δt = T / m
ε = 1e-2
#
# warmup All. SIMD.
compute_price(Spot, σ, K, r, n, m, Δt, 5)
compute_price(Spot + ε, σ, K, r, n, m, Δt, 5)
compute_price(Spot, σ + ε, K, r, n, m, Δt, 5)
compute_price(Spot, σ, K + ε, r, n, m, Δt, 5)
compute_price(Spot, σ, K, r + ε, n, m, Δt, 5)
#Median time brought down from 254 to 219 ms + Vasily to 148 ms
@benchmark compute_price(Spot, σ, K, r, n, m, Δt, 5)
#typeof(Spot), typeof(σ), typeof(K), typeof(r)
#
for order in [5, 10, 25, 50]
    t0 = time();
    P = compute_price(Spot, σ, K, r, n, m, Δt, order);
    dP_dS = (compute_price(Spot + ε, σ, K, r, n, m, Δt, order) - P) / ε;
    dP_dσ = (compute_price(Spot, σ + ε, K, r, n, m, Δt, order) - P) / ε;
    dP_dK = (compute_price(Spot, σ, K + ε, r, n, m, Δt, order) - P) / ε;
    dP_dr = (compute_price(Spot, σ, K, r + ε, n, m, Δt, order) - P) / ε;
    t1 = time();
    out = (t1 - t0)
    println(order, ":  ", 1000 * out)
end
