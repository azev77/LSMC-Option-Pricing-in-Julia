using Statistics, LinearAlgebra, Random, BenchmarkTools

function chebyshev_basis!(x, B)
    B[:,2] .= x
    for n in range(3, stop = size(B,2))
        @. @views B[:,n] = 2. * x * B[:,n - 1] - B[:,n - 2]
    end
end

function ridge_regression!(out, X, Y, λ)
    # should be possible to also do inplace, somehow
    β = (X'X + λ * I) \ (X'Y)
    mul!(out, X, β)
end

function scale!(x)
    xmin,xmax = extrema(x)
    a = 2. / (xmax - xmin)
    b = -0.5 * a * (xmin + xmax)
    @. x = a * x + b
end

function advance!(out, randScratch, S, r, σ, Δt, n)
    dB = sqrt(Δt) * randn!(randScratch)
    @. out = S + r * S * Δt + σ * S * dB
end

function compute_price(Spot, σ, K, r, n, m, Δt, order)
    Random.seed!(0)

    S = zeros(n, m + 1)
    start_view = view(S,:,1)
    randScratch = zeros(n)
    fill!(start_view,Spot)
    for t = 1:m
        # we can reuse the views from the last iteration
        next_view = view(S, :, t+1)
        advance!(next_view, randScratch, start_view, r, σ, Δt, n)
        start_view = next_view
    end

    discount = exp(-r * Δt)
    CFL = max.(0, K .- S)
    value = zeros(n, m)
    value[:, end] .= view(CFL ,: , size(CFL,2)) .* discount

    CV = zeros(n, m)
    chebyshev_scratchpad = ones(n, order)
    for k = 1:m -1
        t = m - k
        t_next = t + 1

        stmp = view(S, :, t_next)
        scale!(stmp)
        chebyshev_basis!(stmp, chebyshev_scratchpad)

        YY = view(value, :, t_next)
        cvview = view(CV, :, t)
        ridge_regression!(cvview, chebyshev_scratchpad, YY, 100)

        cflview = view(CFL, :, t_next)
        for i in axes(value, 1)
            value[i, t] = discount * ifelse(cflview[i] > cvview[i], cflview[i], YY[i])
        end
    end
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
#    return PRICE
end
######
compute_price(36., .2, 40., .06, 100000, 10, 1/10, 5)
@benchmark compute_price(36., .2, 40., .06, 100000, 10, 1/10, 5)

function run()
    Spot = 36.0
    σ = 0.2
    n = 100000
    m = 10
    K = 40.0
    r = 0.06
    T = 1.
    Δt = T / m
    ε = 1e-2
    for order in [5, 10, 25, 50]
        t0 = time();
        P = compute_price(Spot, σ, K, r, n, m, Δt, order);
        dP_dS = (compute_price(Spot + ε, σ, K, r, n, m, Δt, order) - P) / ε;
        dP_dσ = (compute_price(Spot, σ + ε, K, r, n, m, Δt, order) - P) / ε;
        dP_dK = (compute_price(Spot, σ, K + ε, r, n, m, Δt, order) - P) / ε;
        dP_dr = (compute_price(Spot, σ, K, r + ε, n, m, Δt, order) - P) / ε;
        t1 = time()
        out = (t1 - t0)
        println(order, ":  ", 1000 * out)
    end
end

isinteractive() || run()

run()

