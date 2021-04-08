include("Lower.jl")

function coefficients(log_pk::Function,gamma,n,upto)
    tolerance = 1e-12
    log_v = zeros(upto)
    for t = 1:upto
        if t>n; log_v[t] = -Inf; continue; end
        a,c,k,p = 0.0, -Inf, 1, 0.0
        while abs(a-c) > tolerance || p < 1.0-tolerance  # Note: The first condition is false when a = c = -Inf
            if k >= t
                a = c
                b = logabsgamma(k+1)[1]-logabsgamma(k-t+1)[1]-logabsgamma(k*gamma+n)[1]+logabsgamma(k*gamma)[1] + log_pk(k)
                c = logsumexp(a,b)
            end
            p += exp(log_pk(k))
            k = k+1
        end
        log_v[t] = c
    end
    return log_v
end

function coefficients_t(log_pk::Function,gamma,n,t)
    tolerance = 1e-12
    log_v = 0.0
    a,c,k,p = 0.0, -Inf, 1, 0.0
    while abs(a-c) > tolerance || p < 1.0-tolerance  # Note: The first condition is false when a = c = -Inf
        if k >= t
            a = c
            b = logabsgamma(k+1)[1]-logabsgamma(k-t+1)[1]-logabsgamma(k*gamma+n)[1]+logabsgamma(k*gamma)[1] + log_pk(k)
            c = logsumexp(a,b)
        end
        p += exp(log_pk(k))
        k = k+1
    end
    log_v = c
    return log_v
end

logsumexp(a,b) = (m = max(a,b); m == -Inf ? -Inf : log(exp(a-m) + exp(b-m)) + m)

function randp(p,k)
    s = 0.; for j = 1:k; s += p[j]; end
    u = rand()*s
    j = 1
    C = p[1]
    while u > C
        j += 1
        C += p[j]
    end
    @assert(j <= k)
    return j
end

function randlogp!(log_p,k)
    log_s = -Inf; for j = 1:k; log_s = logsumexp(log_s,log_p[j]); end
    p = log_p
    for j = 1:k; p[j] = exp(log_p[j]-log_s); end
    return randp(p,k)
end

function ordered_insert!(index,list,t)
    j = t
    while (j>0) && (list[j]>index)
        list[j+1] = list[j]
        j -= 1
    end
    list[j+1] = index
end

function ordered_insert!(index,x,t,c)
    j = t
    list = x[c, :]
    while (j>0) && (list[j]>index)
        list[j+1] = list[j]
        j -= 1
    end
    list[j+1] = index
    x[c, :] = list
end

function ordered_remove!(index,list,t)
    for j = 1:t
        if list[j]>=index; list[j] = list[j+1]; end
    end
end

function ordered_remove!(index,x,t,c)
    list = x[c, :]
    for j = 1:t
        if list[j]>=index; list[j] = list[j+1]; end
    end
    x[c, :] = list
end

function ordered_next(list)
    j = 1
    while list[j] == j; j += 1; end
    return j
end

function log_lik(y, X, W, sigma2, g, cc, H)
    p, q = H.p, H.q
    
    b = g.b
    theta = g.theta[cc, :]
    
    Xb, Wtheta = 0.0, 0.0
    for i = 1 : p; Xb += X[i] * b[i]; end
    for i = 1 : q; Wtheta += W[i] * theta[i]; end
    res = y - Xb - Wtheta
    res2 = res * res

    return -0.5 * (log(2) + log(pi) + log(sigma2)) - 0.5 * res2 / sigma2
end  

mutable struct Group
    b::Array{Float64,1}
    theta::Array{Float64, 2}
    Group(p, q, t_max) = (g = new(); g.b = zeros(p); g.theta = zeros(t_max + 3, q); g)
end

mutable struct MVN_params
    m::Array{Float64,1} # mean
    L::Array{Float64,2} # lower triangular matrix such that L*L' = covariance
    _L::Array{Float64,2} # _L = inv(L) and _L' * _L = precision
    R::Array{Float64,2} # covariance matrix
    _R::Array{Float64,2} # precision matrix 
    logdetR::Float64 # log of the determinant of the covariance matrix
    d::Int64 # dimension
    function MVN_params(m, R)
        p = new(); p.m = copy(m); p.R = copy(R); p.d = d = length(m)
        p.L = zeros(d,d); Lower.Cholesky!(p.L, p.R, d)
        p._L = zeros(d,d); Lower.inverse!(p._L, p.L, d)
        p._R = zeros(d,d); Lower.multiplyMtN!(p._R, p._L, p._L, d)
        p.logdetR = Lower.logdetsq(p.L, d) 
        return p
    end
end

struct Hyperparameters
    n::Int64 # sample size
    p::Int64 # dimension of X
    q::Int64 # dimension of W

    a0::Float64 # parameters of IG on sigma2
    b0::Float64 # parameters of IG on sigma2

    mb::MVN_params # 
    mtheta::MVN_params # 
end

function construct_hyperparameters(X, W)
    n, p, q = length(X), length(X[1]), length(W[1])
    
    a0, b0 = 1, 1
    
    mb = MVN_params(zeros(p), 10 * Matrix(I, p, p))
    mtheta = MVN_params(zeros(q), 10 * Matrix(I, q, q))
    return Hyperparameters(n, p, q, a0, b0, mb, mtheta)
end

function update_theta!(y, X, Z, z, z1, g, sigma2, c, cc, H)
    n, p, q = H.n, H.p, H.q
    m, L, _L, _R = H.mtheta.m, H.mtheta.L, H.mtheta._L, H.mtheta._R

    zz = zeros(q, q)
    zy = zeros(q)
    zxb = zeros(q)
    
    M = zeros(q, q)
    _M = zeros(q, q)
    A = zeros(q, q)

    for k = 1 : n;
        if z[k] == c
            if z1[k] == cc; 
                Z_k = Z[k]
                for j = 1 : q; for i = 1 : q; zz[i, j] += Z_k[i] * Z_k[j]; end; end
                zy += Z_k * y[k]
                xb = 0.0; for i = 1 : p; xb += X[k][i] * g[c].b[i]; end
                zxb += Z_k * xb
            end
        end
    end
        
    _A = zz + sigma2 * _R
    Lower.Cholesky!(M, _A, q) # M * M' = _A
    Lower.inverse!(_M, M, q) # _M = inv(M)
    Lower.multiplyMtN!(A, _M, _M, q) # _M' * _M = A

    mu = zy - zxb + sigma2 * _R * m
    mu = A * mu
    A = Matrix(Hermitian(A))
    g[c].theta[cc, :] = rand(MvNormal(mu, sigma2 * A))
end

function update_b!(y, X, Z, z, z1, g, sigma2, c, t1, list1, H)
    n, p, q = H.n, H.p, H.q
    m, L, _L, _R = H.mb.m, H.mb.L, H.mb._L, H.mb._R
    
    xx = zeros(p, p)
    res = zeros(p)
    
    M = zeros(p, p)
    _M = zeros(p, p)
    A = zeros(p, p)
   
    for k = 1 : n;
        if z[k] == c; 
            x_k = X[k]
            for j = 1 : p; for i = 1 : p; xx[i, j] += x_k[i] * x_k[j]; end; end
            for j = 1 : t1[c]; cc = list1[c, j]
                if z1[k] == cc; 
                    Z_k = Z[k]
                    ztheta = 0.0; for i = 1 : q; ztheta += Z_k[i] * g[c].theta[cc, i]; end; 
                    res += x_k * (y[k] - ztheta)
                end
            end
        end
    end

    _A = xx + sigma2 * _R
    Lower.Cholesky!(M, _A, p) # M * M' = _A
    Lower.inverse!(_M, M, p) # _M = inv(M)
    Lower.multiplyMtN!(A, _M, _M, p) # _M' * _M = A

    mu = res + sigma2 * _R * m
    mu = A * mu
    A = Matrix(Hermitian(A))
    g[c].b = rand(MvNormal(mu, sigma2 * A))
end


function update_sigma2(y, X, Z, z, z1, g, t, t1, list, list1, H)

    n, p, q = H.n, H.p, H.q
    m_b, L_b, _R_b =  H.mb.m, H.mb.L, H.mb._R
    m_theta, L_theta, _R_theta = H.mtheta.m, H.mtheta.L, H.mtheta._R
    a0, b0 = H.a0, H.b0
    
    res, res2 = 0.0, 0.0
    quad_b, quad_theta = 0.0, 0.0
    for k = 1 : n;
        x_k, z_k = X[k], Z[k]
        z_id, z1_id = z[k], z1[k]
        xb = 0.0; for j = 1 : p; xb += x_k[j] * g[z_id].b[j]; end
        ztheta = 0.0; for j = 1 : q; ztheta += z_k[j] * g[z_id].theta[z1_id, j]; end
        res = y[k] - xb - ztheta
        res2 += res * res
    end
    
    x = rand(InverseGamma(0.5 * n + a0, 0.5 * res2 + b0))
    return x
end

function sampler(y, X, W; n_total = 10000, n_burn = 5000, t_max = 40)
    n = length(y)
    H = construct_hyperparameters(X, W)
    
    ### One cluster
    log_pk = k -> log(0.1)+(k-1)*log(0.9)
    log_pk1 = k -> log(0.1)+(k-1)*log(0.9)
    a = b = 1
    log_v = coefficients(log_pk, a, n, t_max + 3);
    log_v1 = zeros(t_max + 3, n)
    for i = 1 : n; log_v1[:, i] = coefficients(log_pk1, a, i, t_max + 3); end

    z = fill(1, n); z1 = fill(1, n)
    N, N1 = zeros(Int, t_max + 3), zeros(Int, t_max + 3, t_max + 3)
    N[1] = n
    N1[1, 1] = n
    c_next = 2
    c_next1 = zeros(Int, t_max + 3); c_next1[1] = 2; c_next1[2] = 1; 
    c_prop1 = copy(c_next1)
    list = zeros(Int, t_max + 3); list[1] = 1; 
    list1 = zeros(Int, t_max + 3,  t_max + 3); list1[1, 1] = 1;
    t = 1; t1 = zeros(Int, t_max + 3); t1[1] = 1;

    group = [Group(p, q, t_max)::Group for c = 1 : t_max + 3];
    group[1].b = rand(MvNormal(zeros(p), H.mb.R))
    group[1].theta[1, :] = rand(MvNormal(zeros(q), H.mtheta.R))
    sigma2 = rand(InverseGamma(2, 1))

    log_Nb = log.((1 : n) .+ b)
    log_Nb1 = log.((1 : n) .+ b)
    log_p = zeros(t_max + 3, t_max + 3);

    t_r = zeros(Int, n_total)
    t1_r = zeros(Int, t_max + 3, n_total);
    z_r = zeros(Int, n, n_total)
    z1_r = zeros(Int, n, n_total)
    N_r = zeros(Int16, (t_max + 3), n_total)
    N1_r = zeros(Int16, (t_max + 3),  (t_max + 3), n_total)
    sigma2_r = zeros(Float64, n_total)
    b_r = zeros(Float64, t_max + 3, p, n_total)
    theta_r = zeros(Float64, t_max + 3, q, t_max +3, n_total) #theta_r = zeros(Float64, t_max + 3, q, n_total);

    for iteration = 1 : n_total
        for i = 1 : n
            c = z[i]
            N[c] -= 1

            c1 = z1[i]
            N1[c, c1] -= 1

            if N[c] > 0
                c_prop = c_next
                # new-new 
                c_prop1[c_prop] = 1 
                group[c_prop].b = rand(MvNormal(zeros(p), H.mb.R))
                group[c_prop].theta[1, :] = rand(MvNormal(zeros(q), H.mtheta.R))
                # existing-new 
                if N1[c, c1] > 0
                    c_prop1[c] = c_next1[c]
                    group[c].theta[c_prop1[c], :] = rand(MvNormal(zeros(q), H.mtheta.R))
                else
                    c_prop1[c] = c1
                    ordered_remove!(c1, list1, t1[c], c)
                    t1[c] -= 1
                end
            else
                c_prop = c
                ordered_remove!(c, list, t)
                t -= 1
                c_prop1[c] = c1
                ordered_remove!(c1, list1, t1[c], c)
                t1[c] -= 1
            end

            # new group-specific pars for cc that is not equal to c
            for j = 1 : t; cc = list[j]
                if cc != c  
                    c_prop1[cc] = c_next1[cc]
                    group[cc].theta[c_prop1[cc], :] = rand(MvNormal(zeros(q), H.mtheta.R))
                end
            end
            fill!(log_p, 0.0)
            for j = 1 : t; 
                cc = list[j]
                Nd = N[cc] + 1
                log_v_ratio = log_v1[t1[cc] + 1, Nd] - log_v1[t1[cc], Nd] + log(a)
                log_adjconst = 0.0
                log_adjconst = log(exp(log_v_ratio) + N[cc] + t * a)
            
                for jj = 1 : t1[cc]
                    cc1 = list1[cc, jj]
                    Ns_d = N1[cc, cc1]
                    log_p[j, jj] = log_Nb[N[cc]] + log_Nb1[Ns_d] + 
                    log_lik(y[i], X[i], W[i], sigma2, group[cc], cc1, H) - log_adjconst
                end
                log_p[j, t1[cc] + 1] = log_Nb[N[cc]] + log_v_ratio + 
                log_lik(y[i], X[i], W[i], sigma2, group[cc], c_prop1[cc], H) - log_adjconst
            end

            log_p[t + 1, 1] = log_v[t + 1] - log_v[t] + log(a) + 
            log_lik(y[i], X[i], W[i], sigma2, group[c_prop], c_prop1[c_prop], H)

            sum_t = 0; 
            for aa = 1 : t;
                cc = list[aa]
                sum_t += (t1[cc] + 1); 
            end; 
            sum_t = sum_t + 1
            id = [[aa, bb] for aa = 1 : t for bb = 1 : t1[list[aa]]+1]  
            push!(id, [t+1, 1])
            tmp = zeros(length(id))
            for i = 1 : length(id)
                tmp[i] = log_p[id[i][1], id[i][2]]
            end
            j = id[randlogp!(tmp, sum_t)] 

            if j[1] <= t
                c = list[j[1]]
                if j[2] <= t1[c]
                    c1 = list1[c, j[2]]
                else 
                    c1 = c_prop1[c]
                    ordered_insert!(c1, list1, t1[c], c)
                    t1[c] += 1
                    c_next1[c] = ordered_next(list1[c, :])
                end
            else
                c = c_prop
                ordered_insert!(c, list, t)
                t += 1
                c_next1[c_next] = 0
                c_next = ordered_next(list)
                @assert(t <= t_max, "Sampled t has exceeded t_max. Increase t_max and retry.")
                c1 = c_prop1[c]
                ordered_insert!(c1, list1, t1[c], c)
                t1[c] += 1
                c_next1[c] = ordered_next(list1[c, :])
                c_next1[c_next] = ordered_next(list1[c_next, :])
            end

            z[i] = c 
            N[c] += 1
            z1[i] = c1
            N1[c, c1] += 1
        end

        for j = 1 : t; c = list[j]; 
            for jj = 1 : t1[c]; cc = list1[c, jj]
                update_theta!(y, X, W, z, z1, group, sigma2, c, cc, H)
            end
            c_next1[c] = ordered_next(list1[c, :])
        end
        c_next1[c_next] = ordered_next(list1[c_next, :])

        for j = 1 : t; c = list[j];
            update_b!(y, X, W, z, z1, group, sigma2, c, t1, list1, H)
        end

        sigma2 = update_sigma2(y, X, W, z, z1, group, t, t1, list, list1, H)
    
        t_r[iteration] = t
        t1_r[:, iteration] = t1
        z_r[:, iteration] = z
        z1_r[:, iteration] = z1
        N_r[:, iteration] = N
        N1_r[:, :, iteration] = N1
        sigma2_r[iteration] = sigma2

        for j = 1 : t; c = list[j]
            b_r[c, :, iteration] = group[c].b
            for jj = 1 : t1[c]; cc = list1[c, jj]
                theta_r[cc, :, c, iteration] = group[c].theta[cc, :]
            end
        end
    end
    
    return t_r, t1_r, z_r, z1_r, N_r, N1_r, b_r, theta_r, sigma2_r
end

struct Result
    t::Array{Int16, 1}
    t1::Array{Int16, 2}
    z::Array{Int16, 2}
    z1::Array{Int16, 2}
    N::Array{Int16, 2}
    N1::Array{Int16, 3}
    beta::Array{Float64, 3}
    theta::Array{Float64, 4}
    sigma2::Array{Float64, 1}
    elapsed_time::Float64
    time_per_step::Float64
end

function run_sampler(y, X, W; n_total = 10000, n_burn = 5000, t_max = 40)
    print("Running... ")
    
    # Main run
    elapsed_time = @elapsed t, t1, z, z1, N, N1, beta, theta, sigma2 = sampler(y, X, W, n_total = n_total, n_burn = n_burn, t_max = t_max)
    time_per_step = elapsed_time / n_total
    
    println("complete.")
    println("Elapsed time = $elapsed_time seconds")
    println("Time per step ~ $time_per_step seconds")
    
    return Result(t, t1, z, z1, N, N1, beta, theta, sigma2, elapsed_time, time_per_step)
end

# Compute the posterior similarity matrix (probability that i and j are in same cluster).
function similarity_matrix(n, z_r)
    z = z_r
    C = zeros(Int,n,n)
    n_used = 0

    for l = 1 : length(z_r[1, :])
        for i = 1:n, j = 1:n
            if z[i, l] == z[j, l]; C[i,j] += 1; end
        end
    n_used += 1
    end
    
    return C / n_used
end


function get_list(res, n_total, t_max)
    x = zeros(Int, size(res)[1], size(res)[2])
    
    for j = 1 : n_total
        ii, jj = 1, 1
        for i = 1 : t_max
            if res[jj, j] != 0; x[ii, j] = jj; ii += 1; end
            jj += 1; 
        end
    end
    
    return x
end

function get_list1(res; t_max = 40)
    x = zeros(Int, t_max)
    ii, jj = 1, 1
    for i = 1 : t_max
        if res[jj] != 0; x[ii] = jj; ii += 1; end
        jj += 1
    end
    
    return x
end

# post process the memberships for sub-subgroups
function z1_post_ftn(s1; n_total = 10000)
    t_r = s1.t
    t1_r = s1.t1
    z_r = s1.z
    z1_r = s1.z1
    N_r = s1.N
    N1_r = s1.N1
    
    n = size(z_r)[1]
    t_max = size(N_r)[1]
    z1_post = zeros(Int, n, n_total)
        
    list_r = get_list(N_r, n_total, t_max)
  
    for iter = 1 : n_total
        ii = 1
        for j = 1 : t_r[iter]
            c = list_r[j, iter]
            tmp = get_list1(N1_r[c, :, iter], t_max = t_max)
            for jj = 1 : t1_r[c, iter]
                cc = tmp[jj]
                for i = 1 : n
                    if z_r[i, iter] == c
                        if z1_r[i, iter] == cc
                            z1_post[i, iter] = ii
                        end
                    end
                end
                ii += 1
            end
        end
    end
    
    return z1_post
end