using Pkg
using LinearAlgebra
using StatsBase
using Distributions
using FreqTables
using Random
using Plots
using CSV
using Clustering
using SpecialFunctions
using RCall

include("fun.jl")

p = 10; q = 5
p0 = [0.5, 0.5]
n = 200

sigma2 = 0.5
n_total, n_burn = 20000, 10000
n_use = n_total - n_burn
t_max = 60

true_z = wsample(1 : 2, p0, n)
true_z1 = copy(true_z)
for i = 1 : n
    if true_z[i] == 1
        true_z1[i] = wsample(1:2, p0)
    else
        true_z1[i] = wsample(3:4, p0)
    end
end

true_b = [[zeros(p)]; [zeros(p)]]
for i = 1 : 3
    true_b[1][i] = rand(Uniform(-1.2, -0.8))
    true_b[2][i] = rand(Uniform(0.8, 1.2))
end

true_theta = [[zeros(q)]; [zeros(q)]; [zeros(q)]; [zeros(q)];]
for i = 1 : 4
    true_theta[1][i] = rand(Uniform(-1.2, -0.8))
    true_theta[2][i] = rand(Uniform(0.8, 1.2)) 
    true_theta[3][i] = rand(Uniform(-1.2, -0.8)) 
    true_theta[4][i] = rand(Uniform(0.8, 1.2)) 
end

# X - AR structure
mu_x = zeros(p - 1); fill!(mu_x, 1)
S_x = Matrix{Float64}(I, p - 1, p - 1)
for i = 1 : (p - 1); for j = 1 : p - 1; S_x[i, j] = 0.3 ^ abs(i - j); if i == j; S_x[i, j] = 1; end; end; end
X_ar = [rand(MvNormal(mu_x, S_x)) for i = 1 : n]
for i = 1 : n; pushfirst!(X_ar[i], 1.0); end
    
# W - AR structure
mu_w = zeros(q); fill!(mu_w, 1)
S_w = Matrix{Float64}(I, q, q)
for i = 1 : q; for j = 1 : q; S_w[i, j] = 0.3 ^ abs(i - j) end; end; 
W_ar = [rand(MvNormal(mu_w, S_w)) for i = 1 : n]

# Setup 1
X = deepcopy(X_ar)
W = deepcopy(W_ar)
y = zeros(n)
for i = 1 : n
    y[i] = dot(true_b[true_z[i]], X[i]) + dot(true_theta[true_z1[i]], W[i]) + rand(Normal(0, sqrt(sigma2)))
end
    
s1 = run_sampler(y, X, W, n_total = n_total, n_burn = n_burn, t_max = t_max)

# Posterior distribution of the number of subgroups 
freqtable(s1.t[(n_burn + 1) : n_total]) 

# Posterior distribution of the total number of sub-subgroups
freqtable([sum(s1.t1[:, i]) for i = (n_burn + 1) : n_total])

# Posterior similarity matrix for the subgroup
psim = similarity_matrix(n, s1.z[:, (n_burn + 1) : n_total])
heatmap(psim)

# Posterior similarity matrix for the sub-subgroup
z1 = z1_post_ftn(s1; n_total = n_total) 
psim1 = similarity_matrix(n, z1[:, (n_burn + 1) : n_total])
heatmap(psim1)

@rput z1 psim psim1
R"""
library(gplots)
heatmap.2(psim, trace = 'none', dendrogram = 'none', density.info = 'none', key = F, labRow = F, labCol = F)
heatmap.2(psim1, trace = 'none', dendrogram = 'none', density.info = 'none', key = F, labRow = F, labCol = F)
"""