include("Arianna.jl")

using .Arianna
using LinearAlgebra: LowerTriangular
using Plots
using PDMats
using Random

m = GaussianDensity([0.0; 0.0;;], LowerTriangular([1.0 0.0; 0.0 1.0]))
neg_log_dens = q -> -logdensity(m, q)
grad_neg_log_dens = q -> -gradlogdensity(m, q)
metric = PDMat([1.0 0.0; 0.0 1.0])

rng = MersenneTwister(12)

h = EuclideanSystem(neg_log_dens, grad_neg_log_dens, metric)
integrator = LeapfrogIntegrator(h, 0.02, 1)

samples, accepts = sample_chain(h, integrator, [2.0; 2.0;;], 400, rng)

f(x, y) = exp(-neg_log_dens([x; y;;]))

xrange = range(-5, 5, length = 400)
yrange = range(-5, 5, length = 400)

contour(
    xrange,
    yrange,
    (x, y) -> f(x, y),
    xlabel = "x₁",
    ylabel = "x₂",
    title = "Contour of Target Density",
    aspect_ratio = 1,
    fill = true,
    color = :viridis,
    levels = 15,
)

scatter!(samples[:, 1], samples[:, 2], title = "Trace Plot", ms = 3)

savefig("traceplot.png")
