include("Arianna.jl")

using .Arianna
using LinearAlgebra: LowerTriangular
using Plots

m = GaussianDensity([0.0, 0.0], LowerTriangular([1.0 0.0; 0.0 1.0]))
m_logdensity = q -> logdensity(m, q)
m_gradlogdensity = q -> gradlogdensity(m, q)
M = [1.0 0.0; 0.0 1.0;]

H = SimpleHamiltonian(m_logdensity, m_gradlogdensity, M)
integrator = LeapfrogIntegrator(H, 0.01, 0.5)

samples, accepts= sample_chain(H, integrator, [5.0, 5.0], 200)

f(x,y) = exp(m_logdensity([x,y]))

xrange = range(-5, 5, length=200)
yrange = range(-5, 5, length=200)

contour(xrange, yrange, (x, y) -> f(x, y),
    xlabel = "x₁",
    ylabel = "x₂",
    title = "Contour of Target Density",
    aspect_ratio = 1,
    fill = true,
    color = :viridis,
    levels = 15
)

plot!(samples[:, 1], samples[:, 2], title="Trace Plot", ms=1)

savefig("traceplot.png")
