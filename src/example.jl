include("Arianna.jl")

#using .Arianna: GaussianDensity, logdensity, gradlogdensity, SimpleHamiltonian, LeapfrogIntegrator, sample_chain
using PDMats: PDMat
using Plots


M = [1.0 0.0; 0.0 3.0;]
m = GaussianDensity([0.0, 0.0], PDMat(M))

m_logdensity = q -> logdensity(m, q)
m_gradlogdensity = q -> gradlogdensity(m, q)

H = SimpleHamiltonian(m_logdensity, m_gradlogdensity, M)
integrator = LeapfrogIntegrator(H, 0.01, 0.4)

samples, accepts= sample_chain(H, integrator, [5.0, 5.0], 1000)


f(x,y) = exp(m_logdensity([x,y]))

xrange = range(-6, 6, length=200)
yrange = range(-6, 6, length=200)

contour(xrange, yrange, (x, y) -> f(x, y),
    xlabel = "x₁",
    ylabel = "x₂",
    title = "Contour of Target Density",
    aspect_ratio = 1,
    fill = true,
    color = :viridis,
    levels = 15
)

scatter!(samples[:, 1], samples[:, 2], title="Trace Plot", ms=1)

savefig("traceplot.png")
