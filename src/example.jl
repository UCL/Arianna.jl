using Arianna
using LinearAlgebra

m = GaussianDensity([0.0, 0.0], LowerTriangular([1.0 0.0; 0.5 1.0]))
m_logdensity = q -> logdensity(m, q)
m_gradlogdensity = q -> gradlogdensity(m, q)
M = [1.0 0.0; 0.0 1.0;]


H = SimpleHamiltonian(m_logdensity, m_gradlogdensity, M)
integrator = LeapfrogIntegrator(H, 0.1, 3)

result = sample_chain(H, integrator, [0.5, -0.5], 100)

scatter(result[:, 1], aspect_ratio = :equal, legend = false)

