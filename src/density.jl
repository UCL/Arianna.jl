using LinearAlgebra: diag, dot
using PDMats: PDMat, logdet

struct GaussianDensity
    μ::Vector{Float64}
    M::PDMat{Float64}
end

function logdensity(m::GaussianDensity, x::AbstractVector)
    z = m.M.chol.L \ (x .- m.μ)
    logdetΣ = logdet(m.M)
    -0.5(logdetΣ + dot(z, z))
end

function gradlogdensity(m::GaussianDensity, x::AbstractVector)
    -(m.M.chol.U \ (m.M.chol.L \ (x .- m.μ)))
end