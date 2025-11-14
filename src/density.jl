using LinearAlgebra: LowerTriangular, diag, dot

struct GaussianDensity
    μ::Vector{Float64}
    L::LowerTriangular{Float64}
end

function logdensity(m::GaussianDensity, x::AbstractVector)
    z = m.L \ (x .- m.μ)
    logdetΣ = 2sum(log, diag(m.L))
    -0.5(logdetΣ + dot(z, z))
end

function gradlogdensity(m::GaussianDensity, x::AbstractVector)
    -(m.L' \ (m.L \ (x .- m.μ)))
end
