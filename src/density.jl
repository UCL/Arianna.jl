using LinearAlgebra: LowerTriangular, diag, dot

struct GaussianDensity
    μ::AbstractMatrix{Float64}
    L::LowerTriangular{Float64}
end

function logdensity(m::GaussianDensity, x::AbstractMatrix)
    z = m.L \ (x .- m.μ)
    logdetΣ = 2sum(log, diag(m.L))
    -0.5(logdetΣ + dot(z, z))
end

function gradlogdensity(m::GaussianDensity, x::AbstractMatrix)
    -(m.L' \ (m.L \ (x .- m.μ)))
end
