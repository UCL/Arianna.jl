# Integrator methods for solving discretized hamiltonian systems
abstract type Integrator end

struct LeapfrogIntegrator <: Integrator
    h::SimpleHamiltonian
    ε::Float64
    T::Float64
end

function leapfrog_step(
    h::SimpleHamiltonian,
    q::AbstractVector,
    p::AbstractVector,
    ε::Float64,
)
    p_half = p .- (ε / 2) * ∇U(h, q)
    q_new = q .+ ε * (inv(h.M) * p_half)
    p_new = p_half .- (ε / 2) * ∇U(h, q_new)
    return q_new, p_new
end

function integrate(LI::LeapfrogIntegrator, q0::AbstractVector, p0::AbstractVector)
    L = convert(UInt64, floor(LI.T / LI.ε))
    q, p = q0, p0
    for n = 1:L
        q, p = leapfrog_step(LI.h, q, p, LI.ε)
    end
    return q, p
end
