# Integrator methods for solving discretized hamiltonian systems
abstract type AbstractIntegrator end

struct LeapfrogIntegrator <: AbstractIntegrator
    h::AbstractEuclideanSystem
    ε::Float64
    T::Float64
end

function leapfrog_step!(h::AbstractEuclideanSystem, state::ChainState, ε::Float64)
    state.p .-= (ε/2) .* ∂H₁∂q(h, state)
    state.q .+= ε .* ∂H₂∂p(h, state)
    state.p .-= (ε/2) .* ∂H₁∂q(h, state)
end

function integrate!(lfi::LeapfrogIntegrator, state::ChainState)
    L = convert(UInt64, floor(lfi.T / lfi.ε))
    for n = 1:L
        leapfrog_step!(lfi.h, state, lfi.ε)
    end
end
