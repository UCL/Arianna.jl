"""
    AbstractSystem

Base abstract type for Hamiltonian systems with energy
    H(q, p) = H₁(q) + H₂(q, p)
where
    q   --  current position
    p   --  current momentum
    H₁  --  potential energy
    H₂  --  kinetric energy

All abstract systems must have fields:
    neg_log_dens        --  function to calculate negative log density of the
                            target function at the current position
    grad_neg_log_dens   --  function to calculate gradient of the negative log density of 
                            the target function at the current position
"""
abstract type AbstractSystem end

H(h::AbstractSystem, state::ChainState) = H₁(h, state) .+ H₂(h, state)
H₁(h::AbstractSystem, state::ChainState) = h.neg_log_dens(state.q)
H₂(h::AbstractSystem, state::ChainState) =
    error("H₂(h, state) not implemented for $(typeof(h))")

∂H∂q(h::AbstractSystem, state::ChainState) = ∂H₁∂q(h, state) .+ ∂H₂∂q(h, state)
∂H₁∂q(h::AbstractSystem, state::ChainState) = h.grad_neg_log_dens(state.q)
∂H₂∂q(h::AbstractSystem, state::ChainState) = 
    error("∂H₂∂q(h, state) not implemented for $(typeof(h))")

∂H∂p(h::AbstractSystem, state::ChainState) = ∂H₂∂p(h, state)
∂H₂∂p(h::AbstractSystem, state::ChainState) = 
    error("∂H₂∂p(h, state) not implemented for $(typeof(h))")

sample_p(h::AbstractSystem, rng::AbstractRNG) = error("sample_p(h, state) not implemented for $(typeof(h))")

const DEFAULT_ZERO_VEC = Dict{DataType, Vector}()

"""
    AbstractEuclideanSystem

Base abstract type for Euclidean Hamiltonian systems. 
"""
abstract type AbstractEuclideanSystem <: AbstractSystem end

∂H₂∂q(h::AbstractEuclideanSystem, state::ChainState) = cached_zeros(h)

function cached_zeros(h::AbstractEuclideanSystem)
    T = typeof(h)
    get!(DEFAULT_ZERO_VEC, T) do
        zeros(h.dimension)
    end
end

"""
    EuclideanSystem

Composite type for Euclidean Systems
"""
struct EuclideanSystem <: AbstractEuclideanSystem
    neg_log_dens::Function
    grad_neg_log_dens::Function
    metric::AbstractPDMat
end

H₂(h::EuclideanSystem, state::ChainState) = Xt_invA_X(h.metric, state.p)
∂H₂∂p(h::EuclideanSystem, state::ChainState) = h.metric \ state.p
sample_p(h::EuclideanSystem, rng::AbstractRNG) = sqrt(h.metric) * randn(rng, size(h.metric, 1), 1)

"""
    AbstractRiemannianSystem

Base abstract type for Riemannian systems. 
"""
abstract type AbstractRiemannianSystem <: AbstractSystem end







