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

sample_p(h::AbstractSystem, rng::AbstractRNG) =
    error("sample_p(h, state) not implemented for $(typeof(h))")



"""
    AbstractEuclideanSystem

Base abstract type for Euclidean Hamiltonian systems. All Euclidean systems require a 
constant positive definite matrix corresponding to the metric on the *unconstrained* 
position space. Positive definiteness is enforced by requiring metric to be of type 
`AbstractPDMat`.
"""
abstract type AbstractEuclideanSystem <: AbstractSystem end

metric(h::AbstractEuclideanSystem) = h.metric

∂H₂∂q(h::AbstractEuclideanSystem, state::ChainState) = cached_zeros(h)

"""
    EuclideanSystem

Composite type for an (Unconstrained) Euclidean System, with kinetric energy of the form
    H₂(q, p) = ½ pᵀ M⁻¹ p
where M is a constant positive definite matrix.
"""
struct EuclideanSystem <: AbstractEuclideanSystem
    neg_log_dens::Function
    grad_neg_log_dens::Function
    metric::AbstractPDMat
end

H₂(h::EuclideanSystem, state::ChainState) = Xt_invA_X(metric(h), state.p)
∂H₂∂p(h::EuclideanSystem, state::ChainState) = metric(h) \ state.p
sample_p(h::EuclideanSystem, rng::AbstractRNG) =
    sqrt(metric(h)) * randn(rng, size(metric(h), 1), 1)

# Cache for zero vector corresponding to derivative of H₂ w.r.t. q, which is zero for 
# position-independent metrics
const DEFAULT_ZERO_VEC = Dict{DataType,Vector}()

function cached_zeros(h::EuclideanSystem)
    T = typeof(h)
    get!(DEFAULT_ZERO_VEC, T) do
        zeros(h.dimension)
    end
end

"""
    AbstractRiemannianSystem

Base abstract type for Riemannian systems. 
"""
abstract type AbstractRiemannianSystem <: AbstractSystem end

