# Hamiltonian systems representing energy functions and their derivatives

"""
    AbstractSystem

Base abstract type for Hamiltonian systems. The Hamiltonian defines a system of equations 
which govern the dynamics of a physical system in terms of its position and momentum. These
systems are assumed to have the form:

    H(q, p) = U(q) + K(q, p)

where `U(q)` is the potential energy as a function of position `q`, and `K(q, p)` is the 
kinetic energy as a function of position `q` and momentum `p`.

For Hamiltonian Monte Carlo methods, the potential energy `U(q)` is assumed to be the
negative log-density of the target distribution.

All AbstractSystems are assumed to contain methods:
    - U(q): potential energy function
    - ∇U(q): gradient of the potential energy function
"""
abstract type AbstractSystem end

"""
    PositionIndependentFlowSystem
"""
abstract type PositionIndependentFlowSystem <: AbstractSystem end

"""
    EuclideanSystem

Abstract type for a Hamiltonian with a Euclidean metric on the position space. These are
respresented with fixed positive definite mass matrices M, which parameterize the kinetic 
energy as a zero mean Gaussian distribution over momentum, independent of position.

K(p) = 0.5 * p' * M^{-1} * p
"""
abstract type EuclideanSystem <: AbstractSystem end

"""
    ConstrainedSystem

Abstract type for a Hamiltonian with constraint on the position space. These position space
is assumed to be a differentable manifold embedded in a higher dimensional Euclidean space.
"""
abstract type ConstrainedSystem <: AbstractSystem end

"""
    RiemannianSystem

Abstract type for a Hamiltonian with a metric matrix representation of any generic type. The
position space is assumed to be a Riemannian manifold with a metric with position dependent
representation `M(q)`. The momentum is then the zero-mean Gaussian conditional distribution
given position `q`: `p | q ~ N(0, M(q))`.
"""
abstract type RiemannianSystem <: AbstractSystem end

"""
    U(H::AbstractSystem, q::AbstractVector)

Interface for potential energy..
"""
function U(H::AbstractSystem, q::AbstractVector)
    H.U(q)
end

"""
    ∇U(H::AbstractSystem, q::AbstractVector)

Interface for gradient of potential energy.
"""
function ∇U(H::AbstractSystem, q::AbstractVector)
    H.∇U(q)
end

struct EuclideanHamiltonian{T<:Function} <: Intersection{EuclideanSystem,PositionIndependentFlowSystem}
    U::T
    ∇U::T
    M::AbstractMetric
end

struct GaussianEuclideanHamiltonian <: EuclideanHamiltonian
    U::T
    ∇U::T
    M::AbstractMetric
end

function hamiltonian(H::AbstractSystem, q::AbstractVector, p::AbstractVector)
    return U(H, q) + K(H, q, p)
end

function K(H <: PositionIndependentFlowSystem, _, p::AbstractVector)
    K(H, p)
end

function K(H <: EuclideanSystem, p::AbstractVector)
    0.5 * dot(p, inv(H.M) * p)
end





