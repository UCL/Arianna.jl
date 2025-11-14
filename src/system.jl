# Hamiltonian systems representing energy functions and their derivatives

"""
    AbstractHamiltonian

Base abstract type for Hamiltonian systems. The Hamiltonian defines a system of equations 
which govern the dynamics of a physical system in terms of its position and momentum. These
systems are assumed to have the form:

    H(q, p) = U(q) + K(q, p)

where `U(q)` is the potential energy as a function of position `q`, and `K(q, p)` is the 
kinetic energy as a function of position `q` and momentum `p`.

For Hamiltonian Monte Carlo methods, the potential energy `U(q)` is assumed to be the
negative log-density of the target distribution.
"""
abstract type AbstractHamiltonian end

"""
    TractableFlowHamiltonian
"""
abstract type TractableFlowHamiltonian <: AbstractHamiltonian end

"""
    EuclideanHamiltonian

Abstract type for a Hamiltonian with a Euclidean metric on the position space. These are
respresented with fixed positive definite mass matrices M, which parameterize the kinetic 
energy as a zero mean Gaussian distribution over momentum, independent of position.

K(p) = 0.5 * p' * M^{-1} * p
"""
abstract type EuclideanHamiltonian <: AbstractHamiltonian end

"""
    ConstrainedHamiltonian

Abstract type for a Hamiltonian with constraint on the position space. These position space
is assumed to be a differentable manifold embedded in a higher dimensional Euclidean space.
"""
abstract type ConstrainedHamiltonian <: AbstractHamiltonian end

"""
    RiemannianHamiltonian

Abstract type for a Hamiltonian with a metric matrix representation of any generic type. The
position space is assumed to be a Riemannian manifold with a metric with position dependent
representation `M(q)`. The momentum is then the zero-mean Gaussian conditional distribution
given position `q`: `p | q ~ N(0, M(q))`.
"""
abstract type RiemannianHamiltonian <: AbstractHamiltonian end

struct EuclideanHamiltonianImplementation <:
       Intersection{EuclideanHamiltonian,TractableFlowHamiltonian}
    logdensity::Function
    gradlogdensity::Function
    M::AbstractMatrix
end


function U(h::EuclideanHamiltonian, q::AbstractVector)
    -h.logdensity(q)
end

function ∇U(h::EuclideanHamiltonian, q::AbstractVector)
    -h.gradlogdensity(q)
end

function K(h::EuclideanHamiltonian, p::AbstractVector)
    0.5 * dot(p, inv(h.M) * p)
end

function sample_p(h::EuclideanHamiltonian)
    randn(size(h.M, 1))
end
