# ========================================================
# Abstract System Type
# ========================================================

"""
    AbstractSystem

Base abstract type for Hamiltonian systems with energy

    H(q, p) = U(q) + K(q, p)

All abstract systems provide:

    U(H, q)      -- potential energy
    ∇U(H, q)     -- gradient of potential energy
"""
abstract type AbstractSystem end

U(H::AbstractSystem, q) = H.U(q)
∇U(H::AbstractSystem, q) = H.∇U(q)

K(H::AbstractSystem, p::AbstractVector) =
    error("K(H, p) not implemented for $(typeof(H))")

K(H::AbstractSystem, q::AbstractVector, p::AbstractVector) =
    error("K(H, q, p) not implemented for $(typeof(H))")

# ========================================================
# Position-Independent Flow TRAIT
# ========================================================

"""
    position_independent_flow(H)::Val{true/false}

Trait indicating whether K(H, q, p) truly depends on q.
"""
position_independent_flow(::AbstractSystem) = Val(false)


generic_K(H::AbstractSystem, q::AbstractVector, p::AbstractVector) =
    position_independent_flow(H) === Val(true) ?
    K(H, p) :
    K(H, q, p)




# ========================================================
# Hamiltonian
# ========================================================

hamiltonian(H::AbstractSystem, q, p) = U(H, q) + K(H, q, p)


# ========================================================
# Euclidean System (position-independent K)
# ========================================================

abstract type EuclideanSystem <: AbstractSystem end

"""
    mass_matrix(H)

Euclidean systems must define a fixed positive-definite mass matrix.
"""
mass_matrix(H::EuclideanSystem) =
    error("Euclidean systems must implement mass_matrix(H)")

# Declare trait:
position_independent_flow(::EuclideanSystem) = Val(true)

# Implement position-independent K
K(H::EuclideanSystem, p::AbstractVector) =
    0.5 * dot(p, mass_matrix(H) \ p)


# ========================================================
# Concrete Euclidean Hamiltonian
# ========================================================

"""
    EuclideanHamiltonian(U, ∇U, M)

Concrete Euclidean Hamiltonian with fixed mass matrix M.
"""
struct EuclideanHamiltonian{FU,FG,MType} <: EuclideanSystem
    U::FU
    ∇U::FG
    M::MType
end

U(H::EuclideanHamiltonian, q) = H.U(q)
∇U(H::EuclideanHamiltonian, q) = H.∇U(q)
mass_matrix(H::EuclideanHamiltonian) = H.M






