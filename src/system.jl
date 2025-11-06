# Hamiltonian systems representing energy functions and their derivatives

struct SimpleHamiltonian
    logdensity::Function
    gradlogdensity::Function
    M::AbstractMatrix
end

function U(h::SimpleHamiltonian, q::AbstractVector)
    -h.logdensity(q)
end

function ∇U(h::SimpleHamiltonian, q::AbstractVector)
    -h.gradlogdensity(q)
end

function K(h::SimpleHamiltonian, p::AbstractVector)
    0.5 * dot(p, inv(h.M) * p)
end

function sample_p(h::SimpleHamiltonian)
    randn(size(h.M, 1))
end