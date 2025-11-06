# Generate samples from target distribution using Hamiltonian Monte Carlo
function hmc_step(H::SimpleHamiltonian, integrator::T, q0::AbstractVector) where {T<:Integrator}
    p0 = sample_p(H)
    q, p = integrate(integrator, q0, p0)
    current_U = U(H, q0)
    current_K = K(H, p0)
    proposed_U = U(H, q)
    proposed_K = K(H, -p)
    accept_prob = exp(current_U - proposed_U + current_K - proposed_K)
    if rand() < accept_prob
        return q, true
    else
        return q0, false
    end
end


function sample_chain(h::SimpleHamiltonian, integrator::T, q0::AbstractVector, N::Int) where {T<:Integrator}
    samples = zeros(eltype(q0), N, length(q0))
    accepts = BitVector(undef, N)
    q = q0
    for n in 1:N
        q, accepted = hmc_step(h, integrator, q)
        samples[n, :] = q
        accepts[n] = accepted
    end
    return samples, accepts
end