abstract type Transition end

struct IndependentMomentumTransition <: Transition
    H::GaussianEuclideanHamiltonian
end

function sample(t::IndependentMomentumTransition)
    randn(size(t.H.M, 1))
end
