abstract type AbstractTransition end

struct IndependentMomentumTransition <: AbstractTransition
    h::AbstractSystem
end

struct CorrelatedMomentumTransition <: AbstractTransition
    h::AbstractSystem
    state::ChainState
    resample_coefficient::Float64
    function CorrelatedMomentumTransition(h, state, resample_coefficient)
        @assert 0.0 ≤ resample_coefficient ≤ 1.0
        new(h, state, resample_coefficient)
    end
end

sample(transition::IndependentMomentumTransition) = sample_p(transition.h)
