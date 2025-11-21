module RandomWalk

using AbstractMCMC
using Random
using Distributions
using LogDensityProblems

# Step 1 — Model
struct DistributionModel <: AbstractMCMC.AbstractModel
    dist::Distribution
end

LogDensityProblems.logdensity(model::DistributionModel, x) = 
    logpdf(model.dist, x)

LogDensityProblems.dimension(model::DistributionModel) = length(mean(model.dist))


# Step 2 - Sampler
struct RWSampler <: AbstractMCMC.AbstractSampler 
    position::Vector{Float64}
    stepsize::Float64
end

# Step 3 - Random Walk Metropolis Step
function AbstractMCMC.step(
    rng::AbstractRNG,
    model::DistributionModel,
    sampler::RWSampler
    )

    proposal = sampler.position .+ sampler.stepsize .* randn(rng, length(sampler.position))

    logp_current = LogDensityProblems.logdensity(model, sampler.position)
    logp_proposal = LogDensityProblems.logdensity(model, proposal)

    log_accept_ratio = logp_proposal - logp_current

    if log(rand(rng)) < log_accept_ratio
        new_position = proposal
    else
        new_position = sampler.position
    end

    return RWSampler(new_position, sampler.stepsize), logp_current
end

function AbstractMCMC.step(rng::AbstractRNG,
                           model::DistributionModel,
                           sampler::RWSampler,
                           _weight::Float64)

    # Multiple dispatch, ignore the weight and call the usual step
    return AbstractMCMC.step(rng, model, sampler)
end

# Step 4 - Sampling Interface
function AbstractMCMC.sample(
    rng::AbstractRNG,
    model::DistributionModel,
    sampler::RWSampler,
    n_samples::Integer,
    kwargs...
    )

    d = length(sampler.position)
    samples = Matrix{Float64}(undef, n_samples, d)

    current_sampler = sampler 

    for i in 1:n_samples
        current_sampler, logp = AbstractMCMC.step(rng, model, current_sampler)
        samples[i, :] = current_sampler.position
    end

    return samples
end
end