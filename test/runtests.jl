using Arianna
using Test
using AbstractMCMC, Random
using LogDensityProblems
using Distributions
using LinearAlgebra


@testset "Can initialize RandomWalk State Sampler" begin

    d = 2
    θ_init = randn(Float64, d)
    rng = Random.default_rng()

    function log_density(params):: Float64
        # data = rand(Normal(0, 1), 300)

        I_test = Matrix{Float64}(I, d, d)

        return Distributions::logpdf(MvNormal(zeros(Float64, d), I_test), params)
    end

    # https://github.com/TuringLang/AbstractMCMC.jl/blob/390012ece352b90969c80979941b5b6eba990d29/src/logdensityproblems.jl#L1-L12
    # examples here
    # https://github.com/TuringLang/AdvancedHMC.jl/blob/main/test/common.jl
    LogDensityProblems.dimension(::typeof(log_density)) = 2
    LogDensityProblems.logdensity(::typeof(log_density), θ) = log_density(θ)
    function LogDensityProblems.capabilities(::Type{typeof(log_density)})
        return LogDensityProblems.LogDensityOrder{0}()
    end    
    
    # struct DefaultModel <: AbstractMCMC.LogDensityModel end
    model = AbstractMCMC.LogDensityModel(log_density)


    struct RWMSampler <: AbstractMCMC.AbstractSampler 
    
        position  # position vector ... this needs to be coupled to the dimension of the 'problem'
    
    end

    function AbstractMCMC.step(rng::AbstractRNG, model::AbstractMCMC.LogDensityModel, sampler::RWMSampler)
    
        # Random Walk Metropolis step
        proposal = sampler.position .+ 0.5 .* randn(rng, length(sampler.position)) # this code assumes a normal Gaussian, symmetric
    
        logp_current = (model.logdensity(sampler.position))
        logp_proposal = (model.logdensity(proposal))
    
        log_accept_ratio = logp_proposal - logp_current
    
        if log(rand(rng)) < log_accept_ratio
            new_position = proposal
        else
            new_position = sampler.position
        end
    
        return (RWMSampler(new_position), AbstractMCMC.LogDensityStats(logp_current))
    
    end

    sampler = AbstractMCMC.Sample(rng, model, RWMSampler(θ_init)) 
    
    positions = collect(sampler(1:10))
    println("RandomWalk State Sampler initialized and ran successfully.")
    println("Final position: ", positions)

end
