using Arianna
using Test
using AbstractMCMC, Random
using LogDensityProblems


@testset "Can initialize RandomWalk State Sampler" begin

    d = 2
    θ_init = randn(d)
    rng = Random.default_rng()

    function log_density(params)
        return logpdf(MvNormal(zeros(d), I), params)
    end

    # https://github.com/TuringLang/AbstractMCMC.jl/blob/390012ece352b90969c80979941b5b6eba990d29/src/logdensityproblems.jl#L1-L12
    # examples here
    # https://github.com/TuringLang/AdvancedHMC.jl/blob/main/test/common.jl
    LogDensityProblems.dimension(::typeof(log_density)) = 2
    LogDensityProblems.logdensity(::typeof(log_density), θ) = log_density(θ)
    function LogDensityProblems.capabilities(::Type{typeof(log_density)})
        return LogDensityProblems.LogDensityOrder{0}()
    end    

    model = AbstractMCMC.LogDensityModel(log_density)


    struct RWMSampler <: AbstractMCMC.AbstractSampler 
    
        position  # position vector ... this needs to be coupled to the dimension of the 'problem'
    
    end


end
