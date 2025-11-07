using Arianna
using Test
using AbstractMCMC, Random
using LogDensityProblems
using Distributions: logpdf, MvNormal
using LinearAlgebra

@testset "Can initialize LogDensity Model" begin


    struct MyLogDensity
        dim::Int
    end

    LogDensityProblems.dimension(m::MyLogDensity) = m.dim

    function LogDensityProblems.logdensity(m::MyLogDensity, θ::AbstractVector)
        return logpdf(MvNormal(zeros(m.dim), I), θ)
    end

    LogDensityProblems.capabilities(::Type{MyLogDensity}) = 
        LogDensityProblems.LogDensityOrder{0}()

    problem = MyLogDensity(2)
    θ = [0.5, -1.0]
    @show LogDensityProblems.logdensity(problem, θ)
    @show LogDensityProblems.dimension(problem)

    model = AbstractMCMC.LogDensityModel(problem)
    @show LogDensityProblems.logdensity(model.logdensity, θ)
    @show LogDensityProblems.dimension(model.logdensity)
    

    struct RandomWalkSampler <: AbstractMCMC.AbstractSampler
        scale::Float64
    end

    mutable struct RandomWalkState
        q::Vector{Float64}
        logπ::Float64
    end

    sampler = RandomWalkSampler(0.5)
    state = RandomWalkState([0.0, 0.0], LogDensityProblems.logdensity(model.logdensity, [0.0, 0.0]))
    @show state


    function AbstractMCMC.step(
        rng::AbstractRNG,
        model::AbstractMCMC.LogDensityModel{MyLogDensity},
        sampler::RandomWalkSampler,
        state::RandomWalkState,
        kwargs...,
    )
        proposal = state.q .+ sampler.scale .* randn(rng, length(state.q))
        logp_proposal = LogDensityProblems.logdensity(model.logdensity, proposal)

        log_accept_ratio = logp_proposal - state.logπ
        if log(rand(rng)) < log_accept_ratio
            new_position = proposal
        else
            new_position = state.q
        end

        return RandomWalkState(new_position, LogDensityProblems.logdensity(model.logdensity, new_position))
    end

    sampler = AbstractMCMC.Sample(sampler, state)

    positions = collect(sampler(1:10))
    @show positions

end