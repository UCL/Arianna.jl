using Test
using Arianna
import Arianna.RandomWalk
using Distributions
using LogDensityProblems
using Random
using AbstractMCMC
using Plots

@testset "Test Step 1" begin
    @show names(Arianna; all=true)
    @show names(Arianna.RandomWalk; all=true)

    m = RandomWalk.DistributionModel(Normal(0, 1))

    @test LogDensityProblems.logdensity(m, 0.0) isa Float64
    @test LogDensityProblems.logdensity(m, 0.0) ≈ logpdf(Normal(0, 1), 0.0)

end

@testset "Test Step 2" begin
    start_point = 0.0
    s = RandomWalk.RWSampler(start_point, 0.1)

    @test s.position == 0.0
    @test s.stepsize == 0.1
end

@testset "Test Step 3" begin
    rng = Random.default_rng()

    model = RandomWalk.DistributionModel(Normal(0, 1))
    s0 = RandomWalk.RWSampler(0.0, 0.1)

    s1, ll = AbstractMCMC.step(rng, model, s0)

    @test ll == LogDensityProblems.logdensity(model, s0.position)
    @test isa(s1.position, Float64)
    @test s1.stepsize == 0.1
end

@testset "Test Step 4" begin
    rng = Random.default_rng()

    model = RandomWalk.DistributionModel(Normal(0, 1))
    s0 = RandomWalk.RWSampler(0.0, 0.1)

    samples = AbstractMCMC.sample(rng, model, s0, 100)
    # println("DEBUG samples = ", samples)
    # println("DEBUG type(samples) = ", typeof(samples))
    # println("DEBUG map(typeof, samples) = ", map(typeof, samples))


    @test length(samples) == 100
    @test all(x -> isa(x, Float64), samples)
end

@testset "Visual Test" begin
    rng = Random.default_rng()
    model = RandomWalk.DistributionModel(Normal(0, 1))
    s0 = RandomWalk.RWSampler(0.0, 0.1)

    samples = AbstractMCMC.sample(rng, model, s0, 500)

    p = plot(samples, title="Random Walk Trace")
    savefig("trace.png")
    # println("Plot saved to trace.png")
end