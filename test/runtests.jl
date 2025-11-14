using Test
using Arianna
import Arianna.RandomWalk
using Distributions
using LogDensityProblems
using Random
using AbstractMCMC
using Plots
using LinearAlgebra

@testset "Multidimensional Random Walk" begin
    rng = Random.default_rng()

    # 2D Gaussian
    dist = MvNormal([0.0, 0.0], I(2))
    model = RandomWalk.DistributionModel(dist)

    # Initial sampler
    s0 = RandomWalk.RWSampler([0.0, 0.0], 0.1)
    @test s0.position isa Vector{Float64}
    @test length(s0.position) == 2

    # One step
    s1, logp = AbstractMCMC.step(rng, model, s0)
    @test s1.position isa Vector{Float64}
    @test length(s1.position) == 2
    @test logp isa Float64

    # Multiple samples
    samples = AbstractMCMC.sample(rng, model, s0, 100)
    @test samples isa Matrix{Float64}
    @test size(samples) == (100, 2)

    # # Plot each dimension separately
    # x = samples[:,1]
    # y = samples[:,2]
    # t = 1:size(samples,1)

    # p = plot3d(x, y, t,
    # xlabel = "x₁",
    # ylabel = "x₂",
    # zlabel = "Step",
    # title = "3D Random Walk Trajectory",
    # linealpha = 5,
    # legend = false)

    # savefig("trace_mv.png")
end



