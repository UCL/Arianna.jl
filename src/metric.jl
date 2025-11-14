abstract type AbstractMetric end

mutable struct DenseEuclideanMetric <: AbstractMetric
    M::AbstractMatrix{Float64}
    M⁻¹::AbstractMatrix{Float64}

    function DenseEuclideanMetric(M::AbstractMatrix{Float64})
        det(M) == 0 && error("Matrix not invertible!")
        new(M, inv(M))
    end
end
