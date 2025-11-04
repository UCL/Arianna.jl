module Arianna

include("autodiff.jl")
include("system.jl")
include("integrator.jl")
include("sample.jl")

include("scratch.jl")
export dummy_g

end
