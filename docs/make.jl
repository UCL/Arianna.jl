using Arianna
using Documenter

DocMeta.setdocmeta!(Arianna, :DocTestSetup, :(using Arianna); recursive = true)

makedocs(;
    modules = [Arianna],
    authors = "Matt Graham, Ross Ah-Weng, Callum Lau, Anees Hussain, Jordan Simbananiye and contributors",
    sitename = "Arianna.jl",
    format = Documenter.HTML(;
        canonical = "https://raahweng.github.io/Arianna.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/raahweng/Arianna.jl", devbranch = "main")
