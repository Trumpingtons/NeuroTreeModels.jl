"""
    importance(m::NeuroTreeModel, data)

Gradient-based feature importance of `m` over `data`, a `DataFrame` or any other Tables.jl table holding the model's features.

For each feature, the mean absolute derivative of the model's raw outputs (before any link, such as the softmax of a classifier)
with respect to that feature, over the rows of `data`, summed over the outputs and multiplied by the feature's standard deviation in `data`,
so that features on different scales compare. The importances sum to 1.

Returns a `Vector{Pair{String,Float64}}`, sorted by decreasing importance, as `EvoTrees.importance` does.
"""
function importance(m::NeuroTreeModel, data)
    df = data isa AbstractDataFrame ? data : DataFrame(data)
    feature_names = Symbol.(m.info[:feature_names])
    x = Matrix{Float32}(Matrix{Float32}(select(df, feature_names))')
    chain = cpu(m.chain)
    Flux.testmode!(chain)  # the running statistics of BatchNorm, as at inference, not those of the batch
    saliency = try
        outs = size(chain(x), 1)
        sum(1:outs) do o
            g = gradient(x -> sum(view(chain(x), o, :)), x)[1]
            vec(mean(abs.(g); dims=2))
        end
    finally
        Flux.testmode!(chain, :auto)
    end
    imp = saliency .* vec(std(x; dims=2))
    imp ./= sum(imp)
    return sort(String.(feature_names) .=> Float64.(imp); by=last, rev=true)
end
