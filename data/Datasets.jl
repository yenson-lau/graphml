import Downloads, Tar, CodecZlib
using CSV, DataFrames

DATA_DIR = @__DIR__

function CoraDataset()::Dict{Symbol, Any}
  CORA_DIR = "$(DATA_DIR)/cora"

  if !isdir(CORA_DIR)
    try
      url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
      fname = Downloads.download(url)

      open(fname) do tar_gz
        tar = CodecZlib.GzipDecompressorStream(tar_gz)
        outdir = Tar.extract(tar)
        mv("$(outdir)/cora", CORA_DIR)
      end
    catch e
      error("failed to download CORA dataset")
    end
  end

  cites = CSV.File("$(CORA_DIR)/cora.cites"; delim="\t", header=[:u, :v]) |> DataFrame

  content = CSV.File("$(CORA_DIR)/cora.content"; delim="\t", header=false) |> DataFrame
  item_featidx = Dict(u=>i for (i, u) in enumerate(content[:,1]))
  feature_matrix = Matrix(content[:,2:end-1])   # what's the right dtype?

  node_classes = [c for c in Set(content[:,end])]
  label_clsidx = Dict(c=>i for (i,c) in enumerate(node_classes))
  node_labels = [label_clsidx[c] for c in content[:, end]]

  return Dict(
    :edges => cites,
    :item_featidx => item_featidx,
    :features => feature_matrix,
    :node_classes => node_classes,
    :node_labels => node_labels,
  )
end
