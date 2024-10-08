
This is a Julia package for evaluating the quality of cell segmentation in is
situ spatial transcriptomics datasets using the metric of spurious coexpression.

There are two steps in this procedure: (1) identify pairs of genes that will
show exaggerated coexpression when cells are missegmented and (2) measure the
degree of coexpression relative to nuclear segmentation.

The first part works by taking prior nuclear segmentation (which we assume has
low rates of spurious coexpression), and comparing it to nuclear expansion
segmentation (which we assume has high rates of spurious coexpression). Genes that
show dramatically higher rates of coexpression in nuclear expansion are considered
a spurious pair.

```julia

using SpuriousCoexpression

# Read nuclear segmentation and spurious gene pairs from a Xenium transcript table
cc_nuc, spurious_gene_pairs = spuriously_coexpressed_gene_pairs(
    "transcripts.csv.gz",
    "x_location",
    "y_location",
    "feature_name",
    "cell_id",
    -1,
    "overlaps_nucleus",
    1,
)
```

Secondly, we load another segmentation
```julia
cc = coexpression_dataset_from_h5ad("other-segmentation.h5ad")

rel_cc = relative_coexpression(cc_nuc, cc, spurious_gene_pairs)

# summarize rates somehow
@show mean(rel_cc)
```

Note that this is a purely intended for comparison between segmentation methods
on the same dataset; avoid drawing conclusions from how these rates vary across
platforms or datasets.