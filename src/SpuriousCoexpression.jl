module SpuriousCoexpression

export compute_relative_spurious_coexpression

using NearestNeighbors, CSV, DataFrames
using SparseArrays: SparseMatrixCSC, SparseVector, sparse
using ArgParse
using Distributions: Multinomial
using Statistics: mean


struct Transcript
    x::Float32
    y::Float32
    gene::Int
end


# Read transcripts from csv file.
function read_transcripts(
        filename::String,
        x_column::String,
        y_column::String,
        gene_column::String,
        cell_column::String,
        unassigned_value::Any,
        compartment_column::String,
        nuclear_value::Any)

    df = CSV.read(filename, DataFrame)

    cell_index = Dict{Any, Int}()

    gene_to_index = Dict{String, Int}()
    genes = String[]

    transcripts = Transcript[]
    nuc_assignments = Int[]

    for (x, y, gene, cell, compartment) in zip(
            df[!,x_column],
            df[!,y_column],
            df[!,gene_column],
            df[!,cell_column],
            df[!,compartment_column])

        if !haskey(gene_to_index, gene)
            push!(genes, gene)
            gene_to_index[gene] = length(genes)
        end

        if cell == unassigned_value || compartment != nuclear_value
            push!(nuc_assignments, 0)
        else
            push!(nuc_assignments, get!(() -> length(cell_index) + 1, cell_index, cell))
        end

        push!(transcripts, Transcript(x, y, gene_to_index[gene]))
    end

    return transcripts, genes, nuc_assignments
end


function compute_centroids(transcripts::Vector{Transcript}, nuc_assignments::Vector{Int})
    ncells = maximum(nuc_assignments)

    centroids = zeros(Float32, (ncells, 2))
    population = zeros(Int, ncells)

    for (transcript, cell) in zip(transcripts, nuc_assignments)
        if cell == 0
            continue
        end

        population[cell] += 1
        centroids[cell, :] += [transcript.x, transcript.y]
    end

    centroids ./= population

    return centroids
end


function compute_nuclear_expansion_assignments(
        transcripts::Vector{Transcript},
        centroids::Matrix{Float32}, radius::Float32)
    println("indexing centroids...")
    centroid_index = KDTree(transpose(centroids), leafsize=10)
    nucex_assignments = zeros(Int, length(transcripts))

    println("computing nuclear expansion assignments...")
    for (i, transcript) in enumerate(transcripts)
        cell, dist = nn(centroid_index, [transcript.x, transcript.y])
        if dist <= radius
            nucex_assignments[i] = cell
        end
    end

    return nucex_assignments
end


function count_matrix_from_assignments(
        transcripts::Vector{Transcript}, genes::Vector{String}, assignments::Vector{Int})

    ncells = maximum(assignments)
    ngenes = length(genes)

    counts = Dict{Tuple{Int, Int}, Int}()

    for (transcript, cell) in zip(transcripts, assignments)
        if cell == 0
            continue
        end
        key = (Int(cell), Int(transcript.gene))
        if haskey(counts, key)
            counts[key] += 1
        else
            counts[key] = 1
        end
    end

    Is = collect(first.(keys(counts)))
    Js = collect(last.(keys(counts)))
    Vs = collect(values(counts))

    return sparse(Is, Js, Vs, ncells, ngenes)
end


function downsample_counts(X::SparseMatrixCSC, target::Int)
	probs = X ./ sum(X, dims=2)

    rows = SparseVector[]
    for i in 1:size(probs, 1)
        probs_i = probs[i,:]
        # TODO: Why do this instead of just sampling without replacement? Was
        # this just an efficiency thing?
        counts_i = sparse(rand(Multinomial(target, probs_i)))
        push!(rows, counts_i)
    end

    return transpose(reduce(hcat, rows))
end

function counts_to_positivity(X, cutoff, min_pos_rate)
    positivity = X .>= cutoff
    positivity_rate = clamp.(mean(positivity, dims=1)[1,:], min_pos_rate, Inf)
    return (positivity, positivity_rate)
end

function conditional_coexpression(positivity, min_cond_coex)
	coex = transpose(positivity) * positivity
	cond_coex = coex ./ sum(positivity, dims=1)
	cond_coex[isnan.(cond_coex)] .= 0.0

    # Basically pseudocount to avoid zero numerators or denominators
    cond_coex = clamp.(cond_coex, min_cond_coex, Inf)

	return cond_coex
end


function censored_log_ratio(pos_rate_nuc, pos_rate_nucex, CCnuc, CCnucex, min_pos_rate)
	CCdiff = log2.(CCnucex ./ CCnuc)

    # exclude stuff that's barely expressed in one or the other
	CCdiff[(pos_rate_nuc .< min_pos_rate) .| (pos_rate_nucex .< min_pos_rate),:] .= 0
	CCdiff[:,(pos_rate_nuc .< min_pos_rate) .| (pos_rate_nucex .< min_pos_rate)] .= 0

    @assert all(isfinite.(CCdiff))

	return CCdiff
end

# TODO: Should we have default values for a lot of this? Or make keyword arguments.
function compute_relative_spurious_coexpression(
        filename::String,
        x_column::String,
        y_column::String,
        gene_column::String,
        cell_column::String,
        unassigned_value::Any,
        compartment_column::String,
        nuclear_value::Any;
        radius::Number=10.0,
        positivity_cutoff::Number=1,
        min_positivity_rate::Number=0.01,
        min_cond_coex::Number=0.05,
        mincount::Int=40,
        minposrate::Number=0.05)

    transcripts, genes, nuc_assignments = read_transcripts(
        filename, x_column, y_column, gene_column, cell_column, unassigned_value, compartment_column, nuclear_value)

    centroids = compute_centroids(transcripts, nuc_assignments)

    Xnuc = count_matrix_from_assignments(transcripts, genes, nuc_assignments)

    nucex_assignments = compute_nuclear_expansion_assignments(transcripts, centroids, Float32(radius))
    Xnucex = count_matrix_from_assignments(transcripts, genes, nucex_assignments)

    mask = (Array(sum(Xnuc, dims=2))[:,1] .>= mincount) .& (Array(sum(Xnucex, dims=2))[:,1] .>= mincount)

    Xnuc = downsample_counts(Xnuc[mask,:], mincount)
    Xnucex = downsample_counts(Xnucex[mask,:], mincount)

    pos_nuc, pos_rate_nuc = counts_to_positivity(Xnuc, positivity_cutoff, min_positivity_rate)
    pos_nucex, pos_rate_nucex = counts_to_positivity(Xnucex, positivity_cutoff, min_positivity_rate)

    CCnuc = conditional_coexpression(pos_nuc, min_cond_coex)
    CCnucex = conditional_coexpression(pos_nucex, min_cond_coex)

    CCdiff = censored_log_ratio(pos_rate_nuc, pos_rate_nucex, CCnuc, CCnucex, Float32(minposrate))
    @assert all(isfinite.(CCdiff))

    CCranked = sort(reshape([(CCdiff[i, j], genes[i], genes[j]) for i in 1:size(CCdiff, 1), j in 1:size(CCdiff, 2)], :), rev=true)

    @show CCranked[1:40]

    # TODO: What exactly do we want to report? We want to know the gene pairs
    # and also be able to easily compute median relatives spurious coexpression.
end

end # module SpuriousCoexpression
