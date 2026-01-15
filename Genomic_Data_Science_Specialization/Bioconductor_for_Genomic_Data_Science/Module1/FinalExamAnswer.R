# Load the AnnotationHub package without startup messages
suppressPackageStartupMessages(library(AnnotationHub))

# Create the AnnotationHub object without printing messages
ahub <- suppressMessages(AnnotationHub())

# Q1: Use the AnnotationHub package to obtain data on "CpG islands" in the human genome.
#How many islands exists on the autosomes (any chromosome that is not a sex chromosome)

ahub = suppressMessages(subset(ahub, species == "Homo sapiens"))
ahub_cpG <- suppressMessages(query(ahub, c("CpG Islands", "hg19"))[["AH5086"]])
autosome <- paste0("chr", 1:22)
ahub_cpG_autosomes <- suppressMessages(ahub_cpG[seqnames(ahub_cpG) %in% autosome])

# CpG Islands on autosome
print(paste("Question 1. The number of CpG Islands that exists on the autosomes is", length(ahub_cpG_autosomes)))

# Output: Question 1. The number of CpG Islands that exists on the autosomes is 26641

# Q2: How many CpG Islands exists on chromosome 4?
ahub_cpG_chr4 <- suppressMessages(ahub_cpG[seqnames(ahub_cpG) == "chr4"])
print(paste("Question 2. The number of CpG Islands that exists on chromsome 4 is", length(ahub_cpG_chr4)))

# Output: Question 2. The number of CpG Islands that exists on chromsome 4 is 1031

# Q3: Obtain the data for the H3K3me3 histone modification for the H1 cell line from Epigenomics Roadmap, using AnnotationHub.
#Subset these regions to only keep regions mapped to the autosomes (chromsome 1 to 22)
#Question: How many bases does these regions cover?

# Create the AnnotationHub object without printing messages
ahub <- suppressMessages(AnnotationHub())
ahub_H3K4me3 <- suppressMessages(query(ahub, c("H3K4me3", "E003"))[["AH29884"]])
autosome <- paste0("chr", 1:22)

ahub_H3K4me3_autosomes <- ahub_H3K4me3[seqnames(ahub_H3K4me3) %in% autosome]
total_bases <- sum(width(ahub_H3K4me3_autosomes))

print(paste("Question 3. The number of bases these regions cover is", total_bases))

# Output: Question 3. The number of bases these regions cover is 41135164

# Q4: H3K27me3 histone modification for the H1 cell line from Epigenomics Roadmap
get_roadmap_data <- function(ah_id) {
    ah <- suppressPackageStartupMessages(AnnotationHub())
    
    # 1. Get the URL but DO NOT load the object
    resource_url <- ah[ah_id]$sourceurl
    
    # 2. Download to a temp file
    temp_path <- tempfile(fileext = ".narrowPeak.gz")
    download.file(resource_url, temp_path, quiet = TRUE)
    
    # 3. Read as text and fix the strand
    # 'fill=TRUE' handles ragged rows (the second error you saw)
    raw_df <- read.table(temp_path, sep="\t", fill=TRUE, stringsAsFactors=FALSE)
    
    # Fix: Swap '.' for '*' (Bioconductor's code for unstranded)
    raw_df[, 6] <- sub("\\.", "*", raw_df[, 6])
    
    # 4. Convert to GRanges
    # Note: NarrowPeak is 0-based start, GRanges is 1-based start
    gr <- GRanges(
        seqnames = raw_df[, 1],
        ranges   = IRanges(start = raw_df[, 2] + 1, end = raw_df[, 3]),
        strand   = raw_df[, 6],
        score    = raw_df[, 5],
        name     = raw_df[, 4]
    )
    
    # Add metadata columns (signal, pValue, qValue, peak)
    mcols(gr) <- raw_df[, 7:10]
    colnames(mcols(gr)) <- c("signalValue", "pValue", "qValue", "peak")
    
    return(gr)
}

#ahub_H3K27me3 <- query(ahub, c("H3K27me3", "E003"))
ahub_H3K27me3_data <- get_roadmap_data("AH29892")

ahub_H3K27me3_data_autosome <- ahub_H3K27me3_data[seqnames(ahub_H3K27me3_data) %in% autosome]

meanValue <- mean(ahub_H3K27me3_data_autosome$signalValue)
meanValue <- round(meanValue, digits = 5)

print(paste("Question 4. The mean signalValue across all regions on the standard chromosomes is", meanValue))

# Output: Question 4. The mean signalValue across all regions on the standard chromosomes is 4.77073

# Q5: Bivalent regions are bound by both H3K4me3 and H3K27me3
bivalent_regions <- intersect(ahub_H3K4me3_autosomes, ahub_H3K27me3_data_autosome)

print(paste("Question 5. The number of bases on the standard chromosomes that are bivalently marked is", sum(width(bivalent_regions))))

# Output: Question 5. The number of bases on the standard chromosomes that are bivalently marked is 10289096

# Q6: how big a fraction (expressed as a number between 0 and 1) of the bivalent regions, overlap one or more CpG Islands?
ov <- findOverlaps(bivalent_regions, ahub_cpG_autosomes)
fraction <- length(unique(queryHits(ov))) / length(bivalent_regions)

print(paste("Question 6. The fraction of the bivalent regions overlapping with one or more CpG Islands is", fraction))

# Output: Question 6. The fraction of the bivalent regions overlapping with one or more CpG Islands is 0.538364439635741

# Q7: How big a fraction (expressed as a number between 0 and 1) of the bases, which are part of CpG Islands, are also bivalent marked.  
Ov_CpG_bivalent_marked <- intersect(ahub_cpG_autosomes, bivalent_regions)
fraction_overlapping <- sum(width(Ov_CpG_bivalent_marked)) / sum(width(ahub_cpG_autosomes))

print(paste("Question 7. A fraction of the bases, which are part of CpG Islands and also bivalent marked is", fraction_overlapping))

# Output: Question 7. A fraction of the bases, which are part of CpG Islands and also bivalent marked is 0.241687978531942

# Q8: How many bases are bivalently marked within 10kb of CpG Islands?
big_islands <- resize(ahub_cpG_autosomes, width = 20000 + width(ahub_cpG_autosomes), fix = "center")

CpG_10k_bivalent <- intersect(bivalent_regions, big_islands)

print(paste("Question 8. The number of bases that are bivalently marked within 10kb of CpG Islands is", sum(width(CpG_10k_bivalent))))

# Output: Question 8. The number of bases that are bivalently marked within 10kb of CpG Islands is 9782086

# Q9: Fraction of CpG
#How big a fraction (expressed as a number between 0 and 1) of the human genome is contained in a CpG Island?

genome <- query(ahub, "RefSeq")
genome <- genome[["AH5040"]]

ratio <- sum(width(ahub_cpG_autosomes)) / sum(as.numeric(seqlengths(genome)[1:22]))
print(paste("Question 9. The fraction of the human genome that is contained in a CpG Island is", ratio))

# Output: Question 9. The fraction of the human genome that is contained in a CpG Island is 0.00704748053369072

# Q10: Compute an odds-ratio for the overlap of bivalent marks with CpG islands.
# odds ratio
genome_size <- sum(as.numeric(seqlengths(genome)))
inOut = matrix(0, ncol = 2, nrow = 2)
colnames(inOut) = c("in", "out")
rownames(inOut) = c("in", "out")

# inOut
inOut[1,1] = sum(width(intersect(bivalent_regions, 
                                 ahub_cpG_autosomes,
                                 ignore.strand=TRUE)))
inOut[1,2] = sum(width(setdiff(bivalent_regions, 
                               ahub_cpG_autosomes,
                               ignore.strand=TRUE)))
inOut[2,1] = sum(width(setdiff(ahub_cpG_autosomes, 
                               bivalent_regions, 
                               ignore.strand=TRUE)))
inOut[2,2] = genome_size - sum(inOut)

odd_ratio <- inOut[1,1]*inOut[2,2]/(inOut[1,2]*inOut[2,1])

print(paste("Question 10. Odds-ratio for the overlap of bivalent marks with CpG islands is", odd_ratio))

# Output: Question 10. Odds-ratio for the overlap of bivalent marks with CpG islands is 184.264351663869
