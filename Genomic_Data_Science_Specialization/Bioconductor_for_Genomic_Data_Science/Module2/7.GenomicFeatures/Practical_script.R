library(GenomicFeatures)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)

# txdb object contains information abt gene transcripts
#exons and coding sequences
txdb = TxDb.Hsapiens.UCSC.hg19.knownGene

# GRanges select small region on chromosome 1
gr <- GRanges(seqnames = "chr1", strand = "+", ranges=IRanges(start=11874, end=14409))

# giving starts and ends of genes on "txdb" object
genes(txdb)

# Inspecting which genes overlap with the genomic range "gr"
subsetByOverlaps(genes(txdb), gr)

# There are 3 transcripts but transcripts have same start and same end
#It is because all we see here is the start and end of pre-mRNA.
#These three transcripts are different bc they have different exons
#But we cant see that in the output.
subsetByOverlaps(transcripts(txdb), gr)

# => It gonna be easier if we look at exon
# We will see 6 exons, and those exons are combined in different ways
#to form the three different transcripts above
subsetByOverlaps(exons(txdb), gr)

# To figure out how the exons are combined together to 
#form transcript.
subsetByOverlaps(exonsBy(txdb, by = "tx"), gr)

subsetByOverlaps(cds(txdb), gr)
subsetByOverlaps(cdsBy(txdb, by="tx"), gr)

subsetByOverlaps(exonsBy(txdb, by="tx"), gr)["2"]

# Function "transcriptLengths" gives out the lengths of different transcripts
