library(GenomicRanges)

# Constructing two genomic Ranges
gr1 <- GRanges(seqnames = "chr1", ranges = IRanges(start = 1:4, width = 3))
gr2 <- GRanges(seqnames = "chr2", ranges = IRanges(start = 1:4, width = 3))

# Create GRanges list
gL = GRangesList(gr1 = gr1, gr2 = gr2)

gL[1]
gL[[1]]
gL$gr1

# Now we will get a list, in which for each element, we will obtain the 1st value of IRanges
start(gL)

seqnames(gL)
elementLengths(gL)
# The following command will generate a similar result as "elementLengths(gL)"
sapply(gL, length)

shift(gL, 10)

findOverlaps(gL, gr2)
