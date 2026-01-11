# BSgenome package aims to represent the full genome in bioconductor
library(BSgenome)

# List all the genome
available.genomes()

# Look at specific genome
# Download genome using 'BiocManager::install("BSgenome.Scerevisiae.UCSC.sacCer2")'
library("BSgenome.Scerevisiae.UCSC.sacCer2")

# The object pack's name is the name of species. In this case -> Scerevisiae
# Short name of genome object
Scerevisiae

# Get sequence name
seqnames(Scerevisiae)
# Get sequence length
seqlengths(Scerevisiae)

# DNA string of first chromosome
Scerevisiae$chrI

# Calculate GC content
letterFrequency(Scerevisiae$chrI, "GC")
# Displaying GC content in term of probability
letterFrequency(Scerevisiae$chrI, "GC", as.prob = TRUE)

# "param" is an object containing the function that we are going to apply  \
#and the object that we are going to apply to.
param = new("BSParams", X = Scerevisiae, FUN = letterFrequency)

bsapply(param, "GC")	# Get GC number of each chromosome
unlist(bsapply(param, "GC"))

sum(unlist(bsapply(param, "GC"))) / sum(seqlengths(Scerevisiae))
unlist(bsapply(param, "GC", as.prob = TRUE))

