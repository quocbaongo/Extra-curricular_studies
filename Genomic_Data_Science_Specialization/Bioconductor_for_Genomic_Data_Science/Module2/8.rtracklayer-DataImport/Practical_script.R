library(rtracklayer)
library(AnnotationHub)
ahub <- AnnotationHub()

# To view different types of data that you can connect to AnnotationHub() 
table(ahub$rdataclass)

# Let's take the BigWigFile
ahub.bw = subset(ahub, rdataclass == "BigWigFile" & species == "Homo sapiens")
bw = ahub.bw[[1]]

# bw is a pointer to a file
# To just read part of the use import()
#Here we just want to read in chromosome 22
gr.chr22 = import(bw, which=GRanges("chr22", ranges=IRanges(1,10^8)))

# To get GRanges as run length encoding vector
rle.chr22 = import(bw, which=GRanges("chr22", ranges=IRanges(1,10^8)), as = "Rle")

# To get run length encoding vector of chromosome 22
rle.chr22$chr22


# liftover tool: for converting genomic coordinates between different assemblies
# You have some coordinates for a particular version of a reference genome and
#you want to determine the corresponding coordinates on a different version 
#of the reference genome for that species

# Ex: you have a bed file with exon coordinates for human build GRC37 (hg19)
#and wish to update to GRCh38.

# LiftOver tool is provided as part of rtracklayer
# In order to use LiftOver, we need st called chain file
# A chain file contains information about converting one specific genome
#to the other specific genome.

ahub.chain = subset(ahub, rdataclass == "ChainFile")

# Limit to human data only
ahub.chain = subset(ahub.chain, species == "Homo sapiens")
chain=query(ahub.chain, c("hg18", "hg19"))[[1]]

gr.chr22 = import(bw, which=GRanges("chr22", ranges=IRanges(1,10^8)))

# Now to convert from hg19 to hg18
gr.hg18 = liftOver(gr.chr22, chain)

class(gr.hg18)
# What come out is GRanges list

length(gr.hg18)
length(gr.chr22)

table(elementNROWS(gr.hg18))
