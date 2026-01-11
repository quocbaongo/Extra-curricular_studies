# Loading yeast genome version 2
library("BSgenome.Scerevisiae.UCSC.sacCer2")

dnaseq <- DNAString("ACGTACGT")
vi = matchPattern(dnaseq, Scerevisiae$chrI)

ranges(vi)
Scerevisiae$chrI[57932:57939]

# To get alphabet frequency on the DNA strand: representing content of each base
alphabetFrequency(vi)

# Shift the view 10 bases to the right
shift(vi, 10)

# Match single dna sequence "dnaseq" against a collection of sequences "Scerevisiae"
gr = vmatchPattern(dnaseq, Scerevisiae)

vi2 = Views(Scerevisiae, gr)

# AnnotationHub
library(AnnotationHub)
ahub = AnnotationHub()

qh = query(ahub, c("sacCer2", "genes"))

qh
# Here, we get back two objects: "SGD Genes" (Stanford genome database) and "Ensembl Genes" (Ensembl genome database)
# We will take Stanford genome database
genes = ahub[["AH7048"]]

# Examining promoters
prom = promoters(genes)
prom

# To cut off anything outside of the length of genome
prom = trim(prom)

promViews = Views(Scerevisiae, prom)
promViews

Scerevisiae$chrI[128802:131001]

# GC content of promoters from Scerevisiae
gcProm = letterFrequency(promViews, "GC", as.prob = TRUE)
gcProm

plot(density(gcProm))
abline(v=0.38)









