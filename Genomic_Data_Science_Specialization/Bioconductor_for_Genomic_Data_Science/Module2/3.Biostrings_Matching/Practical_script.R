# Loading yeast genome version 2
library("BSgenome.Scerevisiae.UCSC.sacCer2")

dnaseq <- DNAString("ACGTACGT")
dnaseq

# Output of "Scerevisiae$chrI" is a DNA sequence
#230208-letter DNAString object
#seq: CCACACCACACCCACACACCCACACACCACACCACA...GGTGTGTGGGTGTGGTGTGGGTGTGGTGTGTGTGGG

# Matching a single string to another single string
matchPattern(dnaseq, Scerevisiae$chrI)

# To count how many match we have
countPattern(dnaseq, Scerevisiae$chrI)

# To match one single sequence against multiple sequences; output is an GRanges
vmatchPattern(dnaseq, Scerevisiae)

# To check if the 'dnaseq' is equivalent to its own reverse complement
dnaseq == reverseComplement(dnaseq)


# Some useful functions:
# A position weight matrix (PWM): representation of motifs (patterns) in biological sequences.
# PWMs: often derived from a set of aligned sequences that are thought to be functionally related. PWMs allow us to search the genome e.g. for binding sites for a given transcription factor.
matchPWM()

# For pairwise alignment
pairwiseAlignment()

# For trimming off a specific pattern on the left or right of a set of DNA
trimLRPatterns()
