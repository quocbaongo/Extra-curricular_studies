# Library "Biostrings" aims to manipulate biological string e.g. DNA, RNA, etc
library(Biostrings)

dna1 = DNAString("ACGT-G")
dna2=DNAStringSet(c("ACG", "ACGT", "ACGTT"))

# IUPAC code is the way to represent string. For instance: letter "M" means "AC"
print(IUPAC_CODE_MAP)

# DNA string
print(dna1[2:4])

# DNA set
print(dna2[1:2])
print(dna2[[1]])

# We can put a name for each DNA in DNA string set
names(dna2) = paste0("seq", 1:3)

print(width(dna2))
print(sort(dna2))


# Reverse a single dna string
rev(dna1)

# Reverse each dna string within dna strings set
reverse(dna2)

# Reverse Complement
reverseComplement(dna2)

translate(dna2)

# For each dna string within the set, it illustrates how many occurrences of each letter
alphabetFrequency(dna2)
letterFrequency(dna2, letters="GC")
dinucleotideFrequency(dna2)
consensusMatrix(dna2)




