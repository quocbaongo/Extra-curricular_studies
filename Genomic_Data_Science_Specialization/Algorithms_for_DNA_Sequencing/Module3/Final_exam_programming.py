# Dynamic programming to find approximate occurences of a pattern in a text.
#	. Rows of the dynamic programming matrix are labeled with bases from P and columns with bases from T
#	. Elements in the 1st row are set to 0
#	. Elements in the 1st column are set to 0, 1, 2, ... as for edit distance
#	. Other elements are set in the same way as elements of a standard edit distance matrix
#	. The minimal value in the bottom row: edit distance of the closest match between P and T

import time
import bisect
from itertools import permutations

def readGenome(filename):
    genome = ''
    with open(filename, 'r') as f:
        for line in f:
            # ignore header line with genome information
            if not line[0] == '>':
                genome += line.rstrip()
    return genome
    
def readFastq(filename):
    sequences = []
    qualities = []
    with open(filename) as fh:
        while True:
            fh.readline()  # skip name line
            seq = fh.readline().rstrip()  # read base sequence
            fh.readline()  # skip placeholder line
            qual = fh.readline().rstrip() # base quality line
            if len(seq) == 0:
                break
            sequences.append(seq)
            qualities.append(qual)
    return sequences, qualities

class Index(object):
    """ Holds a substring index for a text T """

    def __init__(self, t, k):
        """ Create index from all substrings of t of length k """
        self.k = k  # k-mer length (k)
        self.index = []
        for i in range(len(t) - k + 1):  # for each k-mer
            self.index.append((t[i:i+k], i))  # add (k-mer, offset) pair
        self.index.sort()  # alphabetize by k-mer

    def query(self, p):
        """ Return index hits for first k-mer of p """
        kmer = p[:self.k]  # query with first k-mer
        i = bisect.bisect_left(self.index, (kmer, -1))  # binary search
        hits = []
        while i < len(self.index):  # collect matching index entries
            if self.index[i][0] != kmer:
                break
            hits.append(self.index[i][1])
            i += 1
        return hits
    
def editDistance(x, y):
    # Create distance matrix
    D = []
    for i in range(len(x)+1):
        D.append([0]*(len(y)+1))
    # Initialize first row and column of matrix
    for i in range(len(x)+1):
        D[i][0] = i

    # Fill in the rest of the matrix
    for i in range(1, len(x)+1):
        for j in range(1, len(y)+1):
            distHor = D[i][j-1] + 1
            distVer = D[i-1][j] + 1
            if x[i-1] == y[j-1]:
                distDiag = D[i-1][j-1]
            else:
                distDiag = D[i-1][j-1] + 1
            D[i][j] = min(distHor, distVer, distDiag)
    # Edit distance is the value in the bottom right corner of the matrix
    return min(D[-1])

def overlap(a, b, min_length=3):
    """ Return length of longest suffix of 'a' matching
        a prefix of 'b' that is at least 'min_length'
        characters long.  If no such overlap exists,
        return 0. """
    start = 0  # start all the way at the left
    while True:
        start = a.find(b[:min_length], start)  # look for b's prefix in a
        if start == -1:  # no more occurrences to right
            return 0
        # found occurrence; check for full suffix/prefix match
        if b.startswith(a[start:]):
            return len(a)-start
        start += 1  # move just past previous match
    
def overlap_all_pairs(reads, k):    

    kmer_index={}
    
    for read in reads:
        index_per_sequence=Index(read, k)
        
        for kmer in index_per_sequence.index:
            if kmer[0] not in kmer_index:
                kmer_index[kmer[0]] = set()
            kmer_index[kmer[0]].add(read)

    # Finding overlap
    overlap_pairs=[]
    for a,b in permutations(reads, 2):
        if b not in kmer_index[a[-k:]]:
            continue
        else:
            olen = overlap(a,b,min_length=k)
                
            if olen > 0:
                overlap_pairs.append((a,b))

    return overlap_pairs
    
    
if __name__ == "__main__":

    # Load genome
    genome = readGenome("chr1.GRCh38.excerpt.fasta")
    
    
    # Question 1: What is the edit distance of the best match
    #pattern GCTGATCGATCGTACG and 
    #the excerpt of human chromosome 1?  (Don't consider reverse complements.)
    
    edit_dist=editDistance("GCTGATCGATCGTACG", genome)
    print(f"1. The edit distance of the best match between pattern 'GCTGATCGATCGTACG' and the excerpt of human chromosome 1: {edit_dist}")
    print()

    # Question 2: What is the edit distance of the best match between 
    #pattern GATTTACCAGATTGAG and 
    #the excerpt of human chromosome 1
    
    edit_dist=editDistance("GATTTACCAGATTGAG", genome)
    print(f"2. The edit distance of the best match between pattern 'GATTTACCAGATTGAG' and the excerpt of human chromosome 1: {edit_dist}")
    print()


    # Question 3: Download and parse the read sequences from the provided Phi-X FASTQ file. We'll just use their base sequences, so you can ignore read names and base qualities.  
    #Also, no two reads in the FASTQ have the same sequence of bases.  This makes things simpler.
    #
    #https://d28rh4a8wq0iu5.cloudfront.net/ads1/data/ERR266411_1.for_asm.fastq
    #
    #Next, find all pairs of reads with an exact suffix/prefix match of length at least 30. Don't overlap a read with itself; 
    #if a read has a suffix/prefix match to itself, ignore that match. Ignore reverse complements.
    
    # Hint 1: Your function should not take much more than 15 seconds to run on this 10,000-read dataset, and maybe much less than that.  
    #(Our solution takes about 3 seconds.) If your function is much slower, there is a problem somewhere.
    #Hint 2: Remember not to overlap a read with itself. If you do, your answers will be too high.
    # Hint 3: You can test your implementation by making up small examples, then checking that (a) your implementation runs quickly, and 
    #(b) you get the same answer as if you had simply called overlap(a, b, min_length=k)
    
    #Picture the overlap graph corresponding to the overlaps just calculated.  How many edges are in the graph?  
    #In other words, how many distinct pairs of reads overlap?

    # Parse FASTQ file
    sequences, qualities = readFastq("ERR266411_1.for_asm.fastq")
    overlap_pairs = overlap_all_pairs(sequences, 30)
    
    print(f"3. The number of pairs of reads with an exact suffix/prefix match of length at least 30 is: {len(overlap_pairs)}")
    
    
    # Question 4: Picture the overlap graph corresponding to the overlaps computed for the previous question. 
    #How many nodes in this graph have at least one outgoing edge?  
    #(In other words, how many reads have a suffix involved in an overlap?)
    
    number_of_reads=set()
    
    for pair in overlap_pairs:
        number_of_reads.add(pair[0])

    print(f"4. The number of reads that have a suffix involved in an overlap is: {len(number_of_reads)}")
