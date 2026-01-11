import itertools
import time

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

def overlap(a, b, min_length=3):
    """ Return length of longest suffix of 'a' matching
        a prefix of 'b' that is at least 'min_length'
        characters long.  If no such overlap exists,
        return 0. """
    start = 0  # start all the way at the left
    while True:
        start = a.find(b[:min_length], start)  # look for b's suffx in a
        if start == -1:  # no more occurrences to right
            return 0
        # found occurrence; check for full suffix/prefix match
        if b.startswith(a[start:]):
            return len(a)-start
        start += 1  # move just past previous match

def scs(ss):
    """ Returns shortest common superstring of given
        strings, which must be the same length """
    shortest_sup = None
    for ssperm in itertools.permutations(ss):
        sup = ssperm[0]  # superstring starts as first string
        for i in range(len(ss)-1):
            # overlap adjacent strings A and B in the permutation
            olen = overlap(ssperm[i], ssperm[i+1], min_length=1)
            # add non-overlapping portion of B to superstring
            sup += ssperm[i+1][olen:]    
        if shortest_sup is None or len(sup) < len(shortest_sup):
            shortest_sup = sup  # found shorter superstring
    return shortest_sup  # return shortest

def modified_scs(ss):
    """ Returns shortest common superstring of given
        strings, which must be the same length """
    scs={}
    for ssperm in itertools.permutations(ss):
        sup = ssperm[0]  # superstring starts as first string
        for i in range(len(ss)-1):
            # overlap adjacent strings A and B in the permutation
            olen = overlap(ssperm[i], ssperm[i+1], min_length=1)
            # add non-overlapping portion of B to superstring
            sup += ssperm[i+1][olen:]
        
        if len(sup) not in scs:
            scs.update({len(sup): [sup]})
        else:
            scs[len(sup)].append(sup)

    shortest_sups=scs[min(scs.keys())]
    return shortest_sups  # return shortest common superstrings

def pick_maximal_overlap(reads, k):

    kmer_index={}
    for read in reads:
        index_per_sequence=Index(read, k)
        
        for kmer in index_per_sequence.index:
            if kmer[0] not in kmer_index:
                kmer_index[kmer[0]] = set()
            kmer_index[kmer[0]].add(read)
    
    reada, readb = None, None
    best_olen = 0
    for a,b in itertools.permutations(reads, 2):
        if b not in kmer_index[a[-k:]]:
            continue
        else:
            olen = overlap(a, b, k)
            if olen > best_olen:
                reada, readb = a, b
                best_olen = olen
                
    return reada, readb, best_olen
    
def greedy_scs(reads, k):
    read_a, read_b, olen = pick_maximal_overlap(reads, k)
    while olen > 0:
        reads.remove(read_a)
        reads.remove(read_b)
        reads.append(read_a + read_b[olen:])
        read_a, read_b, olen = pick_maximal_overlap(reads, k)
    return ''.join(reads)


if __name__ == "__main__":


    # Question 1: It's possible for there to be multiple different shortest common superstrings for the same set of input strings. 
    #Consider the input strings "ABC", "BCA" and "CAB". One shortest common superstring is "ABCAB"
    #but another is "BCABC"
    #and another is "CABCA"
    #What is the length of the shortest common superstring of the following strings?
    #"CCT", "CTT", "TGC", "TGG", "GAT", "ATT"

    shortest_common_superstring=scs(["CCT", "CTT", "TGC", "TGG", "GAT", "ATT"])
    print(f"1. The length of the shortest common superstring of the strings including 'CCT', 'CTT', 'TGC', 'TGG', 'GAT', 'ATT' is {len(shortest_common_superstring)}")
    print()
    
    # Question 2: How many different shortest common superstrings are there for the input strings given in the previous question?
    #Hint: You can modify the 'scs' function to keep track of this.
    
    shortest_common_superstrings=modified_scs(["CCT", "CTT", "TGC", "TGG", "GAT", "ATT"])
    print(f"2. The number of different shortest common superstrings given the input strings including 'CCT', 'CTT', 'TGC', 'TGG', 'GAT', 'ATT' is {len(shortest_common_superstrings)}")
    print()
    
    # Question 3: Download this FASTQ file containing synthetic sequencing reads from a mystery virus:
    #https://d28rh4a8wq0iu5.cloudfront.net/ads1/data/ads1_week4_reads.fq
    #All the reads are the same length (100 bases) and 
    #are exact copies of substrings from the forward strand of the virus genome.  
    #You don't have to worry about sequencing errors, ploidy, or reads coming from the reverse strand.

    #Assemble these reads using one of the approaches discussed, such as greedy shortest common superstring.  
    #Since there are many reads, you might consider ways to make the algorithm faster.
    #How many As are there in the full, assembled genome?
    #Hint: the virus genome you are assembling is exactly 15,894 bases long

    # Parse FASTQ file
    sequences, qualities = readFastq("ads1_week4_reads.fq")
    
    #print(scs(sequences))
    start_time = time.time()
    assembled_genome=greedy_scs(sequences, 10)
    number_of_As=assembled_genome.count("A")
    print(f"3. Using the greedy shortest common superstring, the number of As in the full, assembled genome is {number_of_As}, and the exection time of the is {round(time.time() - start_time, 2)} seconds")
    print()
    
    # Question 4: How many Ts are there in the full, assembled genome from the previous question?
    
    number_of_Ts=assembled_genome.count("T")
    print(f"4. The number of Ts in the full, assembled genome is {number_of_Ts}")

























