from bm_preproc import BoyerMoore
import bisect

def boyer_moore_with_counts(p, p_bm, t):
    """ Do Boyer-Moore matching. p=pattern, t=text,
        p_bm=BoyerMoore object for p """
    i = 0
    occurrences = []
    num_alignments = 0 
    num_character_comparisons = 0
    
    while i < len(t) - len(p) + 1:
        shift = 1
        mismatched = False
        num_alignments += 1
        for j in range(len(p)-1, -1, -1):	# j iterate in 3 -> 2 -> 1 -> 0
            num_character_comparisons += 1
            if p[j] != t[i+j]:
                skip_bc = p_bm.bad_character_rule(j, t[i+j])
                skip_gs = p_bm.good_suffix_rule(j)
                shift = max(shift, skip_bc, skip_gs)
                mismatched = True
                break
        if not mismatched:
            occurrences.append(i)
            skip_gs = p_bm.match_skip()
            shift = max(shift, skip_gs)
        i += shift
    return occurrences, num_alignments, num_character_comparisons

def naive_with_counts(p, t):
    occurrences = []
    num_alignments = 0 
    num_character_comparisons = 0
    
    for i in range(len(t) - len(p) + 1):  # loop over alignments
        num_alignments += 1
        match = True
        for j in range(len(p)):  # loop over characters
            num_character_comparisons += 1
            if t[i+j] != p[j]:  # compare characters
                match = False
                break
        if match:
            occurrences.append(i)  # all chars matched; record
    return occurrences, num_alignments, num_character_comparisons

def readGenome(filename):
    genome = ''
    with open(filename, 'r') as f:
        for line in f:
            # ignore header line with genome information
            if not line[0] == '>':
                genome += line.rstrip()
    return genome

class Index(object):
    def __init__(self, t, k):
        ''' Create index from all substrings of size 'length' '''
        self.k = k  # k-mer length (k)
        self.index = []
        for i in range(len(t) - k + 1):  # for each k-mer
            self.index.append((t[i:i+k], i))  # add (k-mer, offset) pair
        self.index.sort()  # alphabetize by k-mer
    
    def query(self, p):
        ''' Return index hits for first k-mer of P '''
        kmer = p[:self.k]  # query with first k-mer
        i = bisect.bisect_left(self.index, (kmer, -1))  # binary search
        hits = []
        while i < len(self.index):  # collect matching index entries
            if self.index[i][0] != kmer:
                break
            hits.append(self.index[i][1])
            i += 1
        return hits

def queryIndex_approximate_match(p, t, n, index):
    # 8-mer index, p is 24, and we allow up to two mismatches
    # Divide p into 3 segments, so at least one of the segment will be an exact match

    segment_length = int(round(len(p) / (n+1)))
    all_matches = set()
    indexhits = 0
    for i in range(n+1):
        # split p into n+1 segments
        start = i*segment_length
        end = min((i+1)*segment_length, len(p))
        matches = index.query(p[start:end])
        indexhits += len(matches)
        
        # Extend matching segments to see if whole p matches
        for m in matches:
            if m < start or m-start+len(p) > len(t):
                continue
            mismatches = 0
            for j in range(0, start):
                if not p[j] == t[m-start+j]:
                    mismatches += 1
                    if mismatches > n:
                        break
            for j in range(end, len(p)):
                if not p[j] == t[m-start+j]:
                    mismatches += 1
                    if mismatches > n:
                        break
            if mismatches <= n:
                all_matches.add(m - start)
    return list(all_matches), indexhits

def naive_nmm_with_count(p, t, n):
    occurrences = []
    num_alignments = 0
    num_character_comparisons = 0
    
    for i in range(len(t) - len(p) + 1):  # loop over alignments
        match = True
        mismatch_count = 0
        num_alignments += 1
        for j in range(len(p)):  # loop over characters
            num_character_comparisons += 1
            if t[i+j] != p[j]:  # compare characters
                mismatch_count += 1
                if mismatch_count > n:
                    match = False
                    break
        if match:
            occurrences.append(i)  # all chars matched; record

    return occurrences, num_alignments, num_character_comparisons

class SubseqIndex(object):
    """ Holds a subsequence index for a text T """
    
    def __init__(self, t, k, ival):
        """ Create index from all subsequences consisting of k characters
            spaced ival positions apart.  E.g., SubseqIndex("ATAT", 2, 2)
            extracts ("AA", 0) and ("TT", 1). """
        self.k = k  # num characters per subsequence extracted
        self.ival = ival  # space between them; 1=adjacent, 2=every other, etc
        self.index = []
        self.span = 1 + ival * (k - 1)
        for i in range(len(t) - self.span + 1):  # for each subseq
            self.index.append((t[i:i+self.span:ival], i))  # add (subseq, offset)
        self.index.sort()  # alphabetize by subseq
    
    def query(self, p):
        """ Return index hits for first subseq of p """
        subseq = p[:self.span:self.ival]  # query with first subseq
        i = bisect.bisect_left(self.index, (subseq, -1))  # binary search
        hits = []
        while i < len(self.index):  # collect matching index entries
            if self.index[i][0] != subseq:
                break
            hits.append(self.index[i][1])
            i += 1
        return hits

def approximate_match_subseq_index(p, t, k, ival):
    hits = 0
    all_matches = set()
    index = SubseqIndex(t, k, ival) # built on 8-mers and subsequence intervals of 3
    for start in range(k+1):
        matches = index.query(p[start:])
        hits += len(matches)
        for m in matches:
            mis_matches=0
            t_substring=t[m-start:(m-start)+len(p)]
            
            for j in range(len(p)):
                if p[j] != t_substring[j]:
                    mis_matches += 1
                    
                    if mis_matches > 2:
                        break
            if mis_matches <= 2:
                all_matches.add(m-start)
    return list(all_matches), hits

if __name__ == "__main__":

    # Load genome
    genome = readGenome("chr1.GRCh38.excerpt.fasta")

    # Question 1: How many alignments does the naive exact matching algorithm try when matching the string 
    #GGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG (derived from human Alu sequences) to the excerpt of human chromosome 1?  
    #(Don't consider reverse complements.)
    
    p='GGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG'
    occurrences, num_alignments, num_character_comparisons = naive_with_counts(p, genome)
    print(f"1. The number of alignment performed when using naive exact matching: {num_alignments}")
    
    # Question 2: How many character comparisons does the naive exact matching algorithm try when matching the string 
    #GGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG (derived from human Alu sequences) to the excerpt of human chromosome 1?  
    #(Don't consider reverse complements.)
    
    p='GGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG'
    occurrences, num_alignments, num_character_comparisons = naive_with_counts(p, genome)
    print(f"2. The number of character comparisons performed when using naive exact matching: {num_character_comparisons}")

    # Question 3: How many alignments does Boyer-Moore try when matching the string 
    #GGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG 
    #(derived from human Alu sequences) to the excerpt of human chromosome 1?  (Don't consider reverse complements.)
    
    p='GGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGG'
    p_bm = BoyerMoore(p)
    occurrences, num_alignments, num_character_comparisons = boyer_moore_with_counts(p, p_bm, genome)
    print(f"3. The number of alignments by Boyer-Moore try when matching the string: {num_alignments}")
    
    # Question 4: How many times does the string GGCGCGGTGGCTCACGCCTGTAAT, which is derived from a human Alu sequence, 
    #occur with up to 2 substitutions in the excerpt of human chromosome 1?  (Don't consider reverse complements here.)
    
    p='GGCGCGGTGGCTCACGCCTGTAAT'
    index = Index(genome,8)
    matches, indexhits=queryIndex_approximate_match(p, genome, 2, index)
    print(f"4. Number of times, in which the inspected string occur with up to 2 substitutions in the excerpt of human chromosome 1: {len(matches)}")
    
    # Question 5: Using the instructions given in Question 4, how many total index hits are there when searching for occurrences of  
    #GGCGCGGTGGCTCACGCCTGTAAT with up to 2 substitutions in the excerpt of human chromosome 1?
    
    p='GGCGCGGTGGCTCACGCCTGTAAT'
    index = Index(genome,8)
    matches, indexhits=queryIndex_approximate_match(p, genome, 2, index)
    print(f"5. Number of total index hits, in which the inspected string occur with up to 2 substitutions in the excerpt of human chromosome 1: {indexhits}")    


    # Question 6: Write a function that, given a length-24 pattern P and given a SubseqIndex 
    #with k = 8 and ival = 3, finds all approximate occurrences of P within T with up to 2 mismatches.
    #
    #When using this function, how many total index hits are there when searching for 
    #GGCGCGGTGGCTCACGCCTGTAAT with up to 2 substitutions in the excerpt of human chromosome 1?  
    #(Again, don't consider reverse complements.)

    p='GGCGCGGTGGCTCACGCCTGTAAT'
    all_matches, indexhits = approximate_match_subseq_index(p, genome, 8, 3)
    print(f"6. Number of total index hits, in which the inspected string occur with up to 2 substitutions in the excerpt of human chromosome 1: {indexhits}")
        


