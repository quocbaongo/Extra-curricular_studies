from collections import Counter
import requests

def naive(p, t):
    occurrences = []
    for i in range(len(t) - len(p) + 1):  # loop over alignments
        match = True
        for j in range(len(p)):  # loop over characters
            if t[i+j] != p[j]:  # compare characters
                match = False
                break
        if match:
            occurrences.append(i)  # all chars matched; record
    return occurrences


def naive_2mm(p, t):
    occurrences = []
    for i in range(len(t) - len(p) + 1):  # loop over alignments
        match = True
        count=0
        for j in range(len(p)):  # loop over characters
            if t[i+j] != p[j]:  # compare characters
                count+=1
                
                if count > 2:
                    match = False
                    break
        if match:
            occurrences.append(i)  # all chars matched; record
    return occurrences


def reverseComplement(s):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    t = ''
    for base in s:
        t = complement[base] + t
    return t
    
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
    
def naive_with_rc(p, t):
    # Get the reverse complement of pattern p
    p_rc = reverseComplement(p)
    
    # Initialize list for storing occurrences
    occurrences = set()  # Using set to avoid duplicates when p == p_rc
    
    # Search for pattern p in text t
    for i in range(len(t) - len(p) + 1):
        match = True
        for j in range(len(p)):
            if t[i+j] != p[j]:
                match = False
                break
        if match:
            occurrences.add(i)

    # Search for reverse complement p_rc in text t
    for i in range(len(t) - len(p) + 1):
        match = True
        for j in range(len(p)):
            if t[i+j] != p_rc[j]:
                match = False
                break
        if match:
            occurrences.add(i)
    
    # Convert set back to sorted list for return
    return sorted(list(occurrences))

def phred33ToQ(qual):
    return ord(qual) - 33

if __name__ == "__main__":    

    # Load Lambda virus genome
    lambda_genome = readGenome("lambda_virus.fa")

    # Question 1: Count occurrences of AGGT or its reverse complement ACCT
    p1 = "AGGT"
    occ1 = naive_with_rc(p1, lambda_genome)
    print(f"1. Occurrences of AGGT or ACCT: {len(occ1)}")
    print()

    # Question 2: Count occurrences of TTAA (same as its reverse complement)
    p2 = "TTAA"
    occ2 = naive(p2, lambda_genome)  # No need for rc since TTAA is palindromic
    print(f"2. Occurrences of TTAA: {len(occ2)}")
    print()

    # Question 3: Leftmost occurrence of ACTAAGT or its reverse complement
    p3 = "ACTAAGT"
    occ3 = naive_with_rc(p3, lambda_genome)
    leftmost3 = min(occ3) if occ3 else -1
    print(f"3. Leftmost offset of ACTAAGT or its reverse complement: {leftmost3}")
    print()

    # Question 4: Leftmost occurrence of AGTCGA or its reverse complement
    p4 = "AGTCGA"
    occ4 = naive_with_rc(p4, lambda_genome)
    leftmost4 = min(occ4) if occ4 else -1
    print(f"4. Leftmost offset of AGTCGA or its reverse complement: {leftmost4}")
    print()

    # Question 5: Count TTCAAGCC with up to 2 mismatches
    p5 = "TTCAAGCC"
    occ5 = naive_2mm(p5, lambda_genome)
    print(f"5. Occurrences of TTCAAGCC with up to 2 mismatches: {len(occ5)}")
    print()

    # Question 6: Leftmost occurrence of AGGAGGTT with up to 2 mismatches
    p6 = "AGGAGGTT"
    occ6 = naive_2mm(p6, lambda_genome)
    leftmost6 = min(occ6) if occ6 else -1
    print(f"6. Leftmost offset of AGGAGGTT with up to 2 mismatches: {leftmost6}")
    print()

    # Question 7: Analyze FASTQ file for bad sequencing cycle
    # Download the FASTQ file if not already present
    url = "https://d28rh4a8wq0iu5.cloudfront.net/ads1/data/ERR037900_1.first1000.fastq"
    response = requests.get(url)
    with open("ERR037900_1.first1000.fastq", "w") as f:
        f.write(response.text)

    # Parse FASTQ file
    sequences, qualities = readFastq("ERR037900_1.first1000.fastq")
    print(f"Number of reads: {len(sequences)}")

    # Calculate average quality score per position
    score_by_pos = {}
    for quality in qualities:
        phred_scores = [phred33ToQ(q) for q in quality if len(quality) > 0]
        for i in range(len(phred_scores)):
            if i not in score_by_pos:
                score_by_pos[i] = [phred_scores[i]]
            else:
                score_by_pos[i].append(phred_scores[i])

    # Compute average quality per position and find the lowest
    avg_scores = {pos: sum(scores) / len(scores) for pos, scores in score_by_pos.items()}
    bad_cycle = min(avg_scores, key=avg_scores.get)
    print(f"7. Sequencing cycle with poor quality: {bad_cycle} (Avg Q-score: {avg_scores[bad_cycle]:.2f})")
    
    
    
    
    
    
    
