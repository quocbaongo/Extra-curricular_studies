#!/usr/bin/env python3

import argparse
from Bio import SeqIO
from collections import Counter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive DNA sequence analysis for multi-FASTA files.')
    parser.add_argument('--fasta_file', required=True, help='Path to the FASTA file')
    parser.add_argument('--frame', type=int, choices=[1, 2, 3], default=1, 
                        help='Reading frame for ORF analysis (1, 2, or 3)')
    parser.add_argument('--repeat_length', type=int, default=5, 
                        help='Length of repeats to analyze')
    return parser.parse_args()

def read_fasta_to_dict(fasta_file):
    """Read a FASTA file into a dictionary."""
    fasta_dict = {}
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            fasta_dict[record.id] = str(record.seq)
        return fasta_dict
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return {}

# Question 1: Record Counting
def count_records(fasta_dict):
    """Count the number of records."""
    return len(fasta_dict)

# Question 2: Sequence Length Analysis
def analyze_sequence_lengths(fasta_dict):
    """Analyze sequence lengths."""
    if not fasta_dict:
        return [], 0, [], 0, 0, [], 0
    all_lengths = {seq_id: len(seq) for seq_id, seq in fasta_dict.items()}
    lengths_list = list(all_lengths.values())
    longest_length = max(lengths_list)
    shortest_length = min(lengths_list)
    longest_ids = [seq_id for seq_id, length in all_lengths.items() if length == longest_length]
    shortest_ids = [seq_id for seq_id, length in all_lengths.items() if length == shortest_length]
    num_longest = len(longest_ids)
    num_shortest = len(shortest_ids)
    return (all_lengths, longest_length, longest_ids, num_longest, 
            shortest_length, shortest_ids, num_shortest)

# Question 3: ORF Analysis
def find_orfs_in_frame(sequence, frame):
    """Find ORFs in a specific reading frame."""
    sequence = sequence.upper()
    orfs = []
    start_pos = frame - 1
    seq_in_frame = sequence[start_pos:]
    for i in range(0, len(seq_in_frame) - 2, 3):  # Ensure codon length
        codon1 = seq_in_frame[i:i+3]
        if codon1 == "ATG":
            position1 = i
            for j in range(position1, len(seq_in_frame) - 2, 3):
                codon2 = seq_in_frame[j:j+3]
                if codon2 in ["TAA", "TAG", "TGA"]:
                    position2 = j
                    orfs.append((position1 + 1 + start_pos, position2 - position1 + 3))
                    break
    return orfs

def analyze_orfs(fasta_dict, frame):
    """Analyze ORFs across all sequences for a given frame."""
    results = {
        'all_orfs': {},
        'longest_orf': 0,
        'longest_orf_seq_id': None,
        'longest_orf_start': None,
        'seq_specific_longest': {}
    }
    for seq_id, sequence in fasta_dict.items():
        orfs = find_orfs_in_frame(sequence, frame)
        results['all_orfs'][seq_id] = orfs
        if orfs:
            longest_in_seq = max(orfs, key=lambda x: x[1])
            results['seq_specific_longest'][seq_id] = longest_in_seq
            if longest_in_seq[1] > results['longest_orf']:
                results['longest_orf'] = longest_in_seq[1]
                results['longest_orf_seq_id'] = seq_id
                results['longest_orf_start'] = longest_in_seq[0]
    return results

# Question 4: Repeat Analysis
def find_repeats_in_sequence(sequence, n):
    """Find all repeats of length n in a sequence."""
    sequence = sequence.upper()
    repeats = {}
    for i in range(len(sequence) - n + 1):
        substring = sequence[i:i+n]
        if substring not in repeats:
            repeats[substring] = []
        repeats[substring].append(i + 1)
    return {seq: positions for seq, positions in repeats.items() if len(positions) > 1}

def analyze_repeats(fasta_dict, n):
    """Analyze repeats of length n across all sequences."""
    results = {
        'repeats_by_sequence': {},
        'all_repeats': Counter(),
    }
    for seq_id, sequence in fasta_dict.items():
        repeats = find_repeats_in_sequence(sequence, n)
        results['repeats_by_sequence'][seq_id] = repeats
        for repeat, positions in repeats.items():
            results['all_repeats'][repeat] += len(positions)
    return results

def main():
    # Parse arguments
    args = parse_arguments()
    fasta_dict = read_fasta_to_dict(args.fasta_file)
    if not fasta_dict:
        print("Failed to load FASTA file. Exiting.")
        return

    # Question 1: How many records?
    num_records = count_records(fasta_dict)
    print(f"1. Number of records in the file: {num_records}")

    # Question 2: Sequence lengths
    (all_lengths, longest_length, longest_ids, num_longest, 
     shortest_length, shortest_ids, num_shortest) = analyze_sequence_lengths(fasta_dict)
    print(f"\n2. Sequence Length Analysis:")
    print("   Lengths of all sequences:")
    for seq_id, length in sorted(all_lengths.items()):
        print(f"      {seq_id}: {length} nucleotides")
    print(f"   Longest sequence length: {longest_length}")
    print(f"   Identifiers of longest sequences: {', '.join(longest_ids)}")
    print(f"   Number of longest sequences: {num_longest}")
    print(f"   Shortest sequence length: {shortest_length}")
    print(f"   Identifiers of shortest sequences: {', '.join(shortest_ids)}")
    print(f"   Number of shortest sequences: {num_shortest}")

    # Question 3: ORF analysis for specified frame
    orf_results = analyze_orfs(fasta_dict, args.frame)
    print(f"\n3. ORF Analysis (Reading Frame {args.frame}):")
    print(f"   Length of longest ORF in file: {orf_results['longest_orf']}")
    print(f"   Sequence containing longest ORF: {orf_results['longest_orf_seq_id']}")
    if orf_results['longest_orf_seq_id']:
        print(f"   Starting position of longest ORF: {orf_results['longest_orf_start']}")
    print("   Longest ORF for each sequence:")
    for seq_id in sorted(orf_results['seq_specific_longest'].keys()):
        start, length = orf_results['seq_specific_longest'][seq_id]
        print(f"      {seq_id}: length {length} starting at position {start}")

    # Question 4: Repeat analysis for specified length
    repeat_results = analyze_repeats(fasta_dict, args.repeat_length)
    print(f"\n4. Repeat Analysis (Length {args.repeat_length}):")
    print("   Repeats found in each sequence:")
    for seq_id, repeats in repeat_results['repeats_by_sequence'].items():
        print(f"      Sequence {seq_id}:")
        if repeats:
            for repeat_seq, positions in sorted(repeats.items()):
                print(f"         '{repeat_seq}' found {len(positions)} times at positions: {positions}")
        else:
            print("         No repeats found")
    print("   Overall repeat statistics:")
    if repeat_results['all_repeats']:
        most_common = repeat_results['all_repeats'].most_common(1)[0]
        print(f"   Most frequent repeat: '{most_common[0]}' with {most_common[1]} total occurrences")
        print("   All repeats sorted by frequency:")
        for repeat, count in repeat_results['all_repeats'].most_common():
            print(f"      '{repeat}': {count} occurrences")
    else:
        print("   No repeats found in any sequence")

if __name__ == "__main__":
    main()
