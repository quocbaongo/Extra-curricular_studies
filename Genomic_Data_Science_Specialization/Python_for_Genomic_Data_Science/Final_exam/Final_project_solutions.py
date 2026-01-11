#!/usr/bin/env python3

from Bio import SeqIO
from collections import Counter

# --- Configuration ---
FILENAME = "dna2.fasta"

# --- Helper Functions (Adapted from provided script) ---

def get_orfs(sequence, frame):
    """
    Finds ORFs in a specific reading frame (1, 2, or 3).
    Returns a list of tuples: (start_position_1_based, length)
    """
    sequence = sequence.upper()
    orfs = []
    # Adjust frame to 0-based index (Frame 1 -> index 0)
    start_pos = frame - 1
    seq_in_frame = sequence[start_pos:]
    
    # Iterate over codons
    for i in range(0, len(seq_in_frame) - 2, 3):
        codon1 = seq_in_frame[i:i+3]
        if codon1 == "ATG":
            for j in range(i + 3, len(seq_in_frame) - 2, 3):
                codon2 = seq_in_frame[j:j+3]
                if codon2 in ["TAA", "TAG", "TGA"]:
                    # Found an ORF
                    # Calculate actual length (including stop codon, usually excluded in some defs, 
                    # but typically length is end-start. Here we use length of the content)
                    # The prompt implies standard length.
                    length = (j + 3) - i
                    # Actual start position (1-based)
                    actual_start = start_pos + i + 1
                    orfs.append((actual_start, length))
                    break
    return orfs

def get_repeats(sequences, n):
    """
    Finds all repeats of length n across all sequences.
    Returns a Counter object with repeat sequences and their total frequencies.
    """
    global_repeats = Counter()
    for seq in sequences:
        seq = str(seq).upper()
        for i in range(len(seq) - n + 1):
            subseq = seq[i:i+n]
            global_repeats[subseq] += 1
    return global_repeats

def main():
    # 1. Load the Data
    try:
        records = list(SeqIO.parse(FILENAME, "fasta"))
    except FileNotFoundError:
        print(f"Error: {FILENAME} not found. Please make sure the file is in the same directory.")
        return

    # --- Questions 1-3: Record Statistics ---
    
    # Q1. How many records are in the multi-FASTA file?
    num_records = len(records)
    
    # Q2 & Q3. Longest and Shortest sequence lengths
    seq_lengths = [len(rec.seq) for rec in records]
    max_len = max(seq_lengths)
    min_len = min(seq_lengths)

    print(f"1. Number of records in the multi-FASTA file: {num_records}")
    print(f"2. The length of the longest sequence in the file: {max_len}")
    print(f"3. The length of the shortest sequence in the file: {min_len}")

    # --- Questions 4-7: ORF Analysis ---
    
    # We will compute ORFs for all sequences and all frames to query them easily
    # Structure: list of dictionaries with info needed for filtering
    all_orf_data = [] 

    target_id_q7 = "gi|142022655|gb|EQ086233.1|16"

    for rec in records:
        seq_str = str(rec.seq)
        for frame in [1, 2, 3]:
            orfs = get_orfs(seq_str, frame)
            for start, length in orfs:
                all_orf_data.append({
                    "id": rec.id,
                    "frame": frame,
                    "length": length,
                    "start": start
                })

    # Q4. Length of longest ORF in reading frame 2
    orfs_frame_2 = [o['length'] for o in all_orf_data if o['frame'] == 2]
    q4_ans = max(orfs_frame_2) if orfs_frame_2 else 0
    print(f"4. The length of the longest ORF appearing in reading frame 2 of any of the sequences: {q4_ans}")

    # Q5. Start position of the longest ORF in reading frame 3
    # First, find the max length in frame 3
    orfs_frame_3 = [o for o in all_orf_data if o['frame'] == 3]
    if orfs_frame_3:
        max_len_f3 = max(o['length'] for o in orfs_frame_3)
        # Find the start position of that specific ORF (or ORFs if ties exist)
        # Usually, if there are ties, any valid one counts, but we'll grab the first found.
        longest_f3_orfs = [o for o in orfs_frame_3 if o['length'] == max_len_f3]
        # In case there are multiple of the same max length, we print the info for the first one found
        # (or you can print all if needed, but the quiz usually asks for 'the' position).
        q5_ans = longest_f3_orfs[0]['start']
    else:
        q5_ans = 0
    print(f"5. The starting position of the longest ORF in reading frame 3 in any of the sequences: {q5_ans}")

    # Q6. Length of the longest ORF appearing in any sequence and in any forward reading frame
    # This is simply the max length in our entire dataset
    q6_ans = max(o['length'] for o in all_orf_data) if all_orf_data else 0
    print(f"6. The length of the longest ORF appearing in any sequence and any forward reading frame: {q6_ans}")

    # Q7. Length of the longest forward ORF in the sequence with identifier gi|142022655|gb|EQ086233.1|16
    q7_orfs = [o['length'] for o in all_orf_data if target_id_q7 in o['id']]
    q7_ans = max(q7_orfs) if q7_orfs else 0
    print(f"7. The length of the longest forward ORF that appears in the sequence with the identifier gi|142022655|gb|EQ086233.1|16: {q7_ans}")

    # --- Questions 8-10: Repeat Analysis ---
    
    # Extract sequences list for processing
    sequences = [rec.seq for rec in records]

    # Q8. Most frequently occurring repeat of length 6
    repeats_6 = get_repeats(sequences, 6)
    # most_common(1) returns a list [(seq, count)]
    q8_ans = repeats_6.most_common(1)[0][1] if repeats_6 else 0
    print(f"8. The number of occurrences of the the most frequently occuring repeat of length 6 in all sequences: {q8_ans}")

    # Q9. Repeats of length 12. How many different 12-base sequences occur Max times?
    repeats_12 = get_repeats(sequences, 12)
    if repeats_12:
        max_count_12 = repeats_12.most_common(1)[0][1]
        # Count how many sequences share this max count
        q9_ans = sum(1 for count in repeats_12.values() if count == max_count_12)
    else:
        q9_ans = 0
    print(f"9. The number of different 12-base sequences occurring Max times: {q9_ans}")

    # Q10. Which one of the following repeats of length 7 has a maximum number of occurrences?
    # Since we don't see the multiple choice options here, we will print the most frequent one(s).
    repeats_7 = get_repeats(sequences, 7)
    if repeats_7:
        max_count_7 = repeats_7.most_common(1)[0][1]
        top_repeats_7 = [(seq, count) for seq, count in repeats_7.items() if count == max_count_7]
        # Formatting for readability
        q10_ans = ", ".join([f"{seq} (Count: {count})" for seq, count in top_repeats_7])
    else:
        q10_ans = "None"
    print(f"10. The repeats of length 7 with a maximum number of occurrences: {q10_ans}")

if __name__ == "__main__":
    main()
