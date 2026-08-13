from Bio import SeqIO


def get_all_protein_ids(fasta):
    return {
            rec.id.split()[0]   # be safe: strip description
            for rec in SeqIO.parse(fasta, "fasta")
        }

def write_no_hit_fasta(in_fasta, out_fasta, no_hit_ids):
    with open(out_fasta, "w") as out:
        for rec in SeqIO.parse(in_fasta, "fasta"):
            rec_id = rec.id.split()[0]
            if rec_id in no_hit_ids:
                SeqIO.write(rec, out, "fasta")