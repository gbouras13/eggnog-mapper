#!/usr/bin/env python3
"""

Some code adapted from @mheinzinger 

https://github.com/mheinzinger/ProstT5/blob/main/scripts/generate_foldseek_db.py

"""


import os
import shutil
import struct
from pathlib import Path
import subprocess

from Bio import SeqIO
from ...emapperException import EmapperException
from ...common import FOLDSEEK

from ...utils import colorify

##
# Structural alphabets, in the state order Foldseek's substitution matrices use
# (data/mat3di.out and data/mat12st.out). The index of a letter here is the
# numeric state Foldseek expects on disk.
#
# 3Di has 20 real states. Foldseek's own structure-derived databases also use
# 'X' at index 20 for "invalid/masked", but eggnog-mapper never writes it: it
# does not mask low-confidence positions (see inference.write_predictions), so
# only the 20 real states appear here.
ALPHABET_3DI = "ACDEFGHIKLMNPQRSTVWY"
# The 12-state alphabet has exactly 12 states and no 'X'; Foldseek uses index 6
# ('H', Alphabet12St::INVALID_STATE) for invalid residues.
ALPHABET_12ST = "ACDEFGHIKLMN"
STATE_CNT_12ST = 12
INVALID_STATE_12ST = 6  # 'H'

# Foldseek packs both alphabets into one byte per residue:
#     byte = state3di * STATE_CNT_12ST + state12st
# so the values we write run 0..239. Such an _ss database is
# flagged by ORing DBTYPE_EXTENDED_3DI_12ST (32) into the high half of the
# dbtype; the low half stays DBTYPE_AMINO_ACIDS (0).
DBTYPE_AMINO_ACIDS = 0
DBTYPE_EXTENDED_3DI_12ST = 32
DBTYPE_SS_3DI_12ST = DBTYPE_AMINO_ACIDS | (DBTYPE_EXTENDED_3DI_12ST << 16)

_IDX_3DI = {c: i for i, c in enumerate(ALPHABET_3DI)}
_IDX_12ST = {c: i for i, c in enumerate(ALPHABET_12ST)}


def clean_aa_sequence(seq: str) -> str:
    """Normalise an input protein sequence.

    Trailing stop codons ('*', very common in translated CDS) are dropped —
    they are not residues, and the tokenizer has no token for them. A '*' left
    anywhere else is an internal stop: it stays as a position but becomes 'X',
    so the residue count is unchanged.

    Both the ModernProst input and the amino-acid database are passed through
    this, which is what guarantees the AA, 3Di and 12-state strings for a given
    query all end up the same length.
    """
    seq = seq.rstrip("*")
    if "*" in seq:
        seq = seq.replace("*", "X")
    return seq


def pack_3di_12st(seq_3di: str, seq_12st: str) -> bytes:
    """Pack a 3Di and a 12-state string into Foldseek's one-byte-per-residue form.

    A 3Di letter outside the 20 real states is an error rather than something to
    fold into a masked state: eggnog-mapper does not mask, so an 'X' here means
    the 3Di FASTA came from somewhere else (a structure-derived database, or a
    masking-enabled predictor) and the two alphabets can no longer be trusted to
    describe the same residue.

    Unknown 12-state letters collapse to INVALID_STATE_12ST, which is Foldseek's
    own encoding for an invalid residue in that alphabet.
    """
    if len(seq_3di) != len(seq_12st):
        raise EmapperException(
            f"3Di and 12-state sequences differ in length "
            f"({len(seq_3di)} vs {len(seq_12st)}); cannot pack them."
        )

    out = bytearray(len(seq_3di))
    for i, (c3, c12) in enumerate(zip(seq_3di, seq_12st)):
        try:
            s3 = _IDX_3DI[c3.upper()]
        except KeyError:
            hint = (" — 'X' is a masked position, which eggnog-mapper does not"
                    " produce and the packed 3Di+12-state encoding cannot"
                    " represent" if c3.upper() == 'X' else "")
            raise EmapperException(
                f"Unknown 3Di character {c3!r} at position {i}{hint}."
            ) from None
        s12 = _IDX_12ST.get(c12.upper(), INVALID_STATE_12ST)
        out[i] = s3 * STATE_CNT_12ST + s12
    return bytes(out)


def write_packed_ss_db(entries, out_db: Path) -> None:
    """Write an _ss database holding packed 3Di+12st bytes.

    entries: iterable of (key, packed_bytes), where key is the numeric MMseqs
    key of the corresponding entry in the amino-acid database.

    Foldseek's `tsv2db` cannot be used for this: the packed bytes span the whole
    0..251 range and would collide with TAB/NUL/newline in a TSV, and
    --output-dbtype cannot set the extended dbtype bit. So the three database
    files are written directly. Each entry is <bytes> + '\\n' + '\\0', which is
    what DBWriter produces and what DBReader's entry length (len + 2) assumes.
    """
    out_db = Path(out_db)
    parts = []
    index_lines = []
    offset = 0

    for key, packed in entries:
        buf = packed + b"\n\x00"
        parts.append(buf)
        index_lines.append(f"{key}\t{offset}\t{len(buf)}")
        offset += len(buf)

    with open(out_db, "wb") as fh:
        fh.write(b"".join(parts))

    with open(f"{out_db}.index", "w") as fh:
        if index_lines:
            fh.write("\n".join(index_lines) + "\n")

    with open(f"{out_db}.dbtype", "wb") as fh:
        fh.write(struct.pack("<i", DBTYPE_SS_3DI_12ST))

    return


def db_has_12st(db_prefix) -> bool:
    """Whether a Foldseek database's _ss holds the packed 3Di+12st alphabet.

    Returns None when it cannot be determined (e.g. the _ss.dbtype file is not
    next to the database, as for a pre-built index), so callers can tell
    "no 12-state" apart from "unknown".
    """
    dbtype_file = Path(f"{db_prefix}_ss.dbtype")
    if not dbtype_file.is_file():
        return None
    try:
        raw = dbtype_file.read_bytes()[:4]
        if len(raw) < 4:
            return None
        dbtype = struct.unpack("<i", raw)[0]
    except OSError:
        return None
    extended = (dbtype >> 16) & 0x7FFE
    return bool(extended & DBTYPE_EXTENDED_3DI_12ST)


def remove_file(file_path: Path) -> None:
    """
    Remove a file if it exists.

    Parameters:
        file_path (Path): Path to the file to remove.

    Returns:
        None
    """
    if file_path.exists():
        file_path.unlink()  # Use unlink to remove the file


def generate_foldseek_db_from_aa_3di(
    fasta_aa: Path, fasta_3di: Path, tmp_dir: Path, querydb: str,
    fasta_12st: Path = None
) -> None:
    """
    Generate Foldseek database from amino-acid and 3Di sequences.

    Args:
        fasta_aa (Path): Path to the amino-acid FASTA file.
        fasta_3di (Path): Path to the 3Di FASTA file.
        tmp_dir (Path): Path to the directory where Foldseek database will be stored.
        querydb (str): PAth to base for the Foldseek database.
        fasta_12st (Path): Path to the 12-state FASTA file. When given, the _ss
            database holds packed 3Di+12st bytes; otherwise it holds plain 3Di
            letters, as older Foldseek versions expect.

    Returns:
        None
    """
    # read in amino-acid sequences, normalised the same way the predictor
    # normalised them so the AA and structural strings stay the same length
    sequences_aa = {}
    for record in SeqIO.parse(fasta_aa, "fasta"):
        sequences_aa[record.id] = clean_aa_sequence(str(record.seq))

    # read in 3Di strings
    sequences_3di = {}
    for record in SeqIO.parse(fasta_3di, "fasta"):
        if not record.id in sequences_aa.keys():
            print(
                "Warning: ignoring 3Di entry {}, since it is not in the amino-acid FASTA file".format(
                    record.id
                )
            )
        else:
            sequences_3di[record.id] = str(record.seq)  # no upper if masked

    # read in 12-state strings
    sequences_12st = {}
    if fasta_12st is not None:
        for record in SeqIO.parse(fasta_12st, "fasta"):
            if record.id in sequences_aa:
                sequences_12st[record.id] = str(record.seq)

    # assert that we parsed 3Di strings for all sequences in the amino-acid FASTA file
    for id in sequences_aa.keys():
        if not id in sequences_3di.keys():
            print(
                "Warning: entry {} in amino-acid FASTA file has no corresponding 3Di string".format(
                    id
                )
            )
            print("Removing: entry {} from the Foldseek database ".format(id))
            sequences_aa = {
                id: sequence
                for id, sequence in sequences_aa.items()
                if id in sequences_3di
            }

    # same, for the 12-state strings
    if fasta_12st is not None:
        missing_12st = [i for i in sequences_aa if i not in sequences_12st]
        for id in missing_12st:
            print(
                "Warning: entry {} has no corresponding 12-state string".format(id)
            )
            print("Removing: entry {} from the Foldseek database ".format(id))
        if missing_12st:
            sequences_aa = {
                id: sequence
                for id, sequence in sequences_aa.items()
                if id in sequences_12st
            }

    # https://github.com/mheinzinger/ProstT5/issues/41

    temp_aa_tsv: Path = Path(tmp_dir) / "aa.tsv"
    with open(temp_aa_tsv, "w") as f:
        for i,id in enumerate(sequences_aa.keys()):
            f.write("{}\t{}\n".format(str(i+1), sequences_aa[id]))

    temp_header_tsv: Path = Path(tmp_dir) / "header.tsv"
    with open(temp_header_tsv, "w") as f:
        for i,id in enumerate(sequences_aa.keys()):
            f.write("{}\t{}\n".format(str(i+1), id))

    #### write temp tsv files

    # create foldseek db names

    aa_db_name: Path = querydb
    ss_db_name: Path = f"{querydb}_ss"
    header_db_name: Path = f"{querydb}_h"

    # create Foldseek database with foldseek tsv2db

    foldseek_tsv2db(temp_aa_tsv, aa_db_name, 0)
    foldseek_tsv2db(temp_header_tsv, header_db_name, 12)

    if fasta_12st is not None:
        # keys must line up with the amino-acid database written above
        write_packed_ss_db(
            ((str(i + 1), pack_3di_12st(sequences_3di[id], sequences_12st[id]))
             for i, id in enumerate(sequences_aa.keys())),
            ss_db_name,
        )
    else:
        temp_3di_tsv: Path = Path(tmp_dir) / "3di.tsv"
        with open(temp_3di_tsv, "w") as f:
            for i,id in enumerate(sequences_aa.keys()):
                f.write("{}\t{}\n".format(str(i+1), sequences_3di[id]))
        foldseek_tsv2db(temp_3di_tsv, ss_db_name, 0)
        remove_file(temp_3di_tsv)

    # clean up
    remove_file(temp_aa_tsv)
    remove_file(temp_header_tsv)


def foldseek_tsv2db(
    in_tsv: Path, out_db_name: Path, db_type: int
) -> None:
    """
    Convert a Foldseek TSV file to a Foldseek database.

    Args:
        in_tsv (Path): Path to the input TSV file.
        out_db_name (Path): Path for the output Foldseek database.
        db_type (int): Type of the output database.
        logdir (Path): Path to the directory where logs will be stored.

    Returns:
        None
    """


    # mmseqs createdb examples/QUERY.fasta queryDB
    cmd = (
        f'{FOLDSEEK} tsv2db \'{in_tsv}\' \'{out_db_name}\' --output-dbtype \'{str(db_type)}\' '
    )


    print(colorify('  '+cmd, 'yellow'))
    try:
        completed_process = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    except subprocess.CalledProcessError as cpe:
        raise EmapperException("Error running 'foldseek tsv2db': "+cpe.stderr.decode("utf-8").strip().split("\n")[-1])
    return cmd

