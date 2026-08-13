##
## CPCantalapiedra 2020

from os.path import join as pjoin

from ..emapperException import EmapperException

from .diamond.diamond import DiamondSearcher
from .mmseqs.mmseqs import MMseqs2Searcher
# NOTE: FoldseekSearcher is imported lazily in get_searcher(). It pulls in
# torch/transformers for 3Di inference, which we do not want to make a hard
# import-time dependency of every eggnog-mapper run.

SEARCH_MODE_NO_SEARCH = "no_search"
SEARCH_MODE_DIAMOND = "diamond"
SEARCH_MODE_MMSEQS2 = "mmseqs"
SEARCH_MODE_FOLDSEEK = "foldseek"
# Two-stage modes: run the sequence searcher first, then Foldseek over the
# queries left unannotated by that first pass.
SEARCH_MODE_DIAMOND_FOLDSEEK = "diamond-foldseek"
SEARCH_MODE_MMSEQS2_FOLDSEEK = "mmseqs-foldseek"

ALL_SEARCH_MODES = (
    SEARCH_MODE_NO_SEARCH,
    SEARCH_MODE_DIAMOND,
    SEARCH_MODE_MMSEQS2,
    SEARCH_MODE_FOLDSEEK,
    SEARCH_MODE_DIAMOND_FOLDSEEK,
    SEARCH_MODE_MMSEQS2_FOLDSEEK,
)

# Modes whose first search pass is run by each backend. The *-foldseek modes
# appear here because their first pass is DIAMOND/MMseqs2; Emapper.run then
# switches self.mode to SEARCH_MODE_FOLDSEEK for the second pass.
DIAMOND_MODES = (SEARCH_MODE_DIAMOND, SEARCH_MODE_DIAMOND_FOLDSEEK)
MMSEQS_MODES = (SEARCH_MODE_MMSEQS2, SEARCH_MODE_MMSEQS2_FOLDSEEK)
FOLDSEEK_MODES = (SEARCH_MODE_FOLDSEEK, SEARCH_MODE_DIAMOND_FOLDSEEK,
                  SEARCH_MODE_MMSEQS2_FOLDSEEK)


def get_searcher(args, mode, data_path):
    if mode == SEARCH_MODE_NO_SEARCH:
        return None
    if mode in DIAMOND_MODES:
        return DiamondSearcher(args, get_eggnog_dmnd_db(args.dmnd_db, mode, data_path))
    if mode in MMSEQS_MODES:
        return MMseqs2Searcher(args)
    if mode == SEARCH_MODE_FOLDSEEK:
        try:
            from .foldseek.foldseek import FoldseekSearcher
        except ImportError as e:
            raise EmapperException(
                f"Foldseek search mode requires extra dependencies "
                f"(torch, transformers, tqdm): {e}"
            )
        return FoldseekSearcher(args)
    raise EmapperException("Unknown search mode %s" % mode)


def get_eggnog_dmnd_db(dmnd_db, mode, data_path):
    if dmnd_db is not None:
        return dmnd_db
    if mode is None or mode in DIAMOND_MODES:
        return pjoin(data_path, "eggnog_proteins.dmnd")
    raise EmapperException(f"Unrecognized mode (-m) {mode} for get_eggnog_dmnd_db")

## END
