##
## CPCantalapiedra 2020

import errno, os, shutil, psutil
from sys import stderr
import time
from os.path import exists as pexists
from os.path import join as pjoin

from .utils import colorify
from .common import silent_rm, decompress_input, detect_compression, \
    ITYPE_GENOME, ITYPE_META, ITYPE_PROTS, ITYPE_CDS, get_data_path
from .emapperException import EmapperException

from .genepred.genepred_modes import GENEPRED_MODE_SEARCH, GENEPRED_MODE_PRODIGAL, get_predictor
from .genepred.util import create_prots_file
from .search.search_modes import get_searcher, SEARCH_MODE_NO_SEARCH, \
    SEARCH_MODE_FOLDSEEK, SEARCH_MODE_DIAMOND_FOLDSEEK, SEARCH_MODE_MMSEQS2_FOLDSEEK
from .search.hits_io import parse_seeds
from .search.foldseek.filter import get_all_protein_ids, write_no_hit_fasta
from .annotation.annotator import Annotator
from .deco.decoration import run_gff_decoration, DECORATE_GFF_NONE, create_blastx_hits_gff

class Emapper:

    genepred_fasta_file = genepred_gff_file = searcher_out_file = seed_orthologs_file = None
    annot_file = orthologs_file = pfam_file = None
    _output_files = None
    
    output_dir = scratch_dir = None
    _current_dir = None

    itype = genepred = mode = annot = decorate_gff = None
    genepred_is_prodigal = genepred_is_blastx = None
    override = resume = None

    ##
    def __init__(self, itype, genepred, mode, annot, excel, report_orthologs, decorate_gff,
                 prefix, output_dir, scratch_dir, resume, override):

        #
        self.output_dir = output_dir
        self.scratch_dir = scratch_dir
        
        # Output and intermediate files
        self.genepred_fasta_file = f"{prefix}.emapper.genepred.fasta"
        self.genepred_gff_file = f"{prefix}.emapper.genepred.gff"

        self.search_out_file = f"{prefix}.emapper.hits"
        
        self.seed_orthologs_file = f"{prefix}.emapper.seed_orthologs"
        self.annot_file = f"{prefix}.emapper.annotations"
        self.no_annot_file = f"{prefix}.emapper.no_annotations.fasta"
        self.orthologs_file = f"{prefix}.emapper.orthologs"
        self.pfam_file = f"{prefix}.emapper.pfam"
        self.excel_file = f"{prefix}.emapper.annotations.xlsx"
        # Phase 7.1c: optional drop log written when --report_dropped is set.
        self.dropped_file = f"{prefix}.emapper.dropped"
        self.deco_gff_file = f"{prefix}.emapper.decorated.gff"

        # Two-stage *-foldseek modes: after the DIAMOND/MMseqs2 pass, a second
        # Foldseek pass runs over the still-unannotated queries and writes its
        # own set of output files under this prefix. See
        # _switch_to_foldseek_stage().
        self.is_two_stage_foldseek = mode in (SEARCH_MODE_DIAMOND_FOLDSEEK,
                                              SEARCH_MODE_MMSEQS2_FOLDSEEK)
        self.foldseek_prefix = f"{prefix}.foldseek.extra"
        self.foldseek_no_hits_file = f"{prefix}.emapper.no_hits.fasta"

        self.genepred = genepred
        self.mode = mode
        
        self.annot = annot
        self.excel = excel
        self.report_orthologs = report_orthologs
        self.decorate_gff = decorate_gff
        self.resume = resume
        self.override = override

        self._output_files = []
        self.itype = itype
        if itype == ITYPE_GENOME or itype == ITYPE_META:
            self._output_files.append(self.genepred_fasta_file)
            self._output_files.append(self.genepred_gff_file)            
            
        if self.decorate_gff != DECORATE_GFF_NONE:
            self._output_files.append(self.deco_gff_file)

        self.genepred_is_prodigal = ((itype == ITYPE_GENOME or itype == ITYPE_META) and
                                     genepred == GENEPRED_MODE_PRODIGAL)
        
        self.genepred_is_blastx = ((itype == ITYPE_GENOME or itype == ITYPE_META) and
                                   genepred == GENEPRED_MODE_SEARCH)
            
        if mode == SEARCH_MODE_NO_SEARCH:
            self._output_files.extend([self.annot_file, self.pfam_file])
        elif not annot:
            self._output_files.extend([self.search_out_file, self.seed_orthologs_file])
        else:
            self._output_files.extend([self.search_out_file, self.seed_orthologs_file,
                                       self.annot_file, self.pfam_file])

        # Register the Foldseek pass outputs too, so that --override/--resume
        # and the scratch-dir copy in wrap_up() treat them like any other output.
        if self.is_two_stage_foldseek:
            fp = self.foldseek_prefix
            self._output_files.extend([
                self.foldseek_no_hits_file,
                f"{fp}.emapper.hits",
                f"{fp}.emapper.seed_orthologs",
            ])
            if annot == True:
                self._output_files.extend([f"{fp}.emapper.annotations",
                                           f"{fp}.emapper.pfam"])

        if annot == True and excel == True:
            self._output_files.append(self.excel_file)

        if report_orthologs == True:
            self._output_files.append(self.orthologs_file)

        # Phase 7.1c: dropped log is written only when --report_dropped is
        # set. Register unconditionally so --override / --resume handle
        # any leftover from a previous run cleanly (silent_rm is no-op
        # when the file is absent).
        self._output_files.append(self.dropped_file)

        # force user to decide what to do with existing files
        files_present = set([pexists(pjoin(self.output_dir, fname)) for fname in self._output_files])
        if True in files_present and not resume and not override:
            raise EmapperException("Output files detected in disk. Use --resume or --override to continue")

        if override == True:
            for outf in self._output_files:
                silent_rm(pjoin(self.output_dir, outf))

        # Some files are not being resumed and will be ovewritten
        if resume == True:
            silent_rm(pjoin(self.output_dir, self.no_annot_file))
            silent_rm(pjoin(self.output_dir, self.excel_file))

        # If using --scratch_dir, change working dir
        # (once finished move them again to output_dir)
        if scratch_dir:
            self._current_dir = scratch_dir
            
            if resume:
                for fname in self._output_files:
                    full_fname = pjoin(self.output_dir, fname)
                    if pexists(full_fname):
                        print("   Copying input file %s to scratch dir %s" % (full_fname, scratch_dir))
                        shutil.copy(full_fname, scratch_dir)
            
        else:
            self._current_dir = output_dir
            
        return

    
    ##
    def gene_prediction(self, args, infile):
        predictor = None

        predictor = get_predictor(args, self.genepred)
        # --resume skips gene prediction to resume from diamond/mmseqs/hmmer hits directly
        if predictor is not None and self.resume == False:
            predictor.predict(infile)
        elif predictor is not None:
            print(colorify("[resume] Skipping gene prediction — reusing existing files.", "blue"),
                  file=stderr)
        
        return predictor

    
    ##
    def search(self, args, infile, predictor = None):

        queries_file = None
        
        # determine input file: from prediction or infile
        if predictor is None:
            queries_file = infile
        else:
            queries_file = predictor.outprots
            args.translate = False
            args.itype = ITYPE_PROTS

        # search
        searcher = get_searcher(args, self.mode, get_data_path())
        searcher_name = None
        hits = None
        if searcher is not None:
            searcher_name = searcher.name
            try:
                hits = searcher.search(queries_file,
                                       pjoin(self._current_dir, self.seed_orthologs_file),
                                       pjoin(self._current_dir, self.search_out_file))
            except Exception as e:
                searcher.clear()
                raise(e)

            # If gene prediction from the hits obtained in the search step
            # create GFF and FASTA of predicted CDS/proteins
            if self.genepred_is_blastx == True:
                
                # create GFF of predicted CDS
                print(colorify("Crafting GFF file of CDS ...", "lgreen"))
                gff_outfile = pjoin(self._current_dir, self.genepred_gff_file)
                hits = create_blastx_hits_gff(hits, gff_outfile, searcher_name, args.decorate_gff_ID_field)
                
                # create fasta file of predicted CDS
                print(colorify("Crafting fasta file of CDS ...", "lgreen"))
                fasta_file = pjoin(self._current_dir, self.genepred_fasta_file)
                silent_rm(fasta_file)
                hits = create_prots_file(queries_file, hits, fasta_file, args.translate, args.trans_table)
                queries_file = fasta_file
                if args.translate == True:
                    args.itype = ITYPE_PROTS
                else:
                    args.itype = ITYPE_CDS
                
        return searcher, searcher_name, hits, queries_file
    
    
    ##
    def annotate(self, args, hits, annotate_hits_table, queries_file):
        annotated_hits = None
        _resumed_ref = [0]

        if self.annot == True or self.report_orthologs:
            annot_in = None  # a generator of hits to annotate

            if annotate_hits_table is not None:
                if not pexists(annotate_hits_table):
                    raise EmapperException(
                        f"Could not find the file with the hits "
                        f"table to annotate: {annotate_hits_table}"
                    )
                annot_in = parse_seeds(annotate_hits_table)
            elif hits is not None:
                annot_in = hits
            else:
                raise EmapperException("Could not find hits to annotate.")

            annotator = Annotator(args, self.annot, self.excel, self.report_orthologs)

            if annot_in is not None and annotator is not None:
                annotated_hits, _resumed_ref = annotator.annotate(
                    annot_in,
                    pjoin(self._current_dir, self.annot_file),
                    pjoin(self._current_dir, self.excel_file),
                    pjoin(self._current_dir, self.orthologs_file),
                    pjoin(self._current_dir, self.pfam_file),
                    queries_file,
                    dropped_file=pjoin(self._current_dir, self.dropped_file),
                )
        else:
            # --no_annot: wrap hits as (hit, None) pairs, or yield nothing if
            # no search was run (e.g. -m no_search --no_annot is a no-op).
            annotated_hits = ((hit, None) for hit in hits) if hits is not None else iter([])

        return annotated_hits, _resumed_ref

    
    ##
    def decorate_gff_f(self, args, predictor, searcher_name, annotated_hits):

        gff_outfile = pjoin(self._current_dir, self.deco_gff_file)
        if predictor is not None:
            if self.resume == True:
                gff_genepred_file = pjoin(self._current_dir, self.genepred_gff_file)
                gff_genepred_fasta = pjoin(self._current_dir, self.genepred_fasta_file)
            else:
                gff_genepred_file = predictor.outgff
                gff_genepred_fasta = predictor.outprots
        else:
            gff_genepred_file = None
            gff_genepred_fasta = None
        
        annotated_hits = run_gff_decoration(self.decorate_gff, args.decorate_gff_ID_field,
                                            self.genepred_is_prodigal, self.genepred_is_blastx,
                                            gff_genepred_file, gff_genepred_fasta, gff_outfile,
                                            predictor, searcher_name, annotated_hits)
        
        return annotated_hits


    ##
    def _switch_to_foldseek_stage(self):
        """Point the output-file attributes at the Foldseek pass, and switch
        self.mode so that search() builds a FoldseekSearcher."""
        prefix = self.foldseek_prefix

        # genepred_fasta_file / genepred_gff_file are deliberately left alone:
        # gene prediction ran once for the whole run, and wrap_up() moves the
        # predictor output to those names.
        self.search_out_file = f"{prefix}.emapper.hits"
        self.seed_orthologs_file = f"{prefix}.emapper.seed_orthologs"
        self.annot_file = f"{prefix}.emapper.annotations"
        self.no_annot_file = f"{prefix}.emapper.no_annotations.fasta"
        self.orthologs_file = f"{prefix}.emapper.orthologs"
        self.pfam_file = f"{prefix}.emapper.pfam"
        self.excel_file = f"{prefix}.emapper.annotations.xlsx"
        self.dropped_file = f"{prefix}.emapper.dropped"
        self.deco_gff_file = f"{prefix}.emapper.decorated.gff"

        self.mode = SEARCH_MODE_FOLDSEEK

        return

    ##
    def filter_infile(self, infile, annotated_ids):
        """Write a FASTA of the queries in infile that got no annotation.

        Returns the path of that FASTA, to be used as input for the Foldseek
        pass of the two-stage *-foldseek modes.
        """
        all_input_ids = get_all_protein_ids(infile)
        no_hit_ids = all_input_ids - annotated_ids

        no_hit_fasta = pjoin(self._current_dir, self.foldseek_no_hits_file)
        write_no_hit_fasta(infile, no_hit_fasta, no_hit_ids)

        print(colorify(f"{len(no_hit_ids)} of {len(all_input_ids)} queries "
                       f"were left unannotated; running Foldseek on them.", 'lgreen'),
              file=stderr)

        return no_hit_fasta

    ##
    def _print_progress(self, n, n_new, start_time, mem_monitor):
        total_time = time.time() - start_time
        percen_mem = psutil.virtual_memory().percent
        percen_avail = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

        if total_time > 0.005:
            rate = (float(n_new) / total_time) if total_time > 0 else 0.0
            n_resumed = n - n_new
            if n_resumed:
                msg = f"{n} ({n_new} new, {n_resumed} resumed) {total_time:.1f}s {rate:.2f} q/s"
            else:
                msg = f"{n} {total_time:.1f}s {rate:.2f} q/s"
            if mem_monitor:
                msg += f" (mem {percen_mem:.1f}% used, {percen_avail:.1f}% avail)"
            print(msg, file=stderr)

        stderr.flush()

        return total_time

    ##
    @staticmethod
    def _item_query_name(item):
        """Query name of an item yielded by the annotation/decoration chain.

        Those generators yield (hit, annotation) pairs, where hit[0] is the
        query name. Kept tolerant so a bare hit list also works.
        """
        hit = item[0] if isinstance(item, tuple) else item
        return hit[0] if isinstance(hit, (list, tuple)) else hit

    ##
    def run_generator(self, generator, resumed_ref=None, CHUNK_SIZE=500, mem_monitor=True,
                      collect_ids=None):
        """Drain the annotation generator, reporting progress.

        collect_ids: optional set, filled with the query name of every item
        consumed. Used by the two-stage *-foldseek modes to work out which
        queries still have no annotation.
        """

        n = 0
        start_time = time.time()

        _n_skipped = resumed_ref if resumed_ref is not None else [0]

        self._print_progress(0, 0, start_time, mem_monitor)

        for item in generator:
            n += 1
            if collect_ids is not None:
                collect_ids.add(self._item_query_name(item))
            if n % CHUNK_SIZE == 0:
                n_new = n - _n_skipped[0]
                self._print_progress(n, n_new, start_time, mem_monitor)

        n_new = n - _n_skipped[0]
        total_time = self._print_progress(n, n_new, start_time, mem_monitor)

        return n, total_time

    ##
    def wrap_up(self, predictor, searcher):
        ##
        # Clear things
        if predictor is not None:
            if self.resume == False: # if self.resume == True there aren't new genepred files to move
                shutil.move(predictor.outprots, pjoin(self._current_dir, self.genepred_fasta_file))
                shutil.move(predictor.outgff, pjoin(self._current_dir, self.genepred_gff_file))
                predictor.clear() # removes gene predictor output directory
        if searcher is not None:
            searcher.clear()
        
        ##
        # If running in scratch, move files to real output dir and clean up
        if self.scratch_dir:
            for fname in self._output_files:
                full_fname = pjoin(self.scratch_dir, fname)
                if pexists(full_fname):
                    print(" Copying result file %s from scratch to %s" % (full_fname, self.output_dir))
                    shutil.copy(full_fname, self.output_dir)

            print(colorify(f"Data in {self.scratch_dir} will be not removed. Please, clear it manually.", 'red'))

        ##
        # Finalize and exit
        print(colorify('Done', 'green'))
        print(colorify('Result files:', 'yellow'))
        for fname in self._output_files:
            pathname = pjoin(self.output_dir, fname)
            if pexists(pathname):
                print("   %s" % (pathname))
                
        return
    
    ##
    def run(self, args, infile, annotate_hits_table=None):

        _tmp_infile = None
        try:
            ##
            # Decompress compressed input transparently (gzip or bzip2, by magic bytes)
            if infile is not None:
                ctype = detect_compression(infile)
                if ctype:
                    print(colorify(
                        f"Detected {ctype} compressed input — decompressing to temp file...",
                        'blue'), file=stderr)
                    infile, _tmp_infile = decompress_input(infile, args.temp_dir)

            ##
            # Step 0. Gene prediction
            predictor = self.gene_prediction(args, infile)

            ##
            # Step 1. Sequence search
            searcher, searcher_name, hits, queries_file = self.search(args, infile, predictor)

            ##
            # Step 2. Annotation
            annotated_hits, _resumed_ref = self.annotate(args, hits, annotate_hits_table, queries_file)

            ##
            # Step 3. Decorate GFF
            annotated_hits = self.decorate_gff_f(args, predictor, searcher_name, annotated_hits)

            ##
            # Run the generators
            annotated_ids = set() if self.is_two_stage_foldseek else None
            n, elapsed_time = self.run_generator(annotated_hits, _resumed_ref,
                                                 collect_ids=annotated_ids)

            ##
            # Two-stage *-foldseek modes: re-run steps 1-3 with Foldseek over
            # the queries the first pass left unannotated.
            if self.is_two_stage_foldseek:
                # the first-pass searcher is done with; release its temp dir
                # before wrap_up() only ever sees the Foldseek one
                if searcher is not None:
                    searcher.clear()
                    searcher = None

                self._switch_to_foldseek_stage()
                foldseek_infile = self.filter_infile(infile, annotated_ids)

                # Step 1. Sequence search. No predictor: search() would use
                # predictor.outprots and ignore our filtered FASTA. Gene
                # prediction already ran in the first pass, and the filtered
                # FASTA holds the predicted proteins that stayed unannotated.
                searcher, searcher_name, hits, queries_file = self.search(
                    args, foldseek_infile, None)

                # Step 2. Annotation. annotate_hits_table is deliberately not
                # reused: it belongs to the first pass, and re-reading it here
                # would annotate the whole table a second time.
                annotated_hits, _resumed_ref = self.annotate(
                    args, hits, None, queries_file)

                # Step 3. Decorate GFF
                annotated_hits = self.decorate_gff_f(args, predictor, searcher_name,
                                                     annotated_hits)

                # Run the generators. elapsed_time is deliberately not summed:
                # the caller reports wall-clock time for the whole run.
                n_foldseek, _ = self.run_generator(annotated_hits, _resumed_ref)
                n += n_foldseek

            ##
            # Finish
            self.wrap_up(predictor, searcher)

        finally:
            if _tmp_infile is not None:
                silent_rm(_tmp_infile)

        return n, elapsed_time

## END
