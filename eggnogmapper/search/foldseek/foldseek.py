##
## Gbouras 2026

import os
from os.path import join as pjoin, isdir as pisdir, isfile as pisfile
import shutil
import subprocess
from sys import stderr as sys_stderr
from tempfile import mkdtemp, mkstemp
from pathlib import Path
import uuid

from ...common import FOLDSEEK, require_tool, resolve_foldseek_db, \
    ITYPE_CDS, ITYPE_PROTS, ITYPE_GENOME, ITYPE_META, get_hf_cache_dir
from ...emapperException import EmapperException
from ...utils import colorify, translate_cds_to_prots

from ..hmmer.hmmer_seqio import iter_fasta_seqs

from ..hits_io import output_seeds
from ..diamond.diamond import hit_does_overlap, ALLOW_OVERLAPS_ALL

from .inference import infer_3Di, MODERNPROST_MODEL
from .create_fs_db import generate_foldseek_db_from_aa_3di, db_has_12st


def create_foldseek_db(dbprefix, in_fasta):
    cmd = (
        f'{FOLDSEEK} createdb \'{in_fasta}\' \'{dbprefix}\''
    )

    print(colorify('  '+cmd, 'yellow'))
    try:
        completed_process = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    except subprocess.CalledProcessError as cpe:
        raise EmapperException("Error running foldseek: "+cpe.stderr.decode("utf-8").strip().split("\n")[-1])
        
    return



def create_foldseek_index(dbprefix, tmp_dir):
    cmd = (
        f'{FOLDSEEK} createindex \'{dbprefix}\' \'{tmp_dir}\''
    )

    print(colorify('  '+cmd, 'yellow'))
    try:
        completed_process = subprocess.run(cmd, capture_output=True, check=True, shell=True)
    except subprocess.CalledProcessError as cpe:
        raise EmapperException("Error running foldseek: "+cpe.stderr.decode("utf-8").strip().split("\n")[-1])
    return

class FoldseekSearcher:

    name = "foldseek"
    
    # Command
    cpu = targetdb = temp_dir = no_file_comments = None

    # for foldseek - do i need these? probably not just defaults

    start_sens = 3
    sens_steps = 3
    final_sens = 7    

    # Filters
    pident_thr = score_thr = evalue_thr = query_cov = subject_cov = None # excluded_taxa = None

    # MMseqs2 options
    sub_mat = None

    in_file = None
    itype = None
    translation_table = None

    allow_overlaps = None
    overlap_tol = None

    resume = None

    ##
    def __init__(self, args):

        self.itype = args.itype
        self.translation_table = args.trans_table

        self.allow_overlaps = args.allow_overlaps
        self.overlap_tol = args.overlap_tol

        self.foldseek_cluster_db = args.foldseek_cluster_db

        # default called foldseek.db in the eggnog_db/foldseek/ dir
        # to make gpu db foldseek makepaddedseqdb foldseek.db foldseek.db.gpu --threads 16
        # ln -s foldseek.db_h foldseek.db.gpu_h
        # ln -s foldseek.db foldseek.db.gpu

        # foldseek makepaddedseqdb foldseek.db.clustered.searchDB foldseek.db.clustered.searchDB.gpu --threads 16 --cluster-search 1
        # and various softlinking

        self.gpu = getattr(args, 'gpu', False)

        # Searching the clustered DB is recommended: the full DB is much slower.
        self.targetdb = resolve_foldseek_db(args.foldseek_db, self.gpu,
                                            self.foldseek_cluster_db)

        self.cpu = args.cpu
        self.start_sens = args.start_sens
        self.sens_steps = args.sens_steps
        self.final_sens = args.final_sens
        
        self.query_cov = args.query_cover
        self.subject_cov = args.subject_cover

        self.pident_thr = args.pident
        self.evalue_thr = args.mmseqs_evalue
        self.score_thr = args.mmseqs_score
        # self.excluded_taxa = args.excluded_taxa if args.excluded_taxa else None

        self.sub_mat = args.mmseqs_sub_mat
        
        # self.temp_dir = args.temp_dir
        self.out_dir = args.output_dir

        self.temp_dir = mkdtemp(prefix='emappertmp_foldseek_', dir=args.temp_dir)
        self.no_file_comments = args.no_file_comments

        self.resume = args.resume

        self.gff_ID_field = args.decorate_gff_ID_field

        self.foldseek_top_hits = args.foldseek_top_hits
        self.foldseek_max_seqs = args.foldseek_max_seqs
        self.foldseek_sensitivity = args.foldseek_sensitivity

        # modernprost
        self.batch_size = args.batch_size
        self.max_batch_residues = args.max_batch_residues
        self.profiles = args.profiles
        # HF cache defaults to the 'foldseek' dir inside the eggNOG data dir
        self.hf_cache_dir = getattr(args, 'hf_cache_dir', None) or get_hf_cache_dir()
        self.modernprost_model = getattr(args, 'modernprost_model', None) or MODERNPROST_MODEL

        # 12-state structural alphabet. Two on-disk forms, both understood by
        # Foldseek (structurealign.cpp: queryHas12St = query3Di12St || q12StDbr):
        #   discrete queries  - packed into the _ss db, one byte per residue
        #   --profiles        - a sibling <querydb>_ss12 PSSM database
        self.ss_12st = getattr(args, 'ss_12st', 1)

        # E-value calibration / scoring options (see search_step)
        self.evalue_nn_mode = getattr(args, 'evalue_nn_mode', None)
        self.evalue_12st_profile_comp = getattr(args, 'evalue_12st_profile_comp', None)
        self.use_reverse_score = getattr(args, 'use_reverse_score', 1)



        
        return
    
    ##
    def clear(self):
        if self.temp_dir is not None and pisdir(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except OSError as err:
                print(f"Warning: OS error while removing {self.temp_dir}", file = sys_stderr)
                print(f"OS error: {err}", file = sys_stderr)
        return

    ##
    def search(self, in_file, seed_orthologs_file, hits_file):
        hits_generator = None
        
        require_tool(FOLDSEEK, 'foldseek', 'foldseek')

        # A packed query against a target DB without the 12-state alphabet is
        # not an error - Foldseek falls back to scoring 3Di alone - but the user
        # then silently loses the whole point of --ss_12st, so say so.
        if self.ss_12st:
            target_has_12st = db_has_12st(self.targetdb)
            if target_has_12st is False:
                print(colorify(
                    f"Foldseek target DB {self.targetdb} was built without the 12-state "
                    f"alphabet. The search will fall back to 3Di-only scoring. Rebuild it "
                    f"with a 12-state-capable Foldseek, or upgrade it in place with "
                    f"'foldseek add12st', to benefit from --ss_12st.", 'red'),
                    file=sys_stderr)

        self.in_file = in_file
        
        try:
            cmds = None

            # 1) either resume from previous hits or run foldseek to generate them.
            # Unlike the other searchers, a missing hits file is not fatal here:
            # in the two-stage *-foldseek modes the Foldseek pass may simply not
            # have been reached by the interrupted run, so we search anyway.
            if self.resume == True and pisfile(hits_file):
                print(f"[resume] Skipping Foldseek search — reusing: {hits_file}",
                      file=sys_stderr)
            else:
                if self.resume == True:
                    self.resume = False
                    print(f"Couldn't find hits file {hits_file} to resume.",
                          file=sys_stderr)
                    print("Running Foldseek anyway", file=sys_stderr)

                querydb = pjoin(self.temp_dir, uuid.uuid4().hex)
                resultdb = pjoin(self.temp_dir, uuid.uuid4().hex)
                bestresultdb = pjoin(self.temp_dir, uuid.uuid4().hex)

                alignmentsdb, cmds = self.run_foldseek(
                    in_file, querydb, self.targetdb, resultdb, bestresultdb,
                    self.foldseek_top_hits, self.foldseek_max_seqs,
                    self.foldseek_cluster_db, self.foldseek_sensitivity,
                    self.gpu, self.profiles)
                shutil.copyfile(f'{alignmentsdb}.m8', hits_file)


            # 2) parse search hits to seeds orthologs
            if self.itype == ITYPE_CDS or self.itype == ITYPE_PROTS:
                hits_generator = self._parse_foldseek(hits_file)
                
            else: #self.itype == ITYPE_GENOME or self.itype == ITYPE_META:
                # parse_genepred (without coordinate change)
                hits_generator = self._parse_genepred(hits_file)
                


            # 3) output seeds
            if self.itype == ITYPE_CDS or self.itype == ITYPE_PROTS:
                change_seeds_coords = False
            else: #self.itype == ITYPE_GENOME or self.itype == ITYPE_META:
                # change seeds coordinates relative to the ORF, not to the contig (to use them for the .seed_orthologs file)
                change_seeds_coords = True
                
            hits_generator = output_seeds(cmds, hits_generator,
                                          seed_orthologs_file,
                                          self.no_file_comments,
                                          change_seeds_coords)

        except Exception as e:
            raise e
            
        return hits_generator
    
    
    def run_foldseek(self, fasta_file, querydb, targetdb, resultdb, bestresultdb, foldseek_top_hits, foldseek_max_seqs, foldseek_cluster_db, foldseek_sensitivity, gpu, profiles):
        cmds = []

        handle = None
        try:
            if self.itype == ITYPE_CDS:
                # CDS always translated to proteins before search; ModernProst
                # predicts the structural alphabets from amino acids.
                handle, query_file = mkstemp(dir = self.temp_dir, text = True)
                translate_cds_to_prots(fasta_file, query_file, self.translation_table)
            else:
                query_file = fasta_file

                        

            cmd = self.createdb(query_file, querydb)
            cmds.append(cmd)

            if profiles:
                # update the query db
                querydb : Path = Path(self.out_dir) / f"query_profiledb" / f"query_profile"

            cmd = self.search_step(querydb, targetdb, resultdb, foldseek_max_seqs, foldseek_cluster_db, foldseek_sensitivity, gpu)
            cmds.append(cmd)

            if self.itype == ITYPE_CDS or self.itype == ITYPE_PROTS:
                cmd = self.filterdb_step(resultdb, bestresultdb, foldseek_top_hits)
                cmds.append(cmd)
            else:
                bestresultdb = resultdb

            cmd = self.convertalis_step(querydb, targetdb, bestresultdb)
            cmds.append(cmd)
        finally:
            if handle is not None:
                os.close(handle)

        return bestresultdb, cmds

    def createdb(self, fasta_file, querydb):

        # run ModernProst inference
        cache_dir = self.hf_cache_dir
        if self.profiles:
            # create profiles
            infer_3Di(fasta_file, self.out_dir, cache_dir, self.batch_size,
                      self.max_batch_residues, profiles=True,
                      model_name=self.modernprost_model)

        else:
            output_3di, output_12st = infer_3Di(
                fasta_file, self.out_dir, cache_dir, self.batch_size,
                self.max_batch_residues, profiles=False,
                model_name=self.modernprost_model)

            if self.ss_12st == 0:
                # user asked for 3Di only; write a legacy single-alphabet _ss
                output_12st = None
            elif output_12st is None:
                raise EmapperException(
                    "The ModernProst model produced no 12-state predictions, but "
                    "the 12-state alphabet is enabled. Re-run with --ss_12st 0 to "
                    "search with 3Di alone."
                )

            # converts to a foldseek querydb
            generate_foldseek_db_from_aa_3di(
                fasta_file, output_3di, self.temp_dir, querydb,
                fasta_12st=output_12st
            )

        # mmseqs createdb examples/QUERY.fasta queryDB
        # cmd = (
        #     f'{MMSEQS2} createdb \'{fasta_file}\' \'{querydb}\''
        # )
        # if self.itype == ITYPE_PROTS or (self.itype == ITYPE_CDS and self.translate == True):
        #     cmd += ' --dbtype 1' # aas queries (proteins)
        # else:
        #     cmd += ' --dbtype 2' # nts queries (CDS, contig, ...)
            
        # print(colorify('  '+cmd, 'yellow'))
        # try:
        #     completed_process = subprocess.run(cmd, capture_output=True, check=True, shell=True)
        # except subprocess.CalledProcessError as cpe:
        #     raise EmapperException("Error running 'mmseqs createdb': "+cpe.stderr.decode("utf-8").strip().split("\n")[-1])
        # return cmd

    def search_step(self, querydb, targetdb, resultdb, foldseek_max_seqs, foldseek_cluster_db, foldseek_sensitivity, gpu):
        # mmseqs search queryDB targetDB resultDB tmp
        # cmd = (
        #     f'{MMSEQS2} search -a true \'{querydb}\' \'{targetdb}\' \'{resultdb}\' \'{self.temp_dir}\' '
        #     f'--start-sens {self.start_sens} --sens-steps {self.sens_steps} -s {self.final_sens} '
        #     f'--threads {self.cpu}'
        # )

        # basic foldseek search

        # assume the target db has been padded with suffix gpu

        cmd = (
                f'{FOLDSEEK} search -a true \'{querydb}\' \'{targetdb}\' \'{resultdb}\' \'{self.temp_dir}\' '
                f'--threads {self.cpu} --max-seqs {foldseek_max_seqs}'
        
        )
        if gpu:

            cmd += f" --gpu 1 --prefilter-mode 1 "


        else:

            cmd += f" -s {foldseek_sensitivity} "
  

        if foldseek_cluster_db:

            cmd += f" --cluster-search 1 "

        # Gate the 12-state structural alphabet. Foldseek maps --ss-12st onto
        # --aux-score in the search workflow, so this one flag covers both the
        # prefilter and the alignment stage.
        cmd += f" --ss-12st {self.ss_12st} "

        # E-value calibration must match the scoring. Foldseek defaults to
        # --evalue-nn-mode 0, the 3Di-only composition network; scoring with the
        # 12-state channel while calibrating without it inflates significance.
        # Mode 2 is the 3Di+12-state network and requires a 12-state query,
        # which is exactly what self.ss_12st guarantees.
        if self.evalue_nn_mode is not None:
            cmd += f" --evalue-nn-mode {self.evalue_nn_mode} "
        elif self.ss_12st:
            cmd += " --evalue-nn-mode 2 "

        # Only meaningful for mode 2 with profile queries: where the network
        # takes the query composition from.
        if self.profiles and self.evalue_12st_profile_comp is not None:
            cmd += f" --evalue-12st-profile-comp {self.evalue_12st_profile_comp} "

        # Subtract the reverse-alignment score from the forward score. Foldseek
        # already defaults this to 1; emitted explicitly so the command line
        # records it rather than depending on the build's default.
        cmd += f" --use-reverse-score {self.use_reverse_score} "

        # if self.sub_mat is not None: cmd += f' --sub-mat {self.sub_mat}'
        # if self.translation_table is not None: cmd += f' --translation-table {self.translation_table}'
        
        print(colorify('  '+cmd, 'yellow'))
        try:
            completed_process = subprocess.run(cmd, capture_output=True, check=True, shell=True)
        except subprocess.CalledProcessError as cpe:
            raise EmapperException("Error running 'foldseek search': "+cpe.stderr.decode("utf-8").strip().split("\n")[-1])
        return cmd

    ##
    def filterdb_step(self, resultdb, bestresultdb, foldseek_top_hits):
        # mmseqs filterdb resultDB bestResultDB --extract-lines 1
        # mmseqs filterdb resultDB bestResultDB --extract-lines {foldseek_top_hits}
        cmd = (
            f'{FOLDSEEK} filterdb \'{resultdb}\' \'{bestresultdb}\' --threads {self.cpu}'
        )
        cmd += f" --extract-lines \'{foldseek_top_hits}\' "
        
        print(colorify('  '+cmd, 'yellow'))
        try:
            completed_process = subprocess.run(cmd, capture_output=True, check=True, shell=True)
        except subprocess.CalledProcessError as cpe:
            raise EmapperException("Error running 'foldseek filterdb': "+cpe.stderr.decode("utf-8").strip().split("\n")[-1])                
        
        return cmd
    
    ##
    def convertalis_step(self, querydb, targetdb, resultdb):
        # mmseqs convertalis queryDB targetDB resultDB resultDB.m8
        cmd = (
            f'{FOLDSEEK} convertalis \'{querydb}\' \'{targetdb}\' \'{resultdb}\' \'{resultdb}.m8\' --threads {self.cpu}'
        )
        # if self.sub_mat is not None: cmd += f' --sub-mat {self.sub_mat}'
            
        # if self.translation_table is not None: cmd += f' --translation-table {self.translation_table}'

        # outfmt - should be the same for Foldseek


        cmd += f' --format-output "query,target,pident,alnlen,qstart,qend,tstart,tend,evalue,bits,qcov,tcov"'
        print(colorify('  '+cmd, 'yellow'))
        try:
            completed_process = subprocess.run(cmd, capture_output=True, check=True, shell=True)
        except subprocess.CalledProcessError as cpe:
            raise EmapperException("Error running 'mmseqs convertalis': "+cpe.stderr.decode("utf-8").strip().split("\n")[-1])        
        return cmd
    
    ##
    def _parse_foldseek(self, raw_foldseek_file):

        prev_query = None
        # parse hits
        with open(raw_foldseek_file, 'r') as raw_f:
            for line in raw_f:
                if not line.strip() or line.startswith('#'):
                    continue
                
                fields = list(map(str.strip, line.split('\t')))
                # check fields in convertalis_step()
        
                query = fields[0]
                            
                # only one result per query
                if prev_query is not None and query == prev_query:
                    continue
                else:
                    prev_query = query
                    
                pident = float(fields[2])
                evalue = float(fields[8])
                score = float(fields[9])
                qcov = float(fields[10]) * 100 # mmseqs uses 0-1 values
                scov = float(fields[11]) * 100 # mmseqs uses 0-1 values

                # note: this could be done with mmmseqs filterdb, but I dont know how to do it in a single step
                if ((self.pident_thr is not None and pident < self.pident_thr) or 
                    (self.evalue_thr is not None and evalue > self.evalue_thr) or
                    (self.score_thr is not None and score < self.score_thr) or
                    (self.query_cov is not None and qcov < self.query_cov) or
                    (self.subject_cov is not None and scov < self.subject_cov)):
                    continue
                
                target = fields[1]
                length = int(fields[3])
                qstart = int(fields[4])
                qend = int(fields[5])
                sstart = int(fields[6])
                send = int(fields[7])

                hit = [query, target, evalue, score, qstart, qend, sstart, send, pident, qcov, scov]
                yield hit # hit
        return

    ##
    def _parse_genepred(self, raw_foldseek_file):
        
        curr_query_hits = []
        prev_query = None
        queries_suffixes = {}        
        
        with open(raw_foldseek_file, 'r') as raw_f:
            for line in raw_f:
                if not line.strip() or line.startswith('#'):
                    continue
                
                fields = list(map(str.strip, line.split('\t')))
                # check fields in convertalis_step()
                # cmd += f' --format-output "query,target,pident,alnlen,
                #qstart,qend,tstart,tend,evalue,bits,qcov,tcov"'

                query = fields[0]
                pident = float(fields[2])
                evalue = float(fields[8])
                score = float(fields[9])
                qcov = float(fields[10]) * 100 # mmseqs uses 0-1 values
                scov = float(fields[11]) * 100 # mmseqs uses 0-1 values

                # note: this could be done with mmmseqs filterdb, but I dont know how to do it in a single step
                if ((self.pident_thr is not None and pident < self.pident_thr) or 
                    (self.evalue_thr is not None and evalue > self.evalue_thr) or
                    (self.score_thr is not None and score < self.score_thr) or
                    (self.query_cov is not None and qcov < self.query_cov) or
                    (self.subject_cov is not None and scov < self.subject_cov)):
                    continue
                
                # query = fields[0]
                target = fields[1]
                length = int(fields[3])
                qstart = int(fields[4])
                qend = int(fields[5])
                sstart = int(fields[6])
                send = int(fields[7])
                
                hit = [query, target, evalue, score, qstart, qend, sstart, send, pident, qcov, scov]

                if query == prev_query:
                    if self.allow_overlaps == ALLOW_OVERLAPS_ALL:
                        yield [f"{hit[0]}_{suffix}"]+hit[1:] # hit
                        
                    else:
                        if not hit_does_overlap(hit, curr_query_hits, self.allow_overlaps, self.overlap_tol):
                            if query in queries_suffixes:
                                queries_suffixes[query] += 1
                                suffix = queries_suffixes[query]
                            else:
                                suffix = 0
                                queries_suffixes[query] = suffix

                            yield [f"{hit[0]}_{suffix}"]+hit[1:] # hit
                            curr_query_hits.append(hit)
                else:
                    if query in queries_suffixes:
                        queries_suffixes[query] += 1
                        suffix = queries_suffixes[query]
                    else:
                        suffix = 0
                        queries_suffixes[query] = suffix
                            
                    yield [f"{hit[0]}_{suffix}"]+hit[1:] # hit
                    curr_query_hits = [hit]
                    
                prev_query = query    
                
        return
    
## END
