# eggnog-mapper with the Foldseek 12-state structural search modes.
#
# Base image matches phold's container so both run on the same HPC systems
# (Pawsey/Setonix, ROCm). Torch comes from the base image and must survive
# every pip install below -- see the guard rails at the end.
ARG BASE_IMAGE=quay.io/pawsey/pytorch:2.7.1-rocm6.3.3

# ---------------------------------------------------------------------------
# Stage 1: build Foldseek with 12-state support.
#
# Bioconda's foldseek has no --ss-12st, and neither does the prebuilt tarball
# from mmseqs.com that phold's container uses, so this one is compiled from
# source. Built in a separate stage FROM the same base: the toolchain (rust,
# cmake, build-essential, ~2GB) stays out of the shipped image, while the glibc
# the binary links against is by construction the one it will run on.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS foldseek-builder

# Dependency list taken from Foldseek's own Dockerfile. xxd is not optional --
# the build shells out to it to embed data/*.out (the substitution matrices,
# including mat12st.out) into the binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl xxd git cmake \
        zlib1g-dev libbz2-dev libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Rust >= 1.68 is required: lib/mmseqs pulls in block-aligner through corrosion
# and hard-fails the configure step below that version. rustup's minimal
# profile is enough (no docs, no clippy).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- --profile minimal --default-host x86_64-unknown-linux-gnu -y
ENV PATH="/root/.cargo/bin:${PATH}"

ARG FOLDSEEK_REPO=https://github.com/Victor-Mihaila/foldseek.git
ARG FOLDSEEK_REF=master

# Shallow, non-recursive clone. The three entries in .gitmodules are all
# regression-test fixtures marked `update = none`, and one has an empty URL
# that makes `--recursive` fail outright; -DHAVE_TESTS=0 means none are needed.
RUN git clone --depth 1 --branch "${FOLDSEEK_REF}" "${FOLDSEEK_REPO}" /opt/foldseek_src

WORKDIR /opt/foldseek_src/build
RUN cmake \
        -DHAVE_AVX2=1 \
        -DHAVE_MPI=0 \
        -DHAVE_TESTS=0 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/foldseek \
        .. \
    && make -j "$(nproc --all)" \
    && make install

# Fail the build here rather than at search time. A Foldseek without --ss-12st
# still runs every eggnog-mapper foldseek command and silently scores 3Di-only,
# which is the exact failure this whole image exists to avoid.
RUN /opt/foldseek/bin/foldseek search --help 2>&1 | grep -q -- '--ss-12st' \
    && echo "OK: foldseek provides --ss-12st"

# The other three flags emapper passes to `foldseek search`. Reported, not
# asserted: MMseqs-family tools hide expert parameters from default --help, so
# absence here is not proof they are unsupported.
RUN for f in --evalue-nn-mode --evalue-12st-profile-comp --use-reverse-score; do \
        if /opt/foldseek/bin/foldseek search --help 2>&1 | grep -q -- "$f"; then \
            echo "  present in search --help: $f"; \
        else \
            echo "  NOT in search --help (may be an expert param): $f"; \
        fi; \
    done

# ---------------------------------------------------------------------------
# Stage 2: the shipped image.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

# Foldseek's runtime shared libraries (its Dockerfile's runtime stage installs
# these). Most are already in the base image; naming them keeps the image
# correct if the base ever slims down.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget tar aria2 gawk \
        libstdc++6 libgomp1 libatomic1 zlib1g libbz2-1.0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=foldseek-builder /opt/foldseek /opt/foldseek
ENV PATH="/opt/foldseek/bin:${PATH}"

# The sequence-search backends. eggnog-mapper resolves all of these through
# shutil.which, so they only need to be on PATH. DIAMOND and MMseqs2 back
# -m diamond / -m mmseqs and therefore the two-stage -m diamond-foldseek and
# -m mmseqs-foldseek modes; prodigal backs --itype genome/metagenome.
ARG DIAMOND_VERSION=v2.2.5
ARG MMSEQS_VERSION=18-8cc5c
ARG PRODIGAL_VERSION=v2.6.3
RUN mkdir -p /opt/bin \
    && wget -q "https://github.com/bbuchfink/diamond/releases/download/${DIAMOND_VERSION}/diamond-linux64.tar.gz" \
    && tar -xzf diamond-linux64.tar.gz diamond && mv diamond /opt/bin/ && rm diamond-linux64.tar.gz \
    && wget -q "https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS_VERSION}/mmseqs-linux-avx2.tar.gz" \
    && tar -xzf mmseqs-linux-avx2.tar.gz && mv mmseqs/bin/mmseqs /opt/bin/ \
    && rm -rf mmseqs mmseqs-linux-avx2.tar.gz \
    && wget -q -O /opt/bin/prodigal "https://github.com/hyattpd/Prodigal/releases/download/${PRODIGAL_VERSION}/prodigal.linux" \
    && chmod +x /opt/bin/prodigal
ENV PATH="/opt/bin:${PATH}"

# The base image ships python3 but no `python`; emapper.py and the scripts/
# helpers assume the latter.
RUN ln -sf "$(command -v python3)" /usr/local/bin/python

# Carried over from phold's container: the base image's PyYAML is installed in
# a way pip refuses to upgrade cleanly.
RUN pip install --ignore-installed PyYAML

# Record the torch we must not break.
RUN python3 -c "import torch; print('torch before install:', torch.__version__)"

# eggnogmapper/annotator ships two Cython extensions (the delta-varint codec
# and the ortholog-collector inner loop). setup.py only catches Cython being
# absent -- if Cython is importable but no compiler is, the C build fails and
# takes the whole install with it. So provide both explicitly rather than
# depending on what the base image happens to carry.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "cython>=3.0"

# Installed from the build context rather than `pip install git+...@REF`
# (phold's approach). Two reasons: on a pull_request build github.sha is the
# merge commit, which is not reliably fetchable from the repo URL, so a
# git-ref install turns every PR red; and copying the tree means tests/ is
# present, which the wheel does not ship -- so the suite can run below.
COPY . /opt/eggnog-mapper
RUN pip install /opt/eggnog-mapper

# transformers and tqdm are the rest of the [foldseek] extra. They are
# installed WITHOUT the extra deliberately: `pip install eggnog-mapper[foldseek]`
# declares torch>=2.0,<3.0, and resolving that against PyPI would replace the
# base image's ROCm torch with a generic CUDA/CPU wheel that cannot see the GPU.
# transformers does not depend on torch, so this is safe on its own.
RUN pip install transformers tqdm

# ---------------------------------------------------------------------------
# Build-time smoke tests. Each one has failed for real at some point.
# ---------------------------------------------------------------------------

# torch is still the base image's ROCm build, not a PyPI replacement. Tested
# via torch.version.hip rather than the version string: ROCm wheels do not
# reliably carry "rocm" in __version__, but hip is None on every CUDA/CPU build.
RUN python3 -c "import torch; print('torch after install:', torch.__version__, 'hip:', torch.version.hip); assert torch.version.hip is not None, 'ROCm torch was replaced by a non-ROCm build'"

RUN foldseek --help > /dev/null && echo "OK: foldseek runs"
RUN foldseek search --help 2>&1 | grep -q -- '--ss-12st' && echo "OK: 12-state support survived the copy"
RUN diamond --version && mmseqs version && { prodigal -v 2>&1 || true; } | head -1

# --help, not --version: --version opens the annotation DB, which this image
# deliberately does not ship (databases are bind-mounted at run time).
RUN emapper.py --help > /dev/null && echo "OK: emapper runs"
# The Foldseek modes are only in this branch; a build off the wrong ref would
# otherwise produce a working-looking image with no 12-state options.
RUN emapper.py --help | grep -q -- '--ss_12st' && echo "OK: foldseek CLI options present"
RUN emapper.py --help | grep -q 'mmseqs-foldseek' && echo "OK: two-stage modes present"

# Import the searcher for real. This is the only check that exercises the
# torch + transformers import path inside the package, which is where a broken
# extra shows up -- FoldseekSearcher is imported lazily at runtime, so a plain
# `emapper.py --help` would not touch it.
RUN python3 -c "from eggnogmapper.search.foldseek.foldseek import FoldseekSearcher; print('OK: FoldseekSearcher imports')"
RUN python3 -c "from eggnogmapper.search.foldseek.create_fs_db import pack_3di_12st; assert max(pack_3di_12st('AY','AN')) == 239; print('OK: 3Di+12st packing')"

# The compiled annotator path is in use. Falling back to pure Python is
# correct (byte-identical results) but slower, and it happens silently -- so
# assert rather than discover it in a profile six weeks from now.
RUN python3 -c "from eggnogmapper.annotator.codec import _BACKEND; print('codec backend:', _BACKEND); assert _BACKEND == 'cython', 'Cython extensions did not build'"

# The unit suite. Runs from /opt/eggnog-mapper because tests/ is not in the
# installed wheel. Expect 82 passed / 94 skipped: the skips are pre-existing
# and want a v7 sample DB that is not shipped here.
RUN pip install pytest \
    && cd /opt/eggnog-mapper \
    && python3 -m pytest tests/unit eggnogmapper/annotator/tests -q

RUN python --version
