# magnetic/model.py
#
# MagneticModel is the thin glue object that holds every component of
# a trained MagneticLM instance: vocabulary, n-gram tables, semantic
# edges, position embedding, importance vector, and the excitation
# engine. It exposes one public `train` method that runs the full
# pipeline end to end and leaves the model ready for the generator
# and evaluator.

import array
import sys
import time
from typing import Iterable, List

try:
    import numpy as np
    import torch
except ImportError:
    print("ERROR: PyTorch and NumPy required.", file=sys.stderr)
    sys.exit(1)

from .config import MagneticConfig
from .tokenizer import Vocabulary, tokenize
from .ngram import NgramTables
from .edges import EdgeBuilder, EdgeSet
from .physics import PhysicsSimulator, compute_importance
from .excitation import ExcitationEngine


class MagneticModel:
    """A trained MagneticLM.

    After calling `train(lines)` the following fields are populated:
      vocab       - Vocabulary instance
      ngram       - NgramTables with Modified KN-5 scoring ready
      edges       - EdgeSet (semantic edge list)
      positions   - (V, dim) physics embedding
      importance  - (V,) per-node importance score
      excitation  - ExcitationEngine ready for inference
      freq_gpu    - (V,) int64 unigram counts
    """

    def __init__(
        self,
        config: MagneticConfig,
        device: torch.device,
        aux_device: torch.device = None,
    ):
        self.config = config
        self.device = device
        self.aux_device = aux_device or device
        self.multi_gpu = (self.aux_device != self.device)

        self.vocab = Vocabulary()
        self.ngram = NgramTables(config, device, self.aux_device)
        self._edge_builder = EdgeBuilder(config)
        self._physics = PhysicsSimulator(config)

        self.freq_gpu: torch.Tensor = None
        self.edges: EdgeSet = None
        self.positions: torch.Tensor = None
        self.importance: torch.Tensor = None
        self.excitation: ExcitationEngine = None

    # -------------------------------------------------------------------
    # Tokenization stage
    # -------------------------------------------------------------------
    def _tokenize_to_gpu(self, lines: Iterable[str]) -> torch.Tensor:
        """Convert an iterable of raw lines into a single int64 GPU
        tensor, building up the vocabulary along the way."""
        w2i = self.vocab.word2id
        id2w = self.vocab.id2word
        buf = array.array('i')
        n_lines = 0
        n_tok = 0
        t0 = time.time()
        for line in lines:
            words = tokenize(line)
            if len(words) < 2:
                continue
            n_lines += 1
            for w in words:
                tid = w2i.get(w)
                if tid is None:
                    tid = len(id2w)
                    w2i[w] = tid
                    id2w.append(w)
                buf.append(tid)
            n_tok += len(words)
        print("  Tokenizing: %d lines, %d tokens (%.0fs) done." %
              (n_lines, n_tok, time.time() - t0))
        np_arr = np.frombuffer(buf, dtype=np.int32)
        return torch.from_numpy(np_arr).to(
            device=self.device, dtype=torch.int64)

    # -------------------------------------------------------------------
    # Full training pipeline
    # -------------------------------------------------------------------
    def train(self, lines: List[str]):
        """Full training: tokenize, build n-grams, build edges, run
        physics, compute importance, initialize the excitation engine.
        """
        t_all = time.time()
        cfg = self.config

        print("  Tokenizing corpus...")
        tokens_gpu = self._tokenize_to_gpu(lines)
        T = tokens_gpu.numel()
        V = len(self.vocab)
        print("  Tokens: %d, Vocab: %d" % (T, V))
        if T == 0 or V == 0:
            return

        # Frequency counts (needed by PMI and importance).
        self.freq_gpu = torch.zeros(V, dtype=torch.int64, device=self.device)
        self.freq_gpu.scatter_add_(
            0, tokens_gpu,
            torch.ones_like(tokens_gpu))

        # N-gram tables (may shard across two GPUs).
        self.ngram.build(tokens_gpu, V, verbose=True)

        # Semantic edges (with optional PMI reweighting).
        t0 = time.time()
        print("  Semantic edges (window +/-%d, PMI=%s, Jaccard=%s)..." %
              (cfg.edge_window, cfg.use_pmi, cfg.use_jaccard),
              end="", flush=True)
        self.edges = self._edge_builder.build(
            tokens_gpu, self.freq_gpu, V, self.device)
        print(" done (%.0fs, %d edges)" % (
            time.time() - t0, self.edges.numel()))

        # Token stream is no longer needed.
        del tokens_gpu
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Physics simulation.
        self.positions, _ = self._physics.run(
            V,
            self.edges.edge_from,
            self.edges.edge_to,
            self.edges.edge_weight,
            self.device,
            verbose=True)

        # Node importance.
        self.importance = compute_importance(
            V, self.edges.edge_from, self.freq_gpu, self.device)

        # Excitation engine (ready for inference). Pass freq_gpu so
        # prompt excitation can be IDF-weighted (common words in the
        # prompt contribute less, preventing specialist-phrase dominance).
        self.excitation = ExcitationEngine(
            cfg,
            self.edges.edge_from,
            self.edges.edge_to,
            self.edges.edge_weight,
            V,
            self.device,
            freq_gpu=self.freq_gpu)

        print("  Model ready. Total time: %.0fs" % (time.time() - t_all))

    # -------------------------------------------------------------------
    # Memory helpers
    # -------------------------------------------------------------------
    def memory_summary(self) -> str:
        if self.device.type != "cuda":
            return "(CPU)"
        lines = []
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / 1024 ** 2
            peak = torch.cuda.max_memory_allocated(i) / 1024 ** 2
            lines.append("GPU%d  cur=%.0f MiB  peak=%.0f MiB" %
                         (i, used, peak))
        return " | ".join(lines)
