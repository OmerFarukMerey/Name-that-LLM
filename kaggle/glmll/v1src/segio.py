"""Read and write the binary files of ntl-segrun (see the header of tools/ntl-segrun/ntl-segrun.cpp)."""
import struct

import numpy as np


def write_texts(path, texts):
    with open(path, 'wb') as f:
        f.write(b'NTLT')
        f.write(struct.pack('<I', len(texts)))
        for t in texts:
            b = t.encode('utf-8')
            f.write(struct.pack('<I', len(b)))
            f.write(b)


def write_tokens(path, seqs):
    """seqs: list of (ids, n_prefix)."""
    with open(path, 'wb') as f:
        f.write(b'NTLK')
        f.write(struct.pack('<I', len(seqs)))
        for ids, n_prefix in seqs:
            ids = np.asarray(ids, dtype='<i4')
            f.write(struct.pack('<II', len(ids), n_prefix))
            f.write(ids.tobytes())


def read_tokens(path):
    out = []
    with open(path, 'rb') as f:
        assert f.read(4) == b'NTLK'
        (n,) = struct.unpack('<I', f.read(4))
        for _ in range(n):
            nt, npre = struct.unpack('<II', f.read(8))
            ids = np.frombuffer(f.read(4 * nt), dtype='<i4')
            out.append((ids, npre))
    return out


def read_scores(path):
    """List of float32 arrays [n_scored, 4]: logprob, entropy, E[logp^2], rank."""
    out = []
    with open(path, 'rb') as f:
        assert f.read(4) == b'NTLP'
        n, nf = struct.unpack('<II', f.read(8))
        for _ in range(n):
            (ns,) = struct.unpack('<I', f.read(4))
            out.append(np.frombuffer(f.read(4 * ns * nf), dtype='<f4').reshape(ns, nf))
    return out


def read_state(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'NTLS'
        dt, w, nt, nr = struct.unpack('<IIIQ', f.read(20))
        dtype = {0: '<f4', 1: '<f2'}.get(dt)
        if dtype is None:  # bf16
            raw = np.frombuffer(f.read(2 * w * nr), dtype='<u2').astype('<u4') << 16
            return raw.view('<f4').reshape(nr, w)
        return np.frombuffer(f.read(np.dtype(dtype).itemsize * w * nr), dtype=dtype).reshape(nr, w).astype('f4')
