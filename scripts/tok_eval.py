"""
Evaluate compression ratio of tokenizers.

Compares our tokenizer against GPT-2 and GPT-4 tokenizers
on various text types: news, Korean, code, math, science, and training data.
"""

import os
import argparse
from pathlib import Path

import tiktoken
import pyarrow.parquet as pq

from attractor.tokenizer import Tokenizer

# -----------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/resource/data")

DATASETS = {
    "fineweb": f"{DATA_DIR}/fineweb-edu-100b-shuffle",
    "huginn": f"{DATA_DIR}/huginn-dataset",
}


def resolve_data_dir(data_dir):
    """Resolve dataset name to path, or return as-is if it's already a path."""
    if data_dir in DATASETS:
        return DATASETS[data_dir]
    return data_dir


# -----------------------------------------------------------------------------
# Sample texts for evaluation

# Random text I got from a random website this morning
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation's food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

"The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening's to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border," said U.S. Secretary of Agriculture Brooke L. Rollins. "Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest."
""".strip()

# Random Korean text (to test non-English compression)
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 '정보의 균형'을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 '무엇이 쟁점인지', '누가 무엇을 주장하는지', '사실은 무엇인지'**를 명확히 전달하는 데 집중합니다.
""".strip()

# Random piece of code
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^2,
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square''—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere's primary productivity.
""".strip()

# -----------------------------------------------------------------------------
# Tokenizer wrappers for consistent interface


class TiktokenWrapper:
    """Wrapper around tiktoken for consistent interface."""

    def __init__(self, encoding):
        self.enc = encoding

    @classmethod
    def from_pretrained(cls, name):
        """Load a tiktoken encoding by name (e.g., 'gpt2', 'cl100k_base')."""
        enc = tiktoken.get_encoding(name)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)


class HFTokenizerWrapper:
    """Wrapper around our HuggingFace tokenizer for consistent interface."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer = Tokenizer.from_directory(tokenizer_dir)
        return cls(tokenizer)

    @classmethod
    def from_pretrained(cls, model_name):
        tokenizer = Tokenizer.from_pretrained(model_name)
        return cls(tokenizer)

    def get_vocab_size(self):
        return len(self.tokenizer)

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids)


# -----------------------------------------------------------------------------
# Data loading utilities


def list_parquet_files(data_dir):
    """Returns full paths to all parquet files in a directory."""
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("*.parquet"))
    return [str(f) for f in parquet_files if not str(f).endswith('.tmp')]


def parquets_iter_batched(data_dir, split="train", start=0, step=1):
    """
    Iterate through parquet files, yielding batches of text.

    Args:
        data_dir: Directory containing parquet files
        split: "train" or "val" - val uses only the last file
        start: Starting row group index (for DDP)
        step: Step between row groups (for DDP)
    """
    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        return

    if split == "val":
        parquet_paths = parquet_paths[-1:]
    else:
        parquet_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            if 'text' in rg.column_names:
                texts = rg.column('text').to_pylist()
                yield texts


# -----------------------------------------------------------------------------
# Evaluation functions


def evaluate_tokenizer(tokenizer, text):
    """
    Evaluate a tokenizer on a piece of text.

    Returns:
        dict with bytes, tokens, and compression ratio
    """
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    # Verify roundtrip
    if decoded != text:
        print(f"Warning: Roundtrip failed! Decoded length: {len(decoded)}, Original: {len(text)}")

    encoded_bytes = text.encode('utf-8')
    ratio = len(encoded_bytes) / len(encoded) if len(encoded) > 0 else 0

    return {
        'bytes': len(encoded_bytes),
        'tokens': len(encoded),
        'ratio': ratio
    }


def print_comparison(baseline_name, baseline_results, ours_results, all_text):
    """Print comparison table between baseline tokenizer and ours."""
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, text in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        # Calculate relative difference (positive means ours is better)
        relative_diff = ((baseline_data['tokens'] - ours_data['tokens']) / baseline_data['tokens']) * 100

        # Determine which has better compression (higher ratio = better)
        if baseline_data['ratio'] > ours_data['ratio']:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data['ratio'] > baseline_data['ratio']:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(f"{name:<10} {baseline_data['bytes']:<8} "
              f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
              f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
              f"{ours_color}{ours_data['tokens']:<7}{RESET} "
              f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
              f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
              f"{better:<10}")


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate tokenizer compression ratios')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer directory or HuggingFace model name')
    parser.add_argument('--data-dir', type=str, default=None,
                        help=f'Dataset name ({", ".join(DATASETS.keys())}) or path to parquet directory')
    args = parser.parse_args()

    # Resolve dataset name to path if needed
    data_dir = resolve_data_dir(args.data_dir) if args.data_dir else None

    # Build list of texts to evaluate
    all_text = [
        ("news", news_text),
        ("korean", korean_text),
        ("code", code_text),
        ("math", math_text),
        ("science", science_text),
    ]

    # Add training/validation data if available
    if data_dir and os.path.isdir(data_dir):
        try:
            train_docs = next(parquets_iter_batched(data_dir, split="train"), None)
            if train_docs:
                train_text = "\n".join(train_docs)
                all_text.append(("data-train", train_text))

            val_docs = next(parquets_iter_batched(data_dir, split="val"), None)
            if val_docs:
                val_text = "\n".join(val_docs)
                all_text.append(("data-val", val_text))
        except Exception as e:
            print(f"Warning: Could not load parquet data: {e}")

    # Initialize tokenizers
    tokenizer_results = {}
    vocab_sizes = {}

    # GPT-2 tokenizer (tiktoken)
    print("Loading GPT-2 tokenizer...")
    gpt2_tokenizer = TiktokenWrapper.from_pretrained("gpt2")
    vocab_sizes["gpt2"] = gpt2_tokenizer.get_vocab_size()
    tokenizer_results["gpt2"] = {}

    # GPT-4 tokenizer (tiktoken cl100k_base)
    print("Loading GPT-4 tokenizer...")
    gpt4_tokenizer = TiktokenWrapper.from_pretrained("cl100k_base")
    vocab_sizes["gpt4"] = gpt4_tokenizer.get_vocab_size()
    tokenizer_results["gpt4"] = {}

    # Our tokenizer
    ours_tokenizer = None
    if args.tokenizer:
        print(f"Loading tokenizer from {args.tokenizer}...")
        try:
            if os.path.isdir(args.tokenizer):
                ours_tokenizer = HFTokenizerWrapper.from_directory(args.tokenizer)
            else:
                ours_tokenizer = HFTokenizerWrapper.from_pretrained(args.tokenizer)
            vocab_sizes["ours"] = ours_tokenizer.get_vocab_size()
            tokenizer_results["ours"] = {}
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            ours_tokenizer = None

    # Evaluate all tokenizers
    tokenizers_to_eval = [("gpt2", gpt2_tokenizer), ("gpt4", gpt4_tokenizer)]
    if ours_tokenizer:
        tokenizers_to_eval.append(("ours", ours_tokenizer))

    for tokenizer_name, tokenizer in tokenizers_to_eval:
        for name, text in all_text:
            tokenizer_results[tokenizer_name][name] = evaluate_tokenizer(tokenizer, text)

    # Print vocab sizes
    print(f"\nVocab sizes:")
    print(f"  GPT-2: {vocab_sizes['gpt2']:,}")
    print(f"  GPT-4: {vocab_sizes['gpt4']:,}")
    if ours_tokenizer:
        print(f"  Ours:  {vocab_sizes['ours']:,}")

    # Print comparisons
    if ours_tokenizer:
        print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['ours'], all_text)
        print_comparison("GPT-4", tokenizer_results['gpt4'], tokenizer_results['ours'], all_text)
    else:
        # Just compare GPT-2 vs GPT-4 if no custom tokenizer
        print_comparison("GPT-2", tokenizer_results['gpt2'], tokenizer_results['gpt4'], all_text)
        print("\nNote: No custom tokenizer specified. Use --tokenizer to compare your own.")
