# Experiment Document: Compression Simulation from Long-Context Attention to a Temporary Parameterized FFN

## 1. Experiment Name

**Context-to-Temporary-FFN: Compression Simulation from Long-Context KV Cache to Temporary FFN Parameter Blocks**

## 2. Background

During long-context inference in Transformer-based large language models, each layer and each attention head stores the Key and Value vectors for historical tokens:

$$
K_C, V_C \in \mathbb{R}^{N \times d}
$$

where $N$ is the context length and $d$ is the hidden dimension of a single attention head. As $N$ increases, the memory cost of the KV cache grows linearly. For a multi-layer and multi-head model, the KV cache size is approximately:

$$
2NLHd
$$

where $L$ is the number of layers and $H$ is the number of attention heads.

The core computation of long-context attention is:

$$
F_C(q) =
\operatorname{softmax}
\left(
\frac{qK_C^\top}{\sqrt d}
\right)V_C
$$

where $q$ is the query representation of the current token, and $K_C,V_C$ are the Key and Value tensors corresponding to the long context.

From a functional perspective, for a fixed context $C$, full attention defines a mapping from a query to a value aggregation result:

$$
q \mapsto F_C(q)
$$

This experiment investigates whether token-level KV cache from a long context can be compressed into a smaller temporary parameter module that approximates the output of full attention.

## 3. Core Objective

The core objective is to validate the following hypothesis:

> For a fixed long context $C$, full attention can be equivalently viewed as a context-generated dynamic softmax-FFN. Furthermore, it may be possible to approximate this attention function with a temporary FFN parameter block that has fewer hidden units, thereby reducing reliance on the full KV cache.

The original attention can be written as:

$$
F_C(q) =
\operatorname{softmax}
\left(
\frac{qK_C^\top}{\sqrt d}
\right)V_C
$$

It can also be rewritten as:

$$
F_C(q) =
W_{2,C}
\operatorname{softmax}
(W_{1,C}q)
$$

where:

$$
W_{1,C}=\frac{K_C}{\sqrt d}
$$

$$
W_{2,C}=V_C^\top
$$

In this view, the original attention is a dynamic FFN with $N$ hidden units, where each hidden unit corresponds to one token in the long context.

The experiment further constructs a compressed temporary FFN:

$$
\tilde{F}_C(q) =
\tilde{W}_{2,C}
\operatorname{softmax}
(\tilde{W}_{1,C}q)
$$

where:

$$
\tilde{W}_{1,C}\in \mathbb{R}^{M\times d}
$$

$$
\tilde{W}_{2,C}\in \mathbb{R}^{d\times M}
$$

and:

$$
M \ll N
$$

The experiment aims to verify:

$$
\tilde{F}_C(q) \approx F_C(q)
$$

In other words, the goal is to use $M$ temporary memory units to approximate the attention behavior originally represented by $N$ token-level KV units.

## 4. Overall Research Questions

This experiment aims to answer the following questions:

1. **Mathematical approximability**: Can long-context attention outputs be effectively fitted by a smaller temporary softmax-FFN module?
2. **Compression scale**: How does attention reconstruction error change as the number of compressed units $M$ decreases?
3. **Applicability to real models**: Are attention activations inside real Transformer models easier to compress than randomly generated $Q,K,V$?
4. **Layer and head differences**: Do different layers and attention heads show significantly different compressibility?
5. **Potential replaceability**: Can a temporary FFN module partially replace long-context KV cache in downstream tasks while reducing memory cost and preserving task performance?

## 5. Experiment 1: Mathematical Simulation with Random $Q,K,V$

### 5.1 Purpose

This experiment validates the most basic mathematical hypothesis:

> Under randomly generated $Q,K,V$, can a small temporary softmax-FFN be trained to fit the output of full attention?

This experiment does not depend on any real language model. It only tests whether the compression approximation is mathematically plausible.

### 5.2 Inputs

Randomly generate:

$$
K_C\in \mathbb{R}^{N\times d}
$$

$$
V_C\in \mathbb{R}^{N\times d}
$$

$$
Q\in \mathbb{R}^{T\times d}
$$

where:

- $N$: context length;
- $d$: single-head dimension;
- $T$: number of queries.

Recommended initial setting:

$$
N=4096,\quad d=64,\quad T=2048
$$

### 5.3 Teacher Output

Compute the teacher output with full attention:

$$
Y=
\operatorname{softmax}
\left(
\frac{QK_C^\top}{\sqrt d}
\right)V_C
$$

where:

$$
Y\in \mathbb{R}^{T\times d}
$$

### 5.4 Student Model

Construct a temporary softmax-FFN:

$$
\tilde{Y}=
\operatorname{softmax}
(Q\tilde{W}_{1,C}^{\top})
\tilde{W}_{2,C}^{\top}
$$

where:

$$
\tilde{W}_{1,C}\in \mathbb{R}^{M\times d}
$$

$$
\tilde{W}_{2,C}\in \mathbb{R}^{d\times M}
$$

Here, $M$ is the number of compressed memory units, satisfying:

$$
M\ll N
$$

### 5.5 Compression Scales to Compare

Recommended values:

$$
M\in \{32,64,128,256,512,1024\}
$$

The goal is to observe the fitting quality under different values of $M$.

### 5.6 Training Objective

Train the temporary FFN so that its output $\tilde{Y}$ approaches the full-attention output $Y$.

The main loss is:

$$
\mathcal{L}_{mse}
=
\frac{1}{T}
\sum_{t=1}^{T}
\|y_t-\tilde{y}_t\|_2^2
$$

An optional directional constraint can be added:

$$
\mathcal{L}_{cos}
=
1-
\frac{
y_t^\top \tilde{y}_t
}{
\|y_t\|\|\tilde{y}_t\|
}
$$

The total loss is:

$$
\mathcal{L}
=
\mathcal{L}_{mse}
+
\lambda\mathcal{L}_{cos}
$$

Recommended initial setting:

$$
\lambda=0.1
$$

### 5.7 Evaluation Metrics

Record the following metrics:

| Metric | Meaning |
| --- | --- |
| MSE | Output reconstruction error |
| Relative Error | Relative reconstruction error |
| Cosine Similarity | Directional similarity of outputs |
| Compression Ratio | Compression ratio $N/M$ |
| Memory Cost | Comparison between original KV cache size and temporary FFN parameter size |

The relative error is defined as:

$$
\operatorname{RelErr}
=
\frac{
\|Y-\tilde{Y}\|_F
}{
\|Y\|_F
}
$$

### 5.8 Expected Observations

This experiment is expected to show:

1. Reconstruction error decreases as $M$ increases;
2. When $M$ is much smaller than $N$, the student may still preserve non-trivial cosine similarity;
3. Random $Q,K,V$ may be difficult to compress because random vectors lack the structure present in real language models;
4. If a stable compression trend is observed even in the random setting, the direction has basic mathematical feasibility.

## 6. Experiment 2: Fitting Real Transformer Activations

### 6.1 Purpose

This experiment evaluates:

> Whether attention activations inside real large language models can be approximated by a small temporary FFN module.

Unlike the random experiment, $Q,K,V$ from real models come from internal language model representations. They may contain low-rank structure, semantic clustering, and other regularities, making them easier to compress than random data.

### 6.2 Target Models

Start with a small open-source model, such as:

| Model | Notes |
| --- | --- |
| GPT-2 small | Simple architecture and easy to analyze |
| Pythia-160M | Standard open-source architecture, suitable for hooking |
| Qwen2.5-0.5B | Closer to modern LLM architectures |
| LLaMA-style 1B/3B models | Suitable for later expansion |

Recommended initial choice:

> GPT-2 small or Pythia-160M.

### 6.3 Data Selection

Use ordinary long text or constructed documents as input context, for example:

| Data | Purpose |
| --- | --- |
| WikiText-103 | General language modeling text |
| PG-19 | Long-form documents |
| LongBench subset | Long-context task text |
| Constructed long documents | Easier control over information distribution |

Recommended initial data:

> WikiText or constructed long documents.

### 6.4 Activation Extraction

Given a long input text:

$$
C=(x_1,\dots,x_N)
$$

Extract the following from layer $l$ and attention head $h$:

$$
Q_h^l,K_h^l,V_h^l
$$

Select the first $N$ tokens as the context to obtain:

$$
K_C,V_C
$$

Select subsequent tokens or additional query tokens to obtain:

$$
Q_{\text{query}}
$$

The full-attention teacher output is:

$$
Y_{\text{teacher}}
=
\operatorname{softmax}
\left(
\frac{
Q_{\text{query}}K_C^\top
}{
\sqrt d
}
\right)V_C
$$

The temporary FFN student output is:

$$
Y_{\text{student}}
=
\operatorname{softmax}
\left(
Q_{\text{query}}
\tilde{W}_{1,C}^{\top}
\right)
\tilde{W}_{2,C}^{\top}
$$

The training target is:

$$
Y_{\text{student}}
\approx
Y_{\text{teacher}}
$$

### 6.5 Initial Setting

The first version should validate one layer and one head:

| Setting | Initial Value |
| --- | --- |
| Layer | Layer 6 |
| Head | Head 0 |
| Context length $N$ | 1024 or 2048 |
| Number of queries $T$ | 512 |
| Compressed units $M$ | 32, 64, 128, 256 |
| Loss | MSE + cosine loss |
| Metrics | MSE, Relative Error, Cosine Similarity |

### 6.6 Analysis Questions

This experiment should analyze:

1. Whether real-model attention is easier to compress than random $Q,K,V$;
2. The reconstruction-error trend under different $M$ values;
3. Whether some heads are more compressible than others;
4. Whether some heads are difficult to compress because they perform precise retrieval or copying;
5. Whether low, middle, and high layers show different compression difficulty.

### 6.7 Expected Observations

Possible observations include:

1. Real-model activations are easier to fit than random data;
2. Some attention heads achieve high cosine similarity even with small $M$;
3. Retrieval-oriented heads have higher compression error;
4. Semantic aggregation heads have lower compression error;
5. Middle and high layers may be easier to compress than lower layers.

## 7. Experiment 3: Compressibility Across Layers and Heads

### 7.1 Purpose

The first two experiments focus on single-layer and single-head compressibility. This experiment further analyzes:

> Whether long-context attention in different layers and attention heads has different compression difficulty.

### 7.2 Experiment Content

Run the experiment across multiple layers and heads.

For each layer $l$ and head $h$, extract:

$$
Q_h^l,K_h^l,V_h^l
$$

Then train the corresponding temporary FFN:

$$
\tilde{F}_C^{l,h}(q)
=
\tilde{W}_{2,C}^{l,h}
\operatorname{softmax}
(
\tilde{W}_{1,C}^{l,h}q
)
$$

Measure its error relative to the full-attention output.

### 7.3 Results to Record

For each layer/head, record:

| Item | Meaning |
| --- | --- |
| Layer ID | Layer index |
| Head ID | Attention head index |
| $M$ | Number of compressed memory units |
| MSE | Reconstruction error |
| Relative Error | Relative error |
| Cosine Similarity | Directional output similarity |
| Compression Ratio | Compression ratio |

### 7.4 Analysis Questions

This experiment should answer:

1. Which layers are easier to compress with temporary FFNs;
2. Which heads have the lowest compression error;
3. Which heads are most sensitive to compression;
4. Whether compressibility increases with layer depth;
5. Whether a small number of key heads are unsuitable for compression while most heads can be compressed.

### 7.5 Expected Observations

Possible observations include:

1. Compressibility varies significantly across heads;
2. Heads responsible for local syntax, copying, or positional tracking are harder to compress;
3. Heads responsible for semantic aggregation, topic modeling, or global information integration are easier to compress;
4. If most heads can be approximated with small $M$, temporary FFNs have further research value as compressed long-context memory.

## 8. Experiment 4: Context Length and Compression Difficulty

### 8.1 Purpose

This experiment analyzes:

> How the approximation difficulty of temporary FFNs changes as context length $N$ increases.

### 8.2 Setting

Fix the model, layer, head, and number of queries. Vary the context length:

$$
N\in\{512,1024,2048,4096,8192\}
$$

Also evaluate multiple compression scales:

$$
M\in\{32,64,128,256,512\}
$$

For each $(N,M)$ pair, train the temporary FFN and evaluate reconstruction error.

### 8.3 Analysis Questions

This experiment should answer:

1. Whether reconstruction error increases significantly as $N$ grows with fixed $M$;
2. Whether $M$ must increase with $N$ to maintain a similar reconstruction error;
3. Whether there is a relatively stable compression ratio $N/M$;
4. Whether real-model activations contain enough redundancy that $M$ does not need to grow linearly with $N$.

### 8.4 Expected Observations

Possible observations include:

1. Compression difficulty increases as $N$ grows;
2. Real text contexts may contain redundancy, so error growth may not be linear in $N$;
3. If small $M$ can cover large $N$, long-context attention contains compressible structure;
4. If error rises quickly with $N$, the method may be better suited for semantic compression rather than exact token-level memory replacement.

## 9. Experiment 5: Downstream Replacement Validation

### 9.1 Purpose

The previous experiments focus on reconstructing attention outputs. This experiment further validates:

> Whether a temporary FFN module can partially replace the full KV cache in practical long-context tasks.

### 9.2 Compared Settings

Compare three settings:

| Setting | Meaning |
| --- | --- |
| Full-context | Use the full long-context KV cache |
| Short-context | Do not use long context, keeping only a short window |
| Temp-FFN-memory | Use a temporary FFN to approximate long-context attention information |

### 9.3 Task Types

Recommended tasks:

| Task | Purpose |
| --- | --- |
| Long-document QA | Test context information utilization |
| Long-text summarization | Test global semantic compression |
| Fact recall | Test contextual fact memory |
| Needle-in-a-haystack | Test precise retrieval |

The first three tasks are more suitable for this method, while needle retrieval can serve as a stress test.

### 9.4 Metrics to Observe

| Metric | Meaning |
| --- | --- |
| Task Accuracy / F1 | Downstream task performance |
| Perplexity | Language modeling quality |
| Memory Cost | Memory usage |
| Generation Speed | Generation speed |
| Degradation from Full-context | Performance drop relative to full context |
| Improvement over Short-context | Performance gain relative to short context |

### 9.5 Analysis Questions

This experiment should answer:

1. Whether the temporary FFN is better than the short-context baseline;
2. How large the performance gap is between temporary FFN and full-context attention;
3. Whether the method is more suitable for semantic tasks or precise retrieval tasks;
4. Whether there is an acceptable trade-off between memory savings and performance degradation;
5. Whether only some layers or heads can be compressed while a small number of key KV caches are retained.

### 9.6 Expected Observations

Possible observations include:

1. On summarization, topic understanding, and semantic QA tasks, Temp-FFN-memory may significantly outperform short-context;
2. On exact citation and needle retrieval tasks, Temp-FFN-memory may be much weaker than full-context;
3. Partial layer/head replacement may be more stable than full replacement;
4. Temporary FFN is more suitable as a semantic memory supplement for long contexts than as a complete replacement for token-level KV cache.

## 10. Memory and Parameter Analysis

### 10.1 Purpose

This experiment quantifies:

> How much context memory can be saved by using temporary FFN parameter blocks instead of the full KV cache.

### 10.2 Original KV Cache Size

Single layer and single head:

$$
\text{KV}_{origin}=2Nd
$$

Multiple layers and multiple heads:

$$
\text{KV}_{origin}=2NLHd
$$

### 10.3 Temporary FFN Parameter Size

Single layer and single head:

$$
\text{FFN}_{temp}=2Md
$$

Multiple layers and multiple heads:

$$
\text{FFN}_{temp}=2MLHd
$$

### 10.4 Theoretical Compression Ratio

$$
\frac{
\text{KV}_{origin}
}{
\text{FFN}_{temp}
}
=
\frac{N}{M}
$$

For example, when:

$$
N=32768,\quad M=512
$$

the compression ratio is:

$$
\frac{32768}{512}=64
$$

This means the theoretical context memory size can be reduced by approximately $64\times$.

### 10.5 Values to Record

For different combinations of $N,M,L,H,d$, record:

| Variable | Meaning |
| --- | --- |
| $N$ | Context length |
| $M$ | Number of temporary memory units |
| $L$ | Number of model layers |
| $H$ | Number of attention heads |
| $d$ | Head dimension |
| Original KV cache size | $2NLHd$ |
| Temporary FFN parameter size | $2MLHd$ |
| Theoretical compression ratio | $N/M$ |

## 11. Key Criteria for Overall Conclusions

After completing the experiments, focus on the following questions:

1. **Feasibility**: Can temporary FFNs stably approximate full-attention outputs?
2. **Compressibility**: When $M\ll N$, can the reconstruction error remain low?
3. **Structure in real activations**: Are real-model $Q,K,V$ easier to compress than random $Q,K,V$?
4. **Layer and head differences**: Are some layers or heads especially suitable for compression?
5. **Task suitability**: Is the method better suited for semantic aggregation tasks, or can it also support precise retrieval?
6. **Memory benefit**: Can the theoretical compression ratio $N/M$ translate into actual inference-time memory savings?
7. **Future research value**: If some heads or layers can be approximated with small $M$, a hybrid approach can be studied further: keep a small number of key KV caches while compressing the remaining context information into temporary FFNs.

## 12. Final Goal Summary

This experiment does not require fully replacing the long-context mechanism. Instead, it first validates a more fundamental question:

> Under a fixed context, can full attention be viewed as a context-generated dynamic FFN and further compressed into a small temporary parameter module?

If the experiments show that:

$$
\tilde{W}_{2,C}
\operatorname{softmax}
(
\tilde{W}_{1,C}q
)
\approx
\operatorname{softmax}
\left(
\frac{qK_C^\top}{\sqrt d}
\right)V_C
$$

and the error remains low when $M\ll N$, then transferring long-context information from sequence-level KV cache to parameter-level temporary memory is a promising direction for further research.
