$$
\newcommand{\FF}{\mathbb{F}}
$$

# Lifting plonky3

This document presents a high-level overview of "lifting" as it applies to STARKs, and how these techniques could be incorporated into the plonky3 toolkit. It contains some details and corrections relating to the [zkSummit 13 talk](https://youtu.be/p6Z3WjRcZD0?feature=shared).

<!-- Not really the case anymore, but MMCS can be replaced at the end -->
In particular, we propose an iterative approach that facilitates the transition for projects implementing the current protocol as recursive verifiers. Focusing on individual components of the protocol and introducing them in parallel also provides an opportunity to benchmark them against the existing implementations.

We invite parties involved in the development to give feedback, especially when changes proposed here may interfere with existing STARK deployments.

## What is lifting?

The intuition for lifting is best understood from the perspective of traces, i.e., matrices.
Given a trace $T \in \FF^{d \times w}$ (where $w$ is the width and $d$ is the height as a power of 2), we can view $T = (T_1, \ldots, T_w)$ as a function over a group $H = \langle \omega_d \rangle$ returning the row for every element in $H$

$$
T(\omega_d^i) = (T_1(\omega_d^i), \ldots, T_w(\omega_d^i)) = (T_{i, 1}, \ldots, T_{i, w}).
$$

The degree-$r$ lift of $T$ is a matrix $T^* \in \FF^{r d \times w}$ such that $T^*_i = T_{i \bmod d}$. That is, we can view $T^*$ as the vertical concatenation of $r$ copies of $T$.

The lifted trace is itself a function over a domain $H^* = \langle \omega_{r d} \rangle$.

The projection map $\pi(X) = X^r$ defines the following properties of the transformation
- The original domain $H$ is the projection of $H^*$ by $\pi$:
  $$
  \pi(H) = \langle \pi(\omega_{r d}) \rangle = \langle \omega_{r d}^r \rangle = \langle \omega_{d} \rangle = H
  $$
- For any column $j \in [w]$, the lifted column $T_j^*$ project onto $T_j$, since for any $i \in [r \cdot d]$,
  $$
  T_j^*(\omega_{r d}^i) = T_j(\pi(\omega_{r d}^i)) = T_j(\omega_{r d}^{r i}) = T_j(\omega_{d}^{i}) = T_j(\omega_{d}^{i \bmod d})
  $$

In what follows and to easy notation, we will refer to a single column $T \in \FF^d$.

Given the low-degree extension polynomial  $\hat{T}(X) \in \FF^{<d}[X]$ of $T$, the corresponding polynomial for the lifted column is given by
$$
\hat{T}^*(X) = \hat{T}(\pi(X)) = \hat{T}(X^r) \in \FF^{<rd}[X].
$$

In order to lift the evaluation of the low-degree extension over a coset, the domains over which the traces are committed must be adapted to satisfy the projection map. In particular, if we take the lifted coset $D^* = a \cdot H_{LDE}^*$, the coset for the original domain must satisfy
$$
D = \pi(D^*) = \pi(a \cdot H_{LDE}^*) = \pi(a) \cdot \pi(H_{LDE}^*) = a^r \cdot H_{LDE}.
$$
Currently, plonky3 uses the same offset for all cosets, but changing this shift should not impact the overall protocol.

Crucially, using the above evaluation domains enables efficient lifting of the LDEs themselves. Indeed, for any $x \in D^*$ there exists $y \in D$ such that $y = x^r$, and the evaluation $\hat{T}^*(x)$ maps to $\hat{T}(y)$. In other words, we can derive the list of evaluation of the LDE over the lifted trace from the LDE of the original trace, without incurring any additional costs.

Concretely, since the list of LDE evaluations are stored in memory in bit-reversed order, the evaluation of the lifted LDE at index $i$ is equal to the evaluation of the original LDE at the same index $i \gg \log r$. In other words, we lift the bit-reversed LDE by repeating each row $r$ times, which can be done virtually.

## Overview

Lifting can be applied to different portions of the IOP, namely the Matrix Commitment Scheme, AIR constraints, DEEP query verification, and FRI.

**Matrix Commitment Scheme**:
- **MMCS (Mixed Matrix Commitment Scheme)**: The existing commitment scheme enabling the prover to commit to mixed-height traces in time linear with the total area of the traces.
- **LMCS (Lifted Matrix Commitment Scheme)**: A new construction with
- the same prover performance profile of MMCS
- a simpler/more uniform verification algorithm, expected to reduce the number of constraints required for in a recursive STARK circuit,
- slightly weaker guarantees about the underlying committed data.

**AIR constraints**:
- **Regular AIR**: Any set of constraints that can be implemented in an existing mixed-domain STARK proof. In particular, sets of AIR constraints do not overlap between traces, though are usually connected through lookup arguments.
- **Lifted AIR**: Subset Regular AIR constraints allowing for weaker assumptions in the other portions of the STARK IOP. Most existing constraints already satisfy this property.

**DEEP quotients**:
- **Regular DEEP**: existing approach whereby the prover uses the MMCS to commit to one batched DEEP quotient per trace.
- **Lifted DEEP (UDR)**: new technique restricted to the unique decoding regime, whereby the prover batches the individual trace quotients by lifting-and-accumulating.
- **Lifted DEEP (LDR)**: similar to the above but compatible in the list decoding regime, with the same performance profile. The resulting description is more uniform and allows for more efficient evaluation by the verifier.

**FRI**:
- **Multi-FRI**: Existing implementation of FRI supporting mixed-domain and multi-point opening proofs.
- **Regular FRI**: classic implementation of the FRI protocol allowing for batch low-degree testing of multiple polynomials over the same domain.
- **Lifted FRI**: a variation of FRI described in the paper which makes non-black-box use of FRI to prove low-degreeness of multiple polynomials over mixed domains. We will not describe it here as we do not think it is of practical interest given the alternative constructions detailed above.


Not all of the above protocol variations can be safely combined, but we describe the following configurations which can implemented.
- The LMCS requires the AIR to be liftable, since from the verifier perspective, it is indistinguishable from a commitment to a single trace. Consequentyl, it also is the most uniform construction which should lead to the most verifier optimizations.
<!-- - In the UDR, MMCS can be used with the UDR variant of the DEEP quotient computation. While it seems that it would also allow for a batched AIR constraint quotient, we avoid making that claim at the moment. This approach could be implemented as an intermediary step as it would require minimal modifications to an existing STARK stack. -->
- If the underlying AIR is liftable, then the MMCS and LMCS are interchangeable, along with the optimizations available at the AIR and DEEP layers. The latter also allows the use of regular FRI as it is only concerned with a batch of DEEP quotients over a single domain.
- Adding periodicity constraints at the DEEP level would be a useful intermediary solution which enables the use of lifted AIR optimizations as well as regular FRI. Later, if the set of AIR constraints can be guaranteed to be liftable, the periodicity constraints can be removed, and the DEEP quotient can be optimized to simplify the verifier.


## Lifting Merkle Trees

Plonky3 currently implements the *Mixed Matrix Commitment Scheme*, which constructs a Merkle tree by injecting hashes of rows at different depths of the tree. While not directly obvious, this construction provides an assumption that is crucial when lifting LDEs. It is however unclear whether this assumption is strong enough to argue the soundness of lifting over all rounds of the STARK IOP.

When LDEs are committed using MMCS over the adjusted domains, with largest domain $D^*$, it turns out that for all $x \in \pi^{-1}(y)$ in the preimage of $y \in D$, the opening of the tree at $x$ always opens the same of the smaller trace at $y$. From this, we can deduce that the LDE of the smaller trace is in fact lifted.

<!-- As we will see later, this property ensures that lifting does not affect the soundness of the overall protocol when applied to other parts of the STARK IOP. -->

However, the MMCS verifier algorithm is significantly more complicated than for a single Merkle tree, particularly impacting the implementation and performance of a recursive verifier.

With lifting in mind, we propose the *Lifted Matrix Commitment Scheme* providing a simpler data structure representation and uniform verification algorithm. We describe it by explaining the following diagram:

![image](https://hackmd.io/_uploads/BkR6hZ5bxg.png)

The first thing to notice is the clear separation between the Merkle tree nodes and the leaves representing the digests of the rows of the input matrices. By avoiding the injection of the digests of the smaller digests, authentication paths become simpler to verify.

The leaves themselves are computed by exploiting the repeated nature of the different matrices. We assume these are represented in bit-reversed order to simplify the diagram, which also reflecting the current representation in plonky3.

After computing the leaf digests of the tree, the root can be computed in the usual way.

We sketch the algorithm as follows. We assume there are $m$ traces $T_0, \ldots, T_{m-1}$ with respective sizes $2^0, \ldots, 2^{m-1}$.
- Initialize a vector $S_0$ containing the state of a sponge after absorbing the single row of $T_0$.
- When using a standard overwrite mode sponge, the state corresponds to the capacity portion of the sponge.
- For simplicity and uniformity, it may also be beneficial to assume that each chunk (i.e. the rows of each trace) are padded with zeros to the next multiple of the permutation's width.
- For $\ell = 1, \ldots, m-1$, given the previous state $S_{\ell-1}$.
- Initialize a new sponge state vector $S_{\ell}$ by duplicating each entry in $S_{\ell-1}$.
- For each $i \in [2^\ell]$ update the sponge states by absorbing the $i$-th row of $T_\ell$: $S_{\ell}[i].\mathsf{absorb}(T_{\ell}[i])$
- Return the leaves of the Merkle tree by finalizing the sponge, i.e. $L[i] = S_{m-1}[i].\mathsf{finalize}()$, for all $i \in [2^{m-1}]$.
- Build the Merkle tree over the digests $L$.

The algorithm ensures that the number of absorbs performed by the prover is proportional to the total area of the combined matrices, rather than their lifts.

The concrete implementation does not require using a sponge in a non-black box way. The state can equivalently be represented as a list of digests, with each new absorb corresponding to the compression of the previous digest with the digest of the rows being hashes. However, the opening procedure depends on the widths of each of the underlying matrices.

When opening a set of leaves, it is likely easier to pack each full row in the proof, even when this would repeat certain rows. In a recursive verifier setting, this would not lead to much overhead as the calls to the permutation can be memoized when using logUp.

We note that this construction does not offer the same guarantee as the MMCS, since from the verifier's perspective, this commitment is indistinguishable from one where each smaller trace is defined over the largest domains. Concretely, they cannot infer that the committed LDEs are in fact lifts of LDEs over smaller traces.

Special care must be taken to use this commitment scheme safely, the conditions for which are described in the next section.


<!-- , as it essentially requires the AIR constraints to be insensitive to lifting, though we will explore this in a later section. -->

In the context of plonky3, the implementation of this LMCS is orthogonal to other applications of lifting. We therefore categorize it as low-priority as it would require projects to change their verifier implementation. Moreover, the theoretical performance improvements would have to be tested in practice by benchmarking it against MMCS. However, it could serve a project for first-time contributors interested in getting involved in plonky3 development.

### Implementation notes

- If all the remaining optimizations are implemented, LMCS would only be used to commit to traces, since all other commitments would be single vector Merkle trees.
- The ordering of the traces depends on their relative heights, which must therefore be communicated by the prover.
- The ordering would only be relevant to the verifier when evaluating the constraints at the out of domain challenge, since the ordering of the DEEP quotient polynomials can match the ordering of the traces.
- The implementation of the LMCS could be simplified if
- We assume all heights are powers of 2.
- The code for constructing the leaf digests and the actual Merkle nodes are separate, as this would enable re-use of the Merkle tree for single vector commitments .
- Adding support for verifier-supplied periodic columns/traces would avoid having to commit to "small" traces, thereby statistically limiting the number of duplicated rows in the trace and making memoization less necessary for the verifier.

## Lifting AIRs

In this section, we explore how to adapt the AIR portion of the STARK IOP, both from an implementation and performance perspective.

We take a generic AIR constraint as we would find in plonky3:
$$
S(X) \cdot C\big(T(X), T(\omega_d \cdot X)\big) \bmod V_{H}(X) \equiv 0.
$$

The constraint $C$ is a multi-variate polynomial enforcing a relation between two consecutive rows of the original trace, including the pair consisting of the last and first row (due its definition over a cyclic group). It is multiplied by a selector $S(X)$ which vanishes in a set of points where the constraint should not be enforced. The trace $T$ is valid when the above identity holds, i.e., when the left hand side vanishes over all points in $H$.

Lifting the constraint simply involves mapping any occurrence of $X$ with $\pi(X) = X^r$. The lifted constraint is given by
$$
S(X^r) \cdot C \big( T^*(X), T^*(\omega_{rd} \cdot X) \big) \bmod V_{H^*}(X) \equiv 0.
$$

We note that
- For the current row polynomial we have $T(X^r) = T(\pi(X)) = T^*(X)$, which corresponds to the lifted LDE polynomial
- For the next row polynomial, the shift is now defined with regards to the generator of the lifted trace $\omega_{rd}$, since
  $$
  T\big( \omega_d \cdot X^r \big) = T\big( \omega_{rd}^r \cdot X^r \big) = T\big( (\omega_{rd} \cdot X)^r \big) = T^*(\omega_{rd} \cdot X).
  $$
- The lifted selector $S(X^r)$ repeats the set of disabled points periodically over the lifted domain $H^*$. This property is ensured by virtue of being evaluated directly by the verifier.
- Selectors applying only to the first and final rows will do the same for the first row of all $r$ sub-traces composing the lifted one.
- The transition constraint selector ensures that the constraint never applies to the last row of every subtrace.
- The vanishing polynomial over the original domain maps neatly to the lifted one, since $V_{H}(X^r) = (X^r)^d - 1 = X^{rd} - 1 = V_{H^*}(X)$.

The derivation of a lifted constraint shows that -- in the case where the traces are in fact lifted -- the identity over the lifted trace implies the identity over the original trace.

To prove the polynomial identity, the prover must send the quotient for the lifted constraints
$$
Q^*(X) = \frac{S(X^r) \cdot C \big( T^*(X), T^*(\omega_{rd} \cdot X) \big)}{V_{H^*}(X)}.
$$

Naively evaluating this quotient over the lifted LDE domain would require work proportional to the size largest domain, drastically affecting prover performance. Instead, observe that $Q^*(X)$ is simply the lift of the original quotient polynomial
$$
Q(X) = \frac{S(X) \cdot C \big( T(X), T(\omega_{d} \cdot X) \big)}{V_{H}(X)}.
$$
This ensures the computation cost remains the same.

Another benefit of lifting AIRs is that the quotients for individual traces can all be lifted to the same target domain. Therefore, they can now all be batched using verifier randomness. We could either use additional powers of $\alpha$ which are already used to batch constraints for the same trace (leading to slightly worse soundness), or sample a fresh challenge $\beta$ to combine the already batched trace quotients into a single one.

This should definitely lead to better prover performance, as they now only need to commit to a single quotient polynomial over the lifted LDE domain, rather than for each trace. Moreover, this also simplifies the opening of this polynomial for the verifier, as it now has to deal with a uniform Merkle tree (regardless of whether LMCS or MMCS is used).

From the perspective of plonky3, lifting AIRs is not currently relevant to plonky3 since it only implements the `uni-stark` prover which only supports a single trace. However, we expect most plonky3-based to adopt this technique.

### Implementation notes

Assume there are $m$ traces $\{T_\ell\}_{\ell=0}^{m-1}$, each of width $w$, and heights $d_\ell$. For ease of notation, we assume each trace is constrained by a list of $a$ constraints $\{C_{\ell,j}\}_{j=0}^{a-1}$, such that
$$
C_{\ell,j}(T_\ell(X)) \equiv 0 \bmod V_{H_\ell}(X).
$$

Since the traces are committed by ascending heights $d_\ell$, the prover sends the permutation $\sigma: [m] \rightarrow [m]$ mapping the original trace order to the committed one.

Given two verifier challenges $\alpha, \beta$, the prover computes for each
$$
R_\ell(X) = \sum_{j=0}^{a-1} \alpha^{j} \cdot C_{\ell,j}(T_\ell(X)),
$$
as the evaluations over the coset $D_{\ell}$.
Note that the challenge $\beta$ can also be taken as $\alpha$ raised to the power of $a$ (or the maximum of the number of constraints in each trace).

It then virtually lifts each numerator to the lifting domain $D^*$ as $R_{\ell}^*(X)$.

The batched and ordered numerator is given by
$$
R^*(X) = \sum_{\ell=0}^{m-1} \beta^{\sigma(\ell)} \cdot R^*_{\sigma(\ell)}(X).
$$

Finally, the prover divides by the lifted domain quotient
$$
Q^*(X) = \frac{R^*(X)}{V_{H^*}(X)},
$$
and splits it into chunks whose degree match the largest domain of the traces.

After committing to the $Q^*$, the verifier sends the out-of-domain point $z$, to which the prover responds with trace evaluations $t_{\ell} = T_{\ell}(\pi_\ell(z)) = T^*_\ell(z)$. Note that these evaluations can be computed efficiently over the original domains by the following
- Start by deriving the barycentric weights $\{L^*_{i}(z)\}_i$ for the lifted/largest domain.
- The weights for the domain $D_\ell$ whose lifting map $\pi_\ell$ has degree $r_\ell$ are given by
  $$
  L_{\ell,i}(z^{r_\ell}) = \sum_{k=0}^{r_\ell-1} L^*_{i + k \cdot d_\ell}(z),
  $$
  since the lifting of a Lagrange polynomial repeats the set of points where it equals one, and therefore is the sum of all Lagrange polynomials over the lifting domain at distance $d_\ell$ of each other.
- Concretely, this means the weights for the smaller traces can be derived iteratively using only summation.

It seems simpler for the prover to send the list of evaluations $t = [t_{\sigma(\ell)}]$ following the same same ordering $\sigma$, since the verifier will only need to care about the ordering when they evaluating the batched numerator $R^*(z)$. In particular, it can simply reorder the powers of $\beta$ as $[\beta^{\sigma(\ell)}]_{\ell=0}^{m-1}$. Similarly, the offset of $t_\ell$ in $t$ is given by $\sum_{\nu=0}^{\ell-1} w_{\sigma(\nu)}$, where $w_\ell$ is the width of the $\ell$-th trace in the pre-defined order. Moreover, since each $R_\ell$ will actually be composed of selectors which must be evaluated at $\pi_\ell(z) = z^{r_\ell}$, these will also have to take into account the ordering $\sigma$.

### Liftable AIRs

Intuitively, lifting an AIR constraint can be interpreted as simply applying the same constraint to a trace over the lifted domain. The only difference is that transition constraints only wrap around at the lifted trace's boundary, rather than wrapping around each individual sub-trace. Concretely, a constraint originally applied to rows $(T_{d-1}, T_{0})$ would now apply to all pairs $(T^*_{kd-1}, T^*_{kd})$ for all $k = 1, \ldots, r-1$ and $(T^*_{rd-1}, T^*_{0})$.

This is problematic when using the LMCS, as we can no longer assume that the traces are in fact lifed, and therefore must deal with the possibility that the first/rows of each sub-trace may differ.

Due to the simpler verifier implementation for LMCS, we expect most projects to make use of it. To do so safely, they must ensure their set of constraints are *liftable*. We define it as a restriction on the set of AIR constraints that can be applied together over a single trace.

Concretely, when taken together, the constraints should be insensitive to the prover maliciously committing to a trace it claims to be lifted but which in fact contains non-identical sub-traces. In practice, certain projects do rely on the wrap-around property, so these must be carefull analyzed and fixed in order safely lift them when using the LMCS.


#### Periodicity constraints

We start by presenting a safe solution that ensures that any existing set of constraints can be lifted. It involves an additional virtual AIR constraint which explicitly proves that all sub-traces are equivalent. It is given by over the lifted trace

$$
T^*(X) - T^*(\omega_{r} \cdot X) \equiv 0 \bmod V_{H^*}(X).
$$


Here, the shift by $\omega_{r}$ is equivalent to shifting the row by $d$ points since $\omega_{r} = \omega_{rd}^d$. It proves that every all rows repeat with period $d$, thereby ensuring periodicity of the entire trace.
By adding this constraint to each trace's set of AIR constraints, it automatically becomes liftable.


We refer to it as a "virtual" constraint since it is linear. In fact it is actually enforced at the DEEP portion of the STARK, since the prover must now prove that the evaluations $T^*(z)$ and $T^*(\omega_r\cdot z)$ are equal for a random out-of-domain point $z$. In addition to the opening of $T^*(\omega_{rd} \cdot z)$, the prover must instead show that the two following quotients are low-degree
$$
\frac{T^*(X) - T^*(z)}{(X-z)(X-\omega_r \cdot z)}, \quad \frac{T^*(X) - T^*(\omega_{rd} \cdot  z)}{X-\omega_{rd} \cdot z}.
$$

Unfortunately, this limits the batching opportunities when combining the lifted DEEP quotients from the mixed-size traces. In particular, this is due to the fact that the additional divisor $(X-\omega_r \cdot z)$ depends on the lifting degree $r$ which is different for every domain size.

In practice though, the current implementation of the DEEP quotients within the `two-adic-pcs` is sufficiently general to allow this extra opening to be verified.

This solution while sound does not seem likely to be implemented as it complicates the computation of the DEEP polynomial which is already one of the bottle necks for recursive verifiers.

However, we will explore this in more detail in the next section.


### Constraint analysis

Focusing on plonky3, we can define clear conditions which ensure that a set of constraints can be safely lifted, and if not, detect which ones may need special attention.

Recall the usual selectors
$$
\begin{aligned}
S_{0}(X) &= \frac{X^n-1}{X-1}, \\
S_{-1}(X) &= \frac{X^n-1}{X-h^{-1}}, \\
S_{*}(X) &= X-h^{-1},
\end{aligned}
$$
which respectively select either the first, last, or all but the last rows.

For a general constraint $C\big(T(X), T(\omega \cdot X)\big)$, we can verify that a constraint is liftable if it has one of the following shapes

$$
\begin{aligned}
S_{0}(X) \cdot& C\big(T(X), T(\omega \cdot X)\big), \\
S_{-1}(X) \cdot& C\big(T(X)\big), \\
S_{*}(X) \cdot& C\big(T(X), T(\omega \cdot X)\big), \\
&C\big(T(X)\big).
\end{aligned}
$$
This ensures that no constraint which can be active in the last row references the "next" row which wraps around to the first row.

Given the implementation of the `SymbolicAirBuilder`, it would be easy to detect if a constraint is one of the above.

However, there are situations where a constraint which is not in the above format, because it interacts with other constraints which make the entire AIR safe to lift.

For example, a boundary constraint in the first row can ensure a value is constant. This would ensure that even if the prover committed to different sub-traces, that accessing the value in the first row of the next trace would always refer to the same value.

This can be used to fix running sum/product constraints which are used for implementing logUp/permutation arguments. Focusing on the former, if we are verifying that the sum of all values of a column $f \in \FF^n$ equals $s$, the prover sends an auxiliary column $g \in \FF^n$ such that
$$
g_{i+1} = g_{i} + f_i - \frac{s}{n},
$$
where $g_0$ can be arbitrary.

This is because in the last row, we would have
$$
g_{n} = g_0 = g_0 + \sum_{i=0}^{n-1} f_i - s \implies \sum_{i=0}^{n-1} f_i = s.
$$
When lifted, the constraint at the boundary of the first trace would just prove that
$$
g_n = g_0 + \sum_{i=0}^{n-1} f_i - s,
$$
which does not guarantee the same sum since $g_n$ is not necessarily equal to $g_0$.

We can however enforce the additional constraint
$$
S_0(X) \cdot g(X) = 0
$$
which guarantees that $g_{n} = g_0 = 0$, thereby ensuring that the running sum "resets" at each boundary.

Note that in the case where the prover maliciously committed to some $f$ which did not repeat as expected, we would still guarantee that each slice of $f$ of length $n$ sums to $s$. In the case of logUp, this would allow the prover to commit to different permutations of the rows of the original trace, though this does not affect the soundness of the lookup argument.

Finally, we note that similar analyses can be performed when constraints may not directly be liftable. One other example would be the constraints for logUp with GKR where the prover commits to the $eq$ polynomial. Given that the prover must already open that polynomial at multiple shifts, adding periodicity constraints would not significant overhead.

## Lifting DEEP Quotients

In this section, we describe how to implement lifting at the DEEP quotient computation layer. We will focus on the case where the AIR is liftable, without the periodicity constraints which would involve additional openings. If that solution is used, then we would recommend using the existing implementation which allows arbitrary opening points for each trace.

For simplicity, we'll restrict ourselves to the case where each trace $T_\ell$ has $w$ columns, and with evaluations (as vectors of $w$ elements):
- $t_\ell = T_\ell(\pi_\ell(z)) = T^*_\ell(z)$
- $t'_\ell = T_\ell(\omega_{H_\ell} \cdot \pi_\ell(z)) = T^*_\ell(w_{H^*}\cdot z)$.

Here the order of the traces corresponds to the commitment order, i.e. ordered by increasing heights.

Here, it may be beneficial to use two verifier challenges $\alpha, \beta$ used for the random linear combinations of the trace columns and for combining evaluations.


The prover starts by computing the random linear combination of all lifted columns
$$
R(X) = \sum_{\ell=0}^{m-1} \alpha^{\ell \cdot m} \sum_{j = 0}^{w-1} T^*_{\ell, j}(X),
$$
where the above can be optimized by accumulating-then-lifting. That is the trace accumulate $T_{\ell}$ into a polynomial of length $|D_{\ell}|$, then lift the running $R(X)$ to the size of the next domain $|D_{\ell+1}|$.

The same is done for both lists of evaluations
- $\tilde{t} = \sum_{\ell=0}^{m-1} \alpha^{\ell \cdot m} \sum_{j = 0}^{w-1} t_{\ell, j}$
- $\tilde{t}' = \sum_{\ell=0}^{m-1} \alpha^{\ell \cdot m} \sum_{j = 0}^{w-1} t'_{\ell, j}$.

Using the challenge $\beta$, the prover then computes the full quotient
$$
Q(X) = \frac{R(X)-\tilde{t} }{X-z} + \beta\frac{R(X)-\tilde{t}'}{X-\omega_{H^*} \cdot z}
$$

For the verifier, this is likely more optimal since when it queries the rows of the traces at various points $x \in D^*$, it can focus on simultaneously computing $R(x)$ and hashing the rows without having to interface with the VM memory.

If sampling a second challenge is expensive, we can just sample $\beta$ and take $\alpha = \beta^2$

This above derivation is actually exactly the same that one would obtain if the STARK was composed of a single trace since it does not actually need to know the individual trace widths.

Moreover, this also where the zero-padding for each trace can be beneficial, since that would be equivalent to the prover having added zero-values columns whose evaluation at the out-of-domain points would also equal zero.


## Low degree testing

When using the above computation of the DEEP polynomial, the only thing that remains is performing a low-degree test on the batched quotient.

### FRI

When using FRI, we can focus purely on the "batched FRI" for testing a set of codewords over a common domain.

It is currently missing from plonky3 which only implements Multi-FRI. However, an implementation of regular FRI could easily be derived from it. The resulting simplicity would also make it easier/efficient to implement optimizations such as a variable folding step sizes as well as custom stopping points.

Rather than replace multi-FRI, regular FRI could be implemented as an additional PCS. The interface for the latter could however be simplified as it could accept a single list of points at which all lifted traces would be opened.

### STIR/WHIR

Both of these new protocols would likely benefit a lot from lifting, as they were designed to work with polynomials over the same domain. For configurations based on lifted AIRs, these would be a drop-in replacement for regular FRI.


## Conclusion

While this document outlines many different avenues towards implementing lifting techniques in plonky3, we strongly believe that the best solutions is **liftable AIRs**. It provides the easiest path towards uniformity and likely lead to the simplest overall implementation.

In summary, the sub-projects to get to that end goal are
- Implementing LMCS
- Define an interface to allow for different DEEP quotient representations, to be reused across different PCSes/low-degree tests (regular FRI, STIR, WHIR)
- Implementing regular FRI purely as a low-degree test, with an interface compatible with STIR/WHIR
- Benchmarking individual components against their non-lifted counter-parts.
