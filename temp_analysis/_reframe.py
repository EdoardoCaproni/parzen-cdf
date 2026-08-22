"""Toglie dal report l'inquadramento come 'delta rispetto alle versioni precedenti'.

Principio: le alternative si presentano come scelte progettuali che un lettore
considererebbe da se', non come nostre versioni passate. Resta legittimo dire "abbiamo
provato X e non funziona": e' normale scrittura scientifica sul proprio lavoro. Sparisce
invece l'archeologia delle passate, che presuppone un lettore che le conosca.
"""

import io

P = "report/report3.tex"
s = io.open(P, encoding="utf-8").read()
n = 0


def sub(old, new):
    global s, n
    assert s.count(old) == 1, f"ancora non univoca ({s.count(old)}): {old[:70]!r}"
    s = s.replace(old, new)
    n += 1


# ------------------------------------------------------------------ abstract
sub("""Three choices distinguish this version of the work from its predecessors, and each is
argued rather than asserted. First, the window width is chosen by cross-validation rather
than by a rule of thumb: we show that a rule of the form $h_1 = c\\,\\shat$ cannot exist,
because the optimal constant varies by more than a factor of ten across distributions, and
that recalibrating it on a larger benchmark makes matters worse rather than better. Second,
the network is an architecture whose output is a valid distribution function for every
value of its parameters, so that monotonicity, the limits at infinity, non-negativity of
the density and unit mass are guaranteed by construction instead of being repaired
afterwards on a grid. Third, since on the data the method is meant for the true
distribution is unknown, quality is reported through quantities computable from the sample
alone; we measured which of those quantities are informative, and found that the most
natural one is actively misleading.

We also state a limitation openly: a mixture of logistic distribution functions has
strictly positive density on the whole line, and therefore rounds off sharp edges. On
distributions with compact support the error grows by a factor between three and fourteen.
We checked whether more capacity repairs this, and it does not come for free.""",
"""Three features of the procedure are not the obvious choices, and the report argues for each
rather than asserting it. The window width is chosen by cross-validation instead of a rule of
thumb, because a rule of the form $h_1 = c\\,\\shat$ cannot be made to work: on a standard
benchmark the optimal constant ranges over a factor of twelve between distributions, and
calibrating it more carefully makes the rule worse. The network is written so that its output
is a valid distribution function at every value of its parameters, which puts monotonicity,
the limits at infinity, the positivity of the density and its unit mass into the model rather
than into a repair step applied to a grid afterwards. And because the distribution is unknown
on the data the method is built for, quality is reported through quantities computable from
the sample alone; we measured which of those are informative, and the one a reader would most
expect to see turns out to be anticorrelated with the error it appears to certify.

On five multimodal targets at $n = 500$ the procedure improves on a conventional neural design
and on the Parzen estimator it learns from. One limitation is stated rather than left implicit:
a mixture of logistic distribution functions has strictly positive density on the whole line,
so distributions with sharp edges are rounded off and the error grows by a factor between
three and fourteen. Adding capacity halves that error and costs a third of the accuracy on the
multimodal targets, so we do not take it.""")

# ------------------------------------------------------------------ via la storia delle passate
inizio = s.index("\\subsection{Where this work comes from}")
fine = s.index("% =====================================================================================\n"
               "\\section{Development data")
s = s[:inizio] + s[fine:]
n += 1

# ------------------------------------------------------------------ capitolo 3
sub("""The choice of kernel carries a subtlety that caused a real error in the first pass of this
project. Kernels differ in how much they smooth at a given $h$, because they have different
spreads:""",
    """The choice of kernel carries a subtlety that is easy to miss, and we did miss it at first.
Kernels differ in how much they smooth at a given $h$, because they have different spreads:""")

# ------------------------------------------------------------------ capitolo 4
sub("""The second pass of the project settled on $h_1 = 1.5\\,\\shat$, with $\\shat$ the sample
standard deviation. The constant was obtained by sweeping $h_1$ over three orders of
magnitude on a ladder of mixtures and picking a value that worked well across them. The
result looked convincing, and on the distributions it was tested against it was.""",
    """The natural way to fix $h_1$ is to tie it to a scale estimated from the data, and the natural
scale is the sample standard deviation. Sweeping $h_1$ over three orders of magnitude on a
ladder of Gaussian mixtures and picking a value that works across them gives
$h_1 = 1.5\\,\\shat$, which is what we used for some time. The result looked convincing, and on
the distributions it was tested against it was.""")
sub("everything measured on it. To these we add the three Gaussian mixtures used in the earlier\npasses of the project, so that the new results can be compared directly against the old\nbenchmark. Sixteen densities in total",
    "everything measured on it. To these we add three Gaussian mixtures of our own, of the kind\nthe assignment describes, so that the benchmark also covers the case of direct interest.\nSixteen densities in total")
sub("constant start & $h = 1/\\sqrt{n}$, the untuned starting point of the earlier study \\\\",
    "constant start & $h = 1/\\sqrt{n}$, an untuned reference point \\\\")
sub("""Before comparing selectors it is worth asking whether the rule the earlier pass was looking
for exists at all.""",
    """Before comparing selectors it is worth asking whether the rule we were looking for exists at
all.""")
sub("""with the rule it replaces. What is not shared is the ratio $\\hhat_1/\\shat$, which the earlier
rule fixed at $1.5$ and which in fact ranges from $0.32$ to $3.77$ across the sixteen""",
    """with the rule it replaces. What is not shared is the ratio $\\hhat_1/\\shat$, which a fixed rule
holds constant and which in fact ranges from $0.32$ to $3.77$ across the sixteen""")

# ------------------------------------------------------------------ capitolo 5
sub("""The architecture inherited from the earlier passes is a small multilayer perceptron with a
sigmoid on the output,""",
    """The first architecture one would reach for, and the one we started from, is a small
multilayer perceptron with a sigmoid on the output,""")
sub("""in the earlier code was built from the true distribution, that last step used information
which does not exist when the method is applied to real data.""",
    """in a study of this kind is naturally built from the known distribution, that last step uses
information which does not exist when the method is applied to real data.""")
sub("The effect is measurable. Training the inherited architecture on the reference trimodal",
    "The effect is measurable. Training architecture~\\eqref{eq:mlp} on the reference trimodal")
sub("""unchanged. The network does not get it for free. In the inherited architecture the weights
are initialised without reference to the data and the biases start at zero, so every sigmoid
is centred at the origin regardless of where the sample lies, and the optimiser works in
absolute units.""",
    """unchanged. A network does not get it for free. In architecture~\\eqref{eq:mlp} the weights are
initialised without reference to the data and the biases start at zero, so every sigmoid is
centred at the origin regardless of where the sample lies, and the optimiser works in absolute
units.""")
sub("""The consequence is severe and easy to miss, because the reference mixtures used throughout
the project all sit near the origin. Translating the same trimodal sample by $100$ and
retraining, the inherited pipeline returns a Kolmogorov--Smirnov distance of $0.50$ on two
seeds out of three and $0.70$ on the third, against $0.03$ untranslated.""",
    """The consequence is severe and easy to miss, because the mixtures one tests on conventionally
all sit near the origin. Translating the same trimodal sample by $100$ and retraining, that
architecture returns a Kolmogorov--Smirnov distance of $0.50$ on two samples out of three and
$0.70$ on the third, against $0.03$ untranslated.""")
sub("""$1$; the inherited architecture obtained the first two after the fact on a grid and could not
obtain the third at all, because a finite sum of bounded terms cannot reach the ends of""",
    """$1$; a multilayer perceptron with a sigmoid output obtains the first two after the fact on a
grid and cannot obtain the third at all, because a finite sum of bounded terms cannot reach
the ends of""")
sub("""input makes the estimate equivariant, which the inherited version was not: translating the
data by $100$ was enough to make it collapse.""",
    """input makes the estimate equivariant, which that design is not: translating the data by $100$
is enough to make it collapse.""")

# ------------------------------------------------------------------ capitolo 6
sub("""Three mechanisms present in the earlier pipeline have no object under the architecture of
Chapter~\\ref{sec:network}, and removing them is not a simplification for its own sake.""",
    """A design built on architecture~\\eqref{eq:mlp} needs three further mechanisms to deliver a
valid estimate. None of them has any object under the architecture of
Chapter~\\ref{sec:network}, and dropping them is not a simplification for its own sake.""")
sub("""The \\textbf{rectification} step, the running maximum followed by rescaling, was doing real
work in the old pipeline, which is worth acknowledging: the unconstrained network does
produce non-monotone curves. Retraining the inherited architecture with its own recipe, about
one initialisation in four yields a curve that decreases somewhere, in the worst case over
$6\\%$ of the grid points.""",
    """The \\textbf{rectification} step, the running maximum followed by rescaling, does real work,
which is worth acknowledging: an unconstrained network genuinely produces non-monotone curves.
Training architecture~\\eqref{eq:mlp} on these labels, about one initialisation in four yields
a curve that decreases somewhere, in the worst case over $6\\%$ of the grid points.""")

# ------------------------------------------------------------------ capitolo 7
sub("""It is worth being concrete about what this replaces, since the earlier arrangement also
produced a plausible-looking density.

There, the density was obtained by evaluating the repaired curve at the nodes of a grid and""",
    """The alternative, which also produces a plausible-looking density, deserves a concrete
comparison.

There, the density is obtained by evaluating the repaired curve at the nodes of a grid and""")
sub("""None of this was visible in the reported figures, because all of them were computed on the
same grid that produced the estimate.""",
    """None of it is visible in a plot, because the plot is drawn on the same grid that produced the
estimate.""")

# ------------------------------------------------------------------ capitolo 8
sub("""The earlier pipeline took it from the truth, as the interval spanned by the component means
plus or minus five component standard deviations, and in a few scripts it was hard-coded
outright. On the data the method is meant for there are no component means, so the interval
has to be built from the sample.""",
    """In a study where the distribution is known there is an obvious answer, which is to take the
interval spanned by the component means plus or minus a few component standard deviations,
and that is what we did while the targets were all synthetic. On the data the method is meant
for there are no component means, so the interval has to be built from the sample.""")
sub("""substantive difference from the earlier pipeline. A value of $0.9998$ says something true""",
    """substantive difference from a pipeline that rescales. A value of $0.9998$ says something true""")

io.open(P, "w", encoding="utf-8").write(s)
print(f"{n} interventi nella prima parte")
