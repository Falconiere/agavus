# 11. Neural networks

![Train versus predict](illustrations/learning-and-predicting.webp)

At the top, changing a weight changes the prediction. The numbers use input 1,
bias 0, and learning rate 1. At the bottom, context grows while weights stay fixed.

## Think of learning which clues matter

When learning to sort unfamiliar items, you may first notice individual
clues, then combinations. A network computes intermediate clues from inputs
and adjusts its weights using errors. Below, weights 4 and 4 with bias -6
show exactly how two inputs combine before a prediction.

**Where the analogy stops:** an artificial neuron is a small calculation,
not a biological cell. Backpropagation calculates derivatives; it isn't
evidence that brains learn by this algorithm. More layers don't verify facts.

A neural network is **many tiny calculators connected together**, whose numbers
are found by trial and correction.

## One neuron

A neuron does three things: **multiply, add, squash**.

```mermaid
flowchart LR
  A["threat? 1"] -->|"x 4"| S["add<br/>start at -6"]
  B["partner? 1"] -->|"x 4"| S
  S --> Q["squash into<br/>0 to 1"] --> O["0.88"]
```

Toy task: output 1 when both binary flags are 1. We name them "threat" and
"partner" only to make the arithmetic concrete. This is an AND exercise,
not a definition or assessment of domestic violence; those two flags are
neither necessary nor sufficient for a legal classification.

| Threat | Partner | Add up | Squashed |
| --- | --- | --- | --- |
| 0 | 0 | -6 | 0.002 |
| 1 | 0 | 4 - 6 = -2 | 0.12 |
| 0 | 1 | 4 - 6 = -2 | 0.12 |
| 1 | 1 | 4 + 4 - 6 = 2 | **0.88** |

- The **4s** are *weights*: how much each clue matters.
- The **-6** is the *bias*: how hard it is to convince.
- The **squash** turns any number into something like a chance.

You know this one: **it's logistic regression**. Our router is one layer like
this, with 384 inputs and 3 outputs.

## Why is one not enough?

Use these four short phrases as a simplified yes/no exercise. Was the request refused?

| "não"? | "negou"? | Text | Refused? |
| --- | --- | --- | --- |
| 0 | 0 | concedeu | 0 |
| 1 | 0 | não concedeu | 1 |
| 0 | 1 | negou | 1 |
| 1 | 1 | não negou | **0** |

One sigmoid neuron puts a straight decision boundary between these input
pairs. No straight line separates the two yes corners from the two no
corners. So we add a layer in between:

```mermaid
flowchart LR
  A["não"] --> H1["h1: how many?"]
  B["negou"] --> H1
  A --> H2["h2: beyond one?"]
  B --> H2
  H1 -->|"x 1"| O["refused?"]
  H2 -->|"x -2"| O
```

`h1 = não + negou`, and `h2 = não + negou - 1`. Negative results become 0.
(That small rule is called **ReLU**.)

| Text | h1 | h2 | h1 - 2 × h2 |
| --- | --- | --- | --- |
| concedeu | 0 | 0 | 0 |
| não concedeu | 1 | 0 | 1 |
| negou | 1 | 0 | 1 |
| não negou | 2 | 1 | 2 - 2 = **0** |

We hand-picked a middle-layer clue: "more than one negative". Training can
learn intermediate clues too; it doesn't have to discover this exact one.
The four phrases form a simplified logic exercise, not a general negation parser.

Without the ReLU in the middle, two layers would collapse into one: adding and
multiplying twice is still just adding and multiplying.

**"Deep" learning** simply means many layers.

## How does it find the numbers?

Nobody types in the 4s. It starts with poor numbers and repeats:

```mermaid
flowchart LR
  A["1. Guess"] --> B["2. Measure surprise<br/>the loss"]
  B --> C["3. Which way should<br/>each weight move?"]
  C --> D["4. Move a<br/>small step"] --> A
```

You know step 2: the loss from lesson 4. Step 3 has a name: the **gradient**.
Imagine walking down a hill in fog. You can't see the bottom, but you feel which
way is down, take a step, and feel again.

A neuron with one weight, where the right answer is 1:

| Step | Weight | Guess | Loss |
| --- | --- | --- | --- |
| 0 | 0.00 | 0.50 | 0.69 |
| 1 | 0.50 | 0.62 | 0.47 |
| 2 | 0.88 | 0.71 | 0.35 |
| 3 | 1.17 | 0.76 | 0.27 |

For this example (input 1, fixed bias 0, learning rate 1, cross-entropy),
the weight update is **answer minus guess**. First step:
1 - 0.50 = 0.50, so the weight moves from 0 to 0.50. The better the guess, the
smaller the step.

With many layers, derivatives are calculated from the output back through
earlier operations. That's **backpropagation**. They tell an optimizer how
a small change to each weight would affect loss; the optimizer updates weights.

## Counting against learning

| | Bigram | Neural network |
| --- | --- | --- |
| Learns by | counting | guessing and correcting |
| Looks at | 1 letter | as much as we connect |
| Never-seen case | add 1 | uses similar cases |
| Can we see why? | yes | hardly |
| Risk | too simple | memorizing (lesson 5) |

## Where they are in Agavus

| Part | Neural network? |
| --- | --- |
| Router | one layer (logistic regression) |
| Encoder (lesson 10) | yes, deep; trained by others |
| Bigram | no, only counts |
| Tiny GPT we'll build | yes, deep; trained by us |

## Python: try it

Train the one-weight neuron from the table. Input is fixed at 1, bias at
0, target at 1, and learning rate at 1; those assumptions make the update simple.

```python
from math import exp, log  # Calculate sigmoid probabilities and their loss.

weight = 0.0  # Start with no preference; fixed input 1 makes the score equal this weight.
target = 1.0  # Every practice answer in this tiny example is yes.
rate = 1.0  # Multiply each correction by this learning rate.
for step in range(4):  # Inspect the state before each of four weight updates.
  guess = 1 / (1 + exp(-weight))  # Sigmoid squashes the score into a probability.
  loss = -log(guess)  # With target 1, score the chance assigned to yes.
  print(f"{step}: weight={weight:.2f}, guess={guess:.2f}, loss={loss:.2f}")  # Trace it.
  gradient = guess - target  # Derivative for this input, sigmoid, and cross-entropy.
  weight -= rate * gradient  # Move against the gradient to reduce the loss.
```

The printed weights should be about 0.00, 0.50, 0.88, and 1.17. This example
has no hidden layer or held-out exam; lesson 12 trains the two-layer network.

## Check yourself

1. Weights 4 and 4, bias -6, squash after. Only "threat" is 1. What is the
   number before squashing?
2. Weight 0.50, guess 0.62, right answer 1. What is the new weight?

<details><summary>Answers</summary>

1. 4×1 + 4×0 - 6 = **-2**, squashed to 0.12.
2. 0.50 + (1 - 0.62) = **0.88**.

</details>

Back to the [lesson list](README.md)
