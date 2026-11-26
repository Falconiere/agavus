# Learning illustrations

Each of the 17 lessons has its own illustration; the learning index has a
separate overview. All were generated with the built-in `image_gen` tool,
using synthetic examples and authored phrases from the lessons. No case
records or personal data were supplied.

The lesson text and worked examples are the reference for the numbers.
The embedding map and generalization curves are illustrative, not measured
projections or experiment results. The training checkpoints reproduce the
lesson's recorded example; they aren't a new run. The sorting trays show
possible categories, not a legal assessment. Attention uses the lesson's
unscaled toy example with rounded shares and output.

The generated PNG originals are retained in the tool's local output folder.
Repository assets are WebP copies at the same dimensions, encoded with the
installed `cwebp -q 90 -m 6 -metadata none`; no dependency was added.

## One image per lesson

| Lesson | Illustration |
| --- | --- |
| Index | [Three different jobs](three-jobs.webp) |
| 1 | [next-letter](next-letter.webp) |
| 2 | [token-addresses](token-addresses.webp) |
| 3 | [tokens-and-counts](tokens-and-counts.webp) |
| 4 | [loss-surprise](loss-surprise.webp) |
| 5 | [practice-and-exam](practice-and-exam.webp) |
| 6 | [sampling-loop](sampling-loop.webp) |
| 7 | [scorecard](scorecard.webp) |
| 8 | [formula-tools](formula-tools.webp) |
| 9 | [memory-and-sources](memory-and-sources.webp) |
| 10 | [embedding-map](embedding-map.webp) |
| 11 | [learning-and-predicting](learning-and-predicting.webp) |
| 12 | [training-steps](training-steps.webp) |
| 13 | [shared-weights](shared-weights.webp) |
| 14 | [word-pieces](word-pieces.webp) |
| 15 | [attention-mixture](attention-mixture.webp) |
| 16 | [transformer-stack](transformer-stack.webp) |
| 17 | [data-and-time](data-and-time.webp) |

Each lesson embeds its image with descriptive alternative text and a caption
connecting it to the example. Open an image above to view it at full size.

## Prompt set

Each image used the common style followed by its specific instructions below.
Text is wrapped for reading. The attention image received the targeted
arrow-direction correction recorded at the end.

### Common style

Use case: scientific-educational. Create a finished educational editorial illustration for a
beginner learning classifiers and language models. Landscape 3:2 composition, generous margins,
warm off-white paper, crisp dark navy ink outlines, flat muted teal, coral and golden-yellow
accents, a little colored-pencil texture. Visually explain with familiar physical objects and
clear arrows, not generic boxes alone. Large legible English labels only as specified; no
paragraphs, no tiny text, no invented labels, no watermark. No brains, robots, biological
anatomy or claims about consciousness. Keep a coherent restrained textbook illustration style.

### three-jobs.webp

Three equally sized side-by-side illustrated panels. LEFT heading exactly 'CONTINUE': a desk
with a paper strip reading 'bom', an arrow to three candidate word cards 'dia', 'trabalho',
'fim'; one 'dia' card being placed next on the strip. CENTER heading exactly 'CLASSIFY': a paper
document with a magnifying glass passes to three physical sorting trays labeled 'Family',
'Domestic violence', 'Other'; document contains abstract horizontal ink lines only, no case
information. RIGHT heading exactly 'VERIFY': an open reference book, two cited-premise index
cards connected by a thread to a conclusion card and a checkmark; no readable book prose. A
small footer under each panel respectively 'Next token', 'Area scores', 'Supported conclusion'.
Distinguish these as different tasks. These are symbolic teaching scenes, not an image of brain
functions.

### next-letter.webp

A simple two-panel story titled 'ONE LETTER OF CONTEXT'. LEFT: a long paper strip spells 'g u a
r d' on separate letter tiles. A sliding cardboard window covers all the earlier tiles g,u,a,r
and exposes ONLY the final 'd'. Clearly show earlier letters physically covered behind a flap,
not available to the model. A small mechanical predictor reads that one exposed tile. RIGHT: the
predictor offers three continuation cards labeled 'a', 'r', 'u'; a hand selects the 'a' card.
Short labels exactly 'Sees only d' and 'Guesses the next letter'. Footer exactly 'The earlier
letters are hidden'. No numerical chances, no weights, no claims that a is certain, no
biological images. Make the window metaphor immediately understandable.

### token-addresses.webp

A top-down illustrated desk teaching tokenization and ID lookup. TOP: paper word strip 'guarda'
is separated into six individual letter tiles exactly g u a r d a. BELOW: six arrows connect
those tiles to six numbered drawer labels in the same order exactly 5 7 3 6 4 3. Draw a small
address-book tab beside the drawers labeled 'Vocabulary'. The identical a tiles must both
connect to ID 3. Heading exactly 'LETTERS GET ADDRESSES'. Footer exactly 'IDs name tokens; they
do not rank them'. No invented extra tokens, no embedding vectors or probabilities.

### tokens-and-counts.webp

A clear left-to-right, three-stage illustrated storyboard. LEFT heading 'PIECES': six individual
cream letter tiles aligned on a desk, exactly g u a r d a, separated enough to show each letter.
MIDDLE heading 'COUNT': show a large tile labeled a with three branches pointing to tiles r, a
visible space symbol ␣, and a tile labeled END. On the branches display exact counts '2', '1',
'1' respectively, using small colored tally chips. RIGHT heading 'OBSERVED SHARES': three
horizontal colored bars, r length half, space symbol ␣ length quarter, END length quarter.
Labels exactly 'r 50%', 'space 25%', 'end 25%'. Bottom text exactly 'guarda guarda'. All shares
describe raw observed pair counts before smoothing, not model predictions. Do not add other
numbers or letters.

### loss-surprise.webp

Three side-by-side illustrated prediction scorecards, all for the same observed letter 'a'.
Heading 'HOW SURPRISING WAS THE ANSWER?'. LEFT: prediction card 'a: 90%' beside a small stack of
score chips labeled 'Loss 0.11'. CENTER: card 'a: 50%' beside a medium stack labeled 'Loss
0.69'. RIGHT: card 'a: 10%' beside a visibly larger stack labeled 'Loss 2.30'. A single observed
answer tile labeled 'Actual: a' centered above all three with thin lines to each. Bottom caption
exactly 'Less chance on the actual answer → higher loss'. These are numerical error scores; do
not show distressed human faces or imply that the model feels emotion. Short labels, no other
numbers.

### practice-and-exam.webp

Educational illustration in two horizontal bands. TOP: three desks clearly separated with space
between them. First desk has a stack of practice flashcards and pencil, heading 'TRAIN'. Second
desk has a practice quiz sheet and two adjustable dial choices, heading 'VALIDATE', footer
'Choose settings'. Third desk has a closed envelope being opened over an unseen exam sheet,
heading 'TEST', footer 'Final check'. A bold divider between train/validate and test suggests
separation; there are NO arrows carrying answer cards into the test desk. BOTTOM: two small
clean line charts. Left chart heading 'GENERALIZES': horizontal axis 'Training steps', vertical
axis 'Loss'; two curves fall together and stay close, teal labeled 'Train', coral labeled
'Unseen'. Right chart heading 'OVERFITS': same axes; teal Train curve keeps falling, coral
Unseen curve falls initially then rises. The words Train and Unseen are colored to match lines
as well as explicit text, never rely only on color. No numerical tick labels, these are
illustrative curves not measured results. No percentages or promises.

### sampling-loop.webp

A clean illustrated circular paper-and-wheel mechanism titled 'WRITE ONE TOKEN, THEN REPEAT'. At
TOP a current token card exactly 'a' feeds a spinner wheel. Wheel has exactly three slices: teal
half labeled 'r 50%', coral quarter labeled 'space 25%', gold quarter labeled 'end 25%'. To the
RIGHT a picked card exactly 'r' is shown as one possible spin result, then added to the end of a
growing paper strip. A curved arrow returns the new current token r to the predictor input at
LEFT, never to weights. A separate small END card leads to a stop sign at bottom. Footer exactly
'Illustrative unsmoothed a row'. The 50/25/25 wheel is for the observed a row only; do not label
it as the smoothed model or imply the next r row uses these same shares. No claims of truth or
intention.

### scorecard.webp

An illustrated notebook spread titled 'FROM CHANCES TO ONE SCORE'. LEFT page shows three
successive prediction tickets with probabilities exactly '1/2', '1/2', '1/4'. Under them the
product exactly '1/16'. RIGHT page shows the same tickets converted into separate surprise chips
exactly '0.69', '0.69', '1.39', then those chips collected into one tray labeled 'Average loss:
0.92'. A bridge between pages is labeled '−ln'. Footer exactly 'An illustrative three-step
example'. The exact unrounded average is 0.9242 and rounds to 0.92; the displayed chips are
rounded. Use paper tickets, physical chips, a calculator, and arrows to teach product versus sum
then average. No other arithmetic or random labels.

### formula-tools.webp

Three large illustrated tools laid out left-to-right on a desk, title 'THREE PROBLEMS, THREE
TOOLS'. LEFT: tally ledger and abacus, label 'Count what follows'. CENTER: an empty drawer slot
gets one spare chip, label 'Allow unseen pairs'. RIGHT: a balance scale weighs stacks of
surprise chips, label 'Score the prediction'. Bottom small labels under the panels respectively
'Count and divide', 'Smoothing', 'Logarithmic loss'. Above each panel a small simple question
card respectively 'What repeats?', 'Never means impossible?', 'How wrong?'. No dates, portraits,
historical quotations, timelines, equations, or claims about how brains work. This is a
conceptual recipe, not a chronological history.

### memory-and-sources.webp

Two contrasting illustrated desks under heading 'A PLAUSIBLE CITATION NEEDS CHECKING'. LEFT
heading 'GENERATE': a small mechanical text predictor emits a card labeled 'Art. ?' with a large
question mark; behind it is a closed book. RIGHT heading 'RETRIEVE AND CHECK': an open reference
book has an excerpt highlighted and a bookmark; a paper slip copies matching highlighted lines
and a citation tag from that exact book, then passes through a magnifying glass with a
checkmark. At the lower right a second slip with an unmatched highlight leads to a pause card
labeled 'Defer'. Footer exactly 'Real source + relevant passage + checked support'. Do not
depict a real legal article number, advice, a verdict, or invented statutory prose. Use abstract
ink lines for all book and excerpt content. The closed book is an analogy, not a literal source
database inside weights.

### embedding-map.webp

Two-panel teaching illustration showing identity versus learned relationships. LEFT small panel
heading 'TOKEN ID'. A card labeled 'a' connected to an address-book drawer labeled '3', and card
labeled 'u' connected to drawer '7'. Underneath exactly 'Addresses, not meaning'. RIGHT large
panel heading 'EMBEDDING'. A stylized paper map with a cluster of two neighboring note cards
exactly 'ele me bateu' and 'fui agredida'; a third card farther away exactly 'comprei um
celular'. Thin dashed lines show different distances but no numeric scores. Beside the map, a
text card transforms through an arrow into a small row of blue numeric dots or squares (no
readable random numbers), then onto the map. Footer exactly 'Illustrative positions'. Maps
represent learned numerical relationships, not a literal map of the brain or verified legal
classification. Do not draw a classifier sorting decision or claim an exact similarity score.

### learning-and-predicting.webp

Two wide horizontal illustrated lanes. TOP lane heading exactly 'TRAINING: ADJUST THE WEIGHTS'.
A practice card with 'Target: 1' feeds a small mechanical calculator with one prominent
adjustable knob labeled 'w'. To its right output ticket says 'Guess: 0.50'. A teacher's simple
scorecard says 'Loss: 0.69'. A curved coral feedback arrow returns to the knob; next to it a
small label 'Adjust'. A small inset to the right shows the updated knob with 'w = 0.50' and
output 'Guess: 0.62'. The starting knob is labeled 'w = 0'. BOTTOM lane heading exactly
'PREDICTION: KEEP THE WEIGHTS'. A paper context strip enters the same mechanical calculator, now
its knob is secured by a small padlock; a few token cards emerge, one card is appended to the
paper context strip and a curved teal arrow feeds the longer strip back to the calculator INPUT,
never to the knob. Bottom short labels exactly 'Context grows' and 'Weights stay fixed'. Avoid
mathematical formulas beyond these labels. Training and prediction loops must look clearly
different.

### training-steps.webp

A wide illustrated practice workbench under heading 'WATCH THE SAME NETWORK IMPROVE'. Three
snapshots of the same small adjustable mechanical calculator, left to right, each solving the
same four tiny blank practice cards. First snapshot label 'Step 0', ticket 'Loss 0.711'. Middle
label 'Step 50', ticket 'Loss 0.132'. Last label 'Step 2000', ticket 'Loss 0.001'. Knob pointers
differ between snapshots; connecting arrows labeled 'Adjust weights' show training. Below, a
gently descending arrow links the three loss tickets. A separate unopened envelope at bottom
labeled 'No unseen test here' is kept outside the workbench. Footer exactly 'Fitting these four
examples is not proof of generalization'. These labels reproduce recorded lesson examples, not a
new measurement; no additional numerical values.

### shared-weights.webp

An illustrated reusable mechanism under heading 'FOUR LETTERS, SHARED WEIGHTS'. TOP row: four
letter tiles exactly m e d i enter four small vector-card holders, then all four feed one small
connected calculator network with common adjustable knobs. Output to the RIGHT is a fan of
next-letter cards labeled d, a, other; no probabilities. BOTTOM row: four different blank letter
tiles feed the VERY SAME calculator, with a clear arrow showing reuse rather than a duplicate
lookup table. At bottom LEFT a huge mostly empty filing cabinet has label 'One drawer per
context?'; it is crossed out, while the reused mechanism has a checkmark. Footer exactly
'Compute from reusable weights'. No fixed meanings assigned to hidden units, no biological
neurons, no claims that the model understands a word.

### word-pieces.webp

Three broad horizontal rows of physical letter tiles, aligned so merging is obvious. Heading
exactly 'REUSE FREQUENT PIECES'. TOP row label 'Letters': show exactly g u a r d a with all six
separated cream tiles. MIDDLE row label 'Merge a + r': show exactly g u [ar] d a, with ar a
single coral tile that is visibly two-letter wide. BOTTOM row label 'Bigger piece': show exactly
[guard] a, where guard is one long teal tile and a remains a separate tile. Beside the rows a
small stack of repeated word strips 'guarda', 'guarda', 'guardar' suggests counting across
repeated text. Two small arrows join adjacent tile pairs with subtle glue marks; no scissors
cutting meanings or brain imagery. Footer exactly 'Frequent does not mean meaningful'. The
middle and bottom rows illustrate selected merge stages, not consecutive operations. Do not add
a false merge order or numerical counts.

### attention-mixture.webp

One large diagram as an illustrated mixing workbench. Across the TOP: three separate source
index cards labeled exactly 'g', 'u', 'a'. Under each source card show a small payload card
respectively '[1, 0]', '[0, 1]', '[1, 1]'. A context tile at the right labeled 'a now' holds a
magnifying glass and label 'query'. It matches available source cards; do not give the tile a
face or human agency. In the MIDDLE, three wide translucent colored ribbons from the payloads
converge into a mixing bowl. The ribbons are labeled exactly '9%', '24%', '67%' respectively
from g, u, a, with differing widths suggesting those shares. BOTTOM output card from bowl reads
exactly '[0.76, 0.91]' and heading 'WEIGHTED MIX'. Small footer exactly 'Match → shares → mix'.
On the far right a later token card labeled 'r later' is behind a closed screen with a clear
barred arrow; label 'No future tokens'. Only available g,u,a send ribbons. No weight updates or
biological neurons. Exact numerical values are the rounded unscaled toy example, not a real
trained attention trace.

### transformer-stack.webp

Make a very clear vertical model overview with NO complex looping connectors. Heading exactly 'A
TINY GPT'. A centered paper strip at the top shows three letter tiles T H E with small position
tags 1 2 3. Below it are FOUR ENCLOSED cream illustrated module cards stacked vertically with
generous spacing. From top to bottom the cards are labeled exactly 'Layer 1', 'Layer 2', 'Layer
3', 'Layer 4'. Inside EACH card, draw a small teal mixing bowl on the left labeled 'Attention'
and a small calculator on the right labeled 'Network', with one short rightward arrow between
them. Each card represents the entire layer as one unit. Outside the cards, draw ONLY five short
straight DOWN arrows on the central axis: input strip to Layer 1 top border, Layer 1 bottom
border to Layer 2 top border, Layer 2 bottom border to Layer 3 top border, Layer 3 bottom border
to Layer 4 top border, Layer 4 bottom border to a bottom output fan of letter cards labeled
'Next-letter scores'. External arrows connect WHOLE CARD BORDERS only, not internal bowls or
calculators. NO sideways or curved external arrows, NO vertical rail, NO plus symbols, NO
residual wires, NO bypass, NO circular arrows. At the right, a simple small note reads 'Residual
additions not shown'. Footer exactly 'Four layers transform the representations'. Keep each
label readable. This is a simplified architecture overview, not a complete circuit. Four modules
only.

### data-and-time.webp

Three small illustrated study desks arranged left-to-right, heading 'MORE TEXT ALSO NEEDS
PRACTICE'. LEFT desk: one small stack of legal-text pages with a circular rereading arrow, short
label 'Small set, many repeats'. CENTER desk: a much larger stack of legal-text pages beside the
SAME size hourglass as left, short label 'More text, same time'. RIGHT desk: larger legal-text
stack beside a visibly larger hourglass and an unopened exam envelope, short label 'More text,
more practice'. Under all three desks, a row of three small dice suggests repeating experiments
with different random starts; label 'Repeat with different seeds'. Footer exactly 'Judge by
unseen-text loss'. Page contents are abstract ink lines only; no case records, legal advice,
numerical results, promises, or guaranteed winner. The drawings explain data and training budget
as separate choices.

### Attention correction

Edit only the dashed blocked arrow between the mixing bowl and the 'r later' card in the
supplied illustration. Reverse its direction: the arrow must start at the future-token card on
the RIGHT and point LEFT toward the mixing bowl, with the red X continuing to block it. This
means information from future tokens cannot enter the current attention mixture. Keep every
other detail exactly unchanged: composition, colors, paper texture, all source cards, 9%, 24%,
67%, payloads [1, 0], [0, 1], [1, 1], output [0.76, 0.91], and every label. Do not add or remove
any other arrow or text.
