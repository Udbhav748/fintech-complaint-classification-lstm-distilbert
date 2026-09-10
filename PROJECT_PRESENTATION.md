# Project Presentation: CFPB Complaint Classification

## 1. What this project is

I took real customer complaints filed with the Consumer Financial Protection Bureau (CFPB), a US government agency, and built a model that reads the complaint text and figures out which of 5 financial product categories it belongs to.

The 5 categories are:
1. Checking or savings account
2. Credit card
3. Debt collection
4. Money transfer, virtual currency, or money service
5. Student loan

My goal wasn't just to build one model and report a score. I wanted to start from a simple model, add one improvement at a time, and measure exactly how much each improvement helped, with real numbers, not guesses. Then, at the end, I compared the best simple model against a modern pretrained transformer, DistilBERT, to see which one won and by how much.

## 2. Why this matters

A bank or financial company receives thousands of complaints a day, and someone has to read each one and route it to the right team. Doing this by hand is slow and expensive. If a model can read the complaint and correctly guess the category, that routing can happen instantly. That's the practical motivation behind this project.

## 3. The dataset

1. Source: CFPB Consumer Complaint Database.
2. Raw size: 107,992 complaints across the 5 categories.
3. After I removed exact duplicate complaints: 101,802 complaints remain.
4. Class sizes are fairly close to each other: Checking or savings 21,524, Money transfer 20,940, Credit card 20,684, Student loan 19,985, Debt collection 18,669. That's roughly a 1.15 to 1 ratio between the biggest and smallest class, mild imbalance, not a serious one.
5. I split the data 80% train, 10% validation, 10% test, and froze that split (saved to disk) so every model gets tested on the exact same data.

## 4. How I organized the project, stage by stage

### Stage 1: Data audit (notebook 01_eda.ipynb)

Before touching any model, I checked the raw data carefully for problems that could quietly wreck the results later. Questions like: is the data clean, are there duplicates, could the model cheat, is anything missing.

What I found and decided:

1. Schema and labels were correct: 107,992 rows, 5 products, no missing or blank complaints, no fully duplicate rows.
2. 7,038 rows repeated text already seen elsewhere (from just 848 distinct texts), mostly template credit-repair letters. I removed these before splitting the data. If the same complaint text appears in both training and test data, the model doesn't truly learn, it just memorizes and looks better than it really is.
3. Even after removing exact duplicates, 7.8% of complaints still had a near-identical twin somewhere else, worded almost the same, just a different name or amount. That's a subtler form of leakage. I measured it directly: a simple model scored 0.922 on the contaminated documents but only 0.858 on the clean ones, a real gap. So I built the train, validation, and test split to keep near-duplicate groups together in one split instead of letting copies leak across all three.
4. Complaint length: typical complaint is 178 words, but some run past 5,000 words. Cutting text at 128 words was a real limitation since the model would only read about half the text on average. That's why I tested a longer 256-word limit later, to see how much it actually helps.
5. CFPB replaces private details, names, dates, amounts, with a placeholder, XXXX. I tested directly whether this placeholder was secretly giving away the answer. Guessing the category using only XXXX counts and nothing else got 0.274 accuracy, barely above the 0.200 you'd get from pure guessing across 5 classes. So it's not a shortcut, and I kept it in the text rather than removing it.
6. I checked whether the task was too easy for a bad reason. People often name their product directly in the complaint, a Student loan complaint often literally says "loan", and that showed up clearly in the data. That's a real, honest signal, not a red flag.
7. I also checked whether pretrained GloVe word vectors would actually help. GloVe covers 99% of all words by raw count, but many of the words that matter most for telling categories apart, brand names like "navient" or "coinbase", are missing from it. So I expected the GloVe experiment to give a real but smaller boost than "99% coverage" makes it sound like.
8. Given the mild 1.15 to 1 class imbalance, I decided against class weighting, since at that ratio it wouldn't meaningfully change results.

### Stage 2: Preprocessing (notebook 02_preprocessing.ipynb)

Here I turned the audit's decisions into an actual pipeline: cleaned the text (fixing an encoding glitch found in 673 rows), built the tokenizer with a 20,000-word vocabulary, and built the final frozen train/validation/test split that every model below trains and is tested on.

### Stage 3: Baseline model, M0

My first real model is a simple one-directional LSTM with randomly initialized word embeddings. Nothing fancy. This is the number every later improvement gets compared against.

Result: Macro-F1 0.8508, averaged and checked across 3 different random seeds to know how much the score naturally wobbles from run to run, a spread of about ±0.0021.

### Stage 4: The enhancement ladder

Starting from M0, I added one change at a time, and measured each one against the previous step and against the original baseline.

**M1: Bidirectional LSTM**
Change: the LSTM now reads the sentence both forward and backward instead of just forward.
Result: 0.8485 Macro-F1, actually slightly lower than M0, and the difference is small enough to fall inside M0's own natural seed-to-seed variation. In plain terms, bidirectionality alone didn't clearly help here, it may just be noise.

**M2: BiLSTM plus dropout and recurrent dropout**
Change: I added Spatial Dropout (0.2) and regular Dropout (0.3), plus recurrent dropout (0.2) inside the LSTM itself, to fight overfitting.
Result: 0.8518 Macro-F1. Still close enough to M0 that the raw number alone isn't conclusive, but the training curves clearly showed less of a gap between training and validation performance, meaning the model was overfitting less. That's a real effect even if the headline number doesn't move much yet.

**M3: BiLSTM plus pretrained GloVe embeddings**
Change: instead of starting the word embeddings from random numbers, I started them from pretrained GloVe vectors that already encode word meaning learned from a huge outside body of text.
Result: 0.8657 Macro-F1, a clear, real jump, well outside M0's natural variation. This is the first change in the ladder that unambiguously helped.

**M4: BiLSTM plus GloVe plus learning-rate scheduling, early stopping, and a longer 256-word limit**
Change: three things bundled together due to a limited compute budget: a learning rate scheduler that lowers the learning rate when progress stalls, early stopping that halts training once validation stops improving, and doubling the text length limit from 128 to 256 words.
Result: 0.8710 Macro-F1, the best score in the whole recurrent (LSTM-based) ladder.
Honest detail I found while building the training curve chart: the learning rate scheduler's one and only drop actually happened after the validation score had already peaked and started falling. So the credit for stopping the model from overfitting further really belongs to early stopping, not the learning rate drop. That's easy to assume without checking, I checked it directly against the actual epoch-by-epoch numbers.

### Stage 5: The transformer, D0

Instead of another LSTM variant, my final experiment swaps the whole approach: fine-tune DistilBERT, a smaller, faster version of the BERT transformer, already pretrained on a massive amount of general text.

Result: 0.8759 Macro-F1, averaged across 3 seeds, spread about ±0.0018, the best score of the entire project. All three DistilBERT seeds individually beat M4's single result.

Worth knowing: DistilBERT's tokenizer breaks words into smaller sub-word pieces, so at the same 256-word setting it actually reads less of each complaint than the LSTM does, about 56% of a typical complaint versus about 69% for the LSTM. Despite reading less raw text, it still won, which makes the result more convincing, not less.

Also worth knowing: DistilBERT reached its best validation score in about 4 epochs, while M4 needed about 8 epochs and finished lower. That's the practical value of starting from a model that already understands language in general, instead of learning everything from scratch on this one dataset.

### Stage 6: Final comparison and error analysis (notebook 05_final_results.ipynb)

Here I pull every result into one place, calculate the improvement (delta) of each step versus the one before it and versus the original baseline, and figure out which deltas are real signal versus which ones are small enough to be just noise.

I also compare M4 (best LSTM-based model) against D0 (DistilBERT) directly on the same 10,181 test complaints:
1. D0 improves per-class F1 in 3 of the 5 categories (Checking/savings, Credit card, Money transfer) and is very slightly behind M4 on the other 2 (Debt collection, Student loan). The differences are small either way.
2. Out of all test complaints, D0 fixed 422 that M4 got wrong, while getting 364 wrong that M4 had gotten right. Net gain: 58 examples in D0's favor, which lines up with its higher overall score.
3. Checking/savings is the hardest category for both models. Student loan is the easiest for both.
4. Neither model has one big, obvious blind spot. DistilBERT's advantage is spread out across categories rather than concentrated in one.

## 5. Where and how I actually trained these models

The heavier models, M2, M3, M4, and DistilBERT, were too slow to train on my laptop CPU in reasonable time, so I ran training on Kaggle's free notebook infrastructure (GPU and CPU kernels), one kernel per experiment per seed. I pulled the output of each run (trained weights, metrics, logs) back down and stored it in this project under checkpoints/ and results/, so the project doesn't depend on Kaggle staying available to show its results.

Training time varied a lot between models, and not for the reason you'd guess:
1. M0 (the plain baseline) trains in under 3 minutes per run.
2. M1 (just adding bidirectionality) takes about 4.5 minutes.
3. M2, M3, and M4 jump to roughly 1.5 to 3 hours each, even though M2 and M3 have almost the same number of parameters as M1. The reason is recurrent dropout, a regularization setting I added from M2 onward. Turning it on disables a fast, GPU-optimized LSTM code path, so training slows down enormously even though the model itself isn't meaningfully bigger.
4. DistilBERT (D0) has about 30 times more parameters than M4 (67 million versus 2.2 million), yet trains faster per run (about 90 minutes versus M4's roughly 180 minutes), because it never hits that recurrent dropout penalty, since it isn't a recurrent model at all. Bigger isn't always slower.

## 6. Data integrity and reproducibility checks

I built in a few engineering safeguards so the results can be trusted and reproduced, not just taken on faith:
1. The dataset and the frozen train/validation/test split are both fingerprinted with SHA-256 hashes, so I can verify later that I never accidentally used a different version of the data for one experiment versus another.
2. Every run's metadata records the exact git commit and, for Kaggle runs, the exact kernel and dataset version that produced it.
3. A project-wide check script (scripts/project_check.py) verifies the project structure, configuration, split manifest, and experiment definitions, and runs the full automated test suite (253 tests) before I trust anything as "done."
4. Every recurrent model's checkpoint saving is set to keep only the best-performing epoch on validation Macro-F1, not just whatever the last epoch happened to produce.

## 7. Model settings, for reference

| Setting | M0 | M1 | M2 | M3 | M4 | D0 |
|---|---|---|---|---|---|---|
| Type | Uni-LSTM | BiLSTM | BiLSTM | BiLSTM | BiLSTM | DistilBERT |
| Embedding | random, 100d | random, 100d | random, 100d | GloVe, 100d | GloVe, 100d | built-in |
| Hidden size | 128 | 128 | 128 | 128 | 128 | built-in |
| Dropout | none | none | 0.3 | 0.3 | 0.3 | built-in |
| Recurrent dropout | none | none | 0.2 | 0.2 | 0.2 | not applicable |
| Max text length | 128 words | 128 words | 128 words | 128 words | 256 words | 256 tokens |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 (scheduled) | 0.00002 |
| Optimizer | Adam | Adam | Adam | Adam | Adam | AdamW |
| Epochs (max) | 10 | 10 | 10 | 10 | 10 | 4 |
| Early stopping | no | no | no | no | yes | yes |
| Parameters | 2,117,893 | 2,235,781 | 2,235,781 | 2,235,781 | 2,235,781 | 66,957,317 |

I capped the vocabulary for every LSTM-based model at the 20,000 most common words (out of 57,396 unique words in the data), which still covers 99.78% of all the text. Anything rarer gets mapped to a generic "unknown word" token instead of being dropped from the input.

## 8. Full results table

| Model | What it is | Macro-F1 | Accuracy | Change from previous step | Change from baseline (M0) |
|---|---|---|---|---|---|
| TF-IDF (reference) | Simple word-count based linear model, not part of the ladder | 0.8707 | 0.8682 | not applicable | not applicable |
| M0 | Unidirectional LSTM, random embeddings (baseline) | 0.8508 | 0.8483 | not applicable | not applicable |
| M1 | Bidirectional LSTM | 0.8485 | 0.8455 | -0.0023 (likely just noise) | -0.0023 |
| M2 | BiLSTM + dropout + recurrent dropout | 0.8518 | 0.8485 | +0.0033 (small) | +0.0010 |
| M3 | BiLSTM + pretrained GloVe embeddings | 0.8657 | 0.8627 | +0.0139 (real improvement) | +0.0148 |
| M4 | BiLSTM + GloVe + LR schedule + early stopping + 256-word limit | 0.8710 | 0.8682 | +0.0054 (real improvement) | +0.0202 |
| D0 | Fine-tuned DistilBERT | 0.8759 | 0.8737 | +0.0048 vs M4 (consistent across seeds) | +0.0250 |

A simple word-count model, TF-IDF, already scores 0.8707, almost as high as the best LSTM (M4, 0.8710) and not far behind DistilBERT. That's not a flaw in the project, it's a genuine finding. This task is strongly lexical, people tend to describe their product directly in the complaint, so even a model with no understanding of word order or context does quite well. The real value of DistilBERT is that it still wins on top of that already-strong baseline.

## 9. What I didn't try, and why

1. I didn't try stacked LSTM layers, more than one LSTM layer on top of each other. Given a fixed, pre-planned compute budget of 10 total training runs, I spent that budget on the ladder above instead.
2. I didn't use class weighting, since the class imbalance was mild (1.15 to 1) and wouldn't meaningfully change results at that ratio.
3. I only ran M1, M2, and M3 with a single random seed each (only M0 and D0 got 3 runs), again due to the fixed compute budget. That means I know how much M0 and D0's scores naturally wobble, but not M1 through M4 individually. Every conclusion about them is stated carefully with that limitation in mind.
4. The comparison between D0 (3 seeds) and M4 (1 seed) isn't a formal statistical significance test. It's accurate to say D0 beat the one observed M4 result across all three of its own seeds, but not accurate to claim a proven statistical difference.

## 10. The one-paragraph summary, if asked to explain the whole project in 30 seconds

This project classifies real CFPB financial complaints into 5 categories. I started with a plain LSTM baseline, added one change at a time, bidirectional reading, dropout, pretrained GloVe embeddings, learning rate scheduling, early stopping, a longer context window, and measured the exact improvement from each change against the baseline. I finished by fine-tuning a DistilBERT transformer. GloVe and the final bundle of tweaks gave the two clear, real wins in the LSTM ladder. DistilBERT then beat the best LSTM model by a further, consistent margin across seeds, while a simple word-count model showed the task is strongly lexical to begin with.

---

## 11. Likely examiner questions

### Basic concept questions

**What is an LSTM?**
A type of neural network built for sequences (like sentences), which reads words one at a time and keeps a running memory of what it has seen so far, so it can use earlier words to understand later ones.

**What does "bidirectional" mean?**
A bidirectional LSTM reads the sentence twice, once left to right and once right to left, and combines both readings. The idea is that context from later words can help understand earlier words too.

**What is an embedding?**
A way of turning a word into a list of numbers (a vector) so a neural network can process it. Similar words end up with similar numbers.

**What is GloVe?**
A set of pretrained word embeddings. Instead of starting each word's numbers from random values and learning them from scratch on this one dataset, GloVe gives the model a head start using word meanings already learned from a much larger amount of text.

**What is DistilBERT?**
A smaller, faster version of BERT, a transformer model that reads an entire sentence at once (instead of one word at a time like an LSTM) and figures out how every word relates to every other word. It comes pretrained on a huge amount of general text, and I fine-tune it (keep training it a bit further) on the CFPB complaints specifically.

**What does "fine-tuning" mean?**
Taking a model that is already trained on a large, general dataset, and training it a little more on your specific, smaller dataset, so it adapts to your exact task without having to learn language from zero.

**What is Macro-F1, and why not just use accuracy?**
F1 balances two things: how many of the model's guesses for a category were correct (precision), and how many of the real examples in that category the model actually caught (recall). Macro-F1 calculates this separately for each of the 5 categories and then averages them equally, so one big category can't hide poor performance on a smaller one. Since the categories here are fairly close in size, macro-F1 and accuracy end up close together, but macro-F1 is still the safer, fairer choice.

**What is overfitting?**
When a model starts memorizing quirks of the training data instead of learning general patterns, so it looks great on training data but performs worse on new, unseen data.

**What is a train/validation/test split, and why keep it frozen?**
Training data teaches the model, validation data is used during training to decide things like when to stop, and test data is only touched once at the very end to report the final score. Freezing the split (saving the exact same row assignments to disk) means every model in the project gets judged on the exact same unseen data, which makes the comparison fair.

**What is a random seed, and why run some models with 3 different seeds?**
A seed controls the random numbers used when starting training (like the initial random word embeddings). Two runs with different seeds but the same setup will get slightly different scores just by chance. Running the same model 3 times with 3 different seeds shows how much that natural chance variation actually is, so a real improvement can be told apart from ordinary luck.

**What is a confusion matrix?**
A table showing what the model actually predicted versus what the true answer was, for every category, so you can see exactly which categories get mixed up with each other.

### Project-specific questions

**Why start with a plain baseline instead of just using DistilBERT from the start, since it wins anyway?**
Because my goal was to measure the value of each individual improvement, not just report the single best score. Starting from a plain LSTM and adding one change at a time shows exactly how much bidirectionality, dropout, pretrained embeddings, and a longer context window each contribute on their own. It also lets me compare a strong pretrained transformer against the best version of a from-scratch model, not just a weak one.

**Where did you actually train these models?**
The lighter models train in minutes on a normal CPU, but I trained the heavier ones, M2 through M4, and especially DistilBERT, on Kaggle's free notebook infrastructure, one run per experiment per seed, since some individual runs took 1.5 to 3 hours. I downloaded the trained weights and all metrics afterward and stored them in this project's checkpoints/ and results/ folders, so the results don't depend on Kaggle staying available.

**How do you know the pipeline itself is correct, separate from whether the results are good?**
I built a full automated test suite, 253 tests, covering data loading, preprocessing, splitting, metric calculations, and checkpoint logic, plus a project-wide check script that verifies the folder structure, configuration files, and split manifest are all consistent before I treat any result as final.

**How is people's private information handled in this data?**
CFPB already removes and replaces personal details like names, dates, and dollar amounts with a placeholder before publishing the complaints. I don't attempt to recover that information. I even checked directly whether a model could exploit the placeholder pattern as a shortcut instead of reading the actual complaint, and confirmed it couldn't, 0.274 accuracy from placeholders alone, barely above random guessing.

**Why did you remove duplicate complaints?**
Because 7,038 rows were repeated text, mostly template letters, and if the exact same complaint appears in both training and test data, the model can just memorize it instead of genuinely learning. That would make the score look better than it really is.

**What is near-duplicate leakage, and why was it a bigger deal than exact duplicates?**
Exact duplicates are byte-for-byte identical text. Near-duplicates are the same template complaint with small changes, a different name or dollar amount, which a simple duplicate check would miss. I measured this directly: a model scored 0.922 on near-duplicate-contaminated test documents versus 0.858 on clean ones, a genuine 6-plus point gap. So I built the data split to keep near-duplicate groups entirely inside one split, train, validation, or test, not scattered across all three.

**Why did you keep the XXXX redaction placeholder instead of removing it?**
Because I tested it directly. Guessing the category using only redaction counts got 0.274 accuracy, barely above the 0.200 you'd get by pure random guessing across 5 classes. So it carries almost no shortcut signal, while removing it would throw away a real clue that a name, date, or amount was mentioned at that point in the sentence.

**Why didn't you use class weights, given there is class imbalance?**
The imbalance is mild, about 1.15 to 1 between the biggest and smallest category. At that ratio, class weighting would only rescale the loss by at most about 15%, not enough to meaningfully move the result on a dataset with roughly 19,000 examples per class.

**Why is TF-IDF (a simple, non-neural model) scoring almost as well as your neural models?**
Because this task is strongly lexical. People tend to name their product directly in the complaint, a Credit card complaint often literally mentions a card, so a model that just counts which words appear already captures most of the signal. That's a genuine property of the dataset, not a flaw in the neural models.

**Why did bidirectionality (M1) not clearly improve the score?**
Its score, 0.8485, was actually a little below the baseline, 0.8508, and the gap is smaller than the baseline's own natural seed-to-seed variation, about ±0.0021. So the honest conclusion is that I couldn't show this specific change helped, not that it definitely hurt.

**Why did GloVe (M3) help so clearly when bidirectionality (M1) didn't?**
GloVe gives the model word meanings learned from a huge amount of outside text, instead of making it learn every word's meaning from scratch using only this dataset. That's a much bigger, more useful head start than just letting the model read the sentence in two directions.

**You said the learning rate schedule "wasn't really responsible" for M4's improvement. Explain that.**
When I plotted the actual epoch-by-epoch validation scores, the one and only learning rate drop happened at epoch 7, but the validation score had already peaked at epoch 5 and was already falling by then. So the learning rate drop reacted to the problem after the fact, it didn't cause the good result. The real credit for stopping the model before it overfit further belongs to early stopping. I only found this by actually checking the numbers rather than assuming the bundled changes worked the way I expected.

**Is DistilBERT beating your best LSTM model (M4) a fair, proven comparison?**
Not a formally proven one. I ran DistilBERT (D0) with 3 different seeds and it beat M4's score every single time, a strong pattern. But I only ran M4 once due to the compute budget, so I don't know M4's own natural seed-to-seed variation, and I didn't run a real statistical significance test. The honest claim is "D0 consistently beat the one M4 result I have," not "D0 is proven better than M4."

**DistilBERT reads less of each complaint than your LSTM at the same length setting. Why?**
DistilBERT's tokenizer breaks unfamiliar or complex words into smaller sub-word pieces, for example "navient" becomes three pieces, so the same 256-token limit covers less of the actual complaint, about 56% versus about 69% for the LSTM's simpler tokenizer. DistilBERT still won despite this handicap, which makes its advantage more convincing, not less.

**What would you do next if you had more time or compute?**
I'd try stacked LSTM layers, test class weighting anyway to confirm it truly doesn't help, run M1 through M4 across multiple seeds to know their real variation, and give DistilBERT a longer, GPU-backed training run to see if the gap over the LSTM ladder grows further.

**How do you know your results are reproducible and not just a lucky run?**
I ran the baseline (M0) and the final model (D0) 3 times each with different random seeds, and their spreads were small, about ±0.0021 and ±0.0018 respectively, which shows the results are stable rather than a fluke. The data split, tokenizer settings, and preprocessing steps are all frozen and version-checked, so re-running the pipeline reproduces the same inputs every time.

**Why compare against a simple TF-IDF model at all?**
To have an honest floor to check every fancier model against. If a complicated neural model can't beat a simple word-count model, that's important to know. In this case it also reveals something true about the dataset itself, that it's strongly lexical, not just about the models.
