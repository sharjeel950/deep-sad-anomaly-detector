# Viva / presentation cheat-sheet

## 30-second pitch
Anomalies are rare and hard to label, but in practice we often know a *few* of them. Deep SAD trains a neural network to map normal data into a small hypersphere around a fixed centre, and uses the few labeled anomalies to push known-bad regions *away* from that centre. The anomaly score is the distance to the centre.

## Likely questions

**1. What is the difference between Deep SVDD and Deep SAD?**
Deep SVDD is unsupervised: it only minimises the distance of all data to the centre. Deep SAD adds a term for labeled data: labeled normals are pulled in, labeled anomalies are pushed out via the *inverse* squared distance. With zero labels Deep SAD *is* Deep SVDD (the left edge of the Label-efficiency plot).

**2. Why the inverse distance for anomalies?**
Minimising `1/(d²+ε)` makes the loss smaller the farther the anomaly is from `c`, so labeled anomalies are repelled. A plain "maximise the distance" would be unbounded.

**3. What is hypersphere collapse and how do you avoid it?**
If the network has bias terms it can output the constant `c` for every input, giving zero loss and a useless detector. We use bias-free linear layers and batch-norm without affine parameters, and keep every coordinate of `c` at least 0.1 away from zero.

**4. Why pretrain with an autoencoder?**
It gives the encoder a meaningful starting point, so the centre (mean of the initial embeddings) is representative of normal data. Labeled anomalies are excluded from pretraining so the autoencoder doesn't learn to reconstruct them.

**5. What does η do?**
It weights the labeled term. Large η = the few labels matter more; η = 0 ignores labels. The paper finds results are fairly robust for η around 0.5–5.

**6. How is the threshold chosen?**
The score is continuous; the app uses the (1 − expected anomaly rate) quantile of the training scores as the default and lets you move it. Threshold-free metrics (AUROC, average precision) are the main comparison.

**7. Why does Isolation Forest win on the 2-D toy data?**
In 2-D, axis-aligned random splits isolate clusters and scattered points very efficiently, and a deep network has little advantage. On higher-dimensional and real data Deep SAD is ahead – we report this honestly in `docs/BENCHMARK.md`.

**8. Why is a supervised classifier not enough?**
With a handful of labeled anomalies a classifier overfits to *those* anomaly types. The "labels cover only one of three types" experiment shows the Random Forest dropping (e.g. 0.954 → 0.747 AUROC on Two Moons) while Deep SAD keeps modelling normality and stays robust.

**9. How did you avoid data leakage?**
The test set is split first, fully labeled, and never used for scaling, pretraining, centre initialisation or thresholds. In the label-efficiency study only the *labeled subset* changes between repeats.

**10. What are the limitations?**
MLP only (no images / sequences), threshold relies on an assumed contamination rate, hyper-parameters tuned by hand.

## Demo script (3 minutes)
1. **Data tab** – show the toy data: grey unlabeled cloud, a few orange labeled anomalies.
2. **Train tab** – press *Train Deep SAD*; point out the two loss curves (pretraining, Deep SAD).
3. **Evaluate tab** – comparison table, ROC/PR, move the threshold and watch the confusion matrix; show the score surface (2-D data).
4. **Label efficiency** – run it; explain that 0 labels = Deep SVDD and a few labels already help.
5. **Inference** – edit a feature value in a row and watch the verdict flip.
