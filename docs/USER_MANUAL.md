# User Manual

## What Is This App?

This application classifies Raman spectra of garnet minerals into one of five types: **Almandine, Andradite, Grossular, Pyrope, or Spessartine**. Upload a spectrum file from your spectrometer and receive an instant classification with confidence score.

Open the application at: **http://localhost:8501**

---

## How To Predict

1. Go to the **Predict** tab
2. Upload a `.txt` spectrum file or a `.zip` of multiple spectra
3. The file must contain two columns: **wavenumber** and **intensity**
4. Click the file uploader — prediction runs automatically
5. View the predicted class and confidence score

**Accepted file format (.txt):**
```
200.6    733.9
201.5    493.0
202.6    964.2
```
Two columns: wavenumber (cm⁻¹) and intensity, tab or space separated, no header.

---

## Understanding Results

| Indicator | Meaning |
|-----------|---------|
| Confidence ≥ 70% | High confidence — reliable prediction |
| Confidence 50–70% | Low confidence — review recommended |
| Drift Score > 0.5 | Spectrum differs from training data — results may be less reliable |

---

## How To Submit Feedback

1. Go to the **Pending Feedback** tab
2. Find predictions without ground truth
3. Select the correct class from the dropdown
4. Click **Submit Feedback**
5. Wrong predictions are saved for future model retraining after review by domain expertise

---

## FAQ

**What file types are supported?**
Single `.txt` spectrum files or `.zip` archives containing multiple `.txt` files.

**What if my prediction shows Low Confidence?**
The model is uncertain. Verify visually and submit correct feedback.

**What does Drift Score mean?**
A high drift score means your spectrum differs significantly from training data. Results may be less reliable.

**How do I improve model accuracy?**
Submit feedback with correct labels. Wrong predictions are automatically collected for future model retraining.
