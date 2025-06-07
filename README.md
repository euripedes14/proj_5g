# Project Μαθήματος "Αρχιτεκτονικές 5G, τεχνολογίες, εφαρμογές και βασικοί δείκτες απόδοσης"
 Ακαδημαικό έτος 2024-2025, Εαρινό Εξάμηνο
 ΠΑΝΕΠΙΣΤΗΜΙΟ ΠΑΤΡΩΝ, ΤΜΗΜΑ ΜΗΧΑΝΙΚΩΝ Η/Υ ΚΑΙ ΠΛΗΡΟΦΟΡΙΚΗΣ

# Τίτλος Project
 ΠΡΟΒΛΕΨΗ ΠΟΛΥΔΙΑΣΤΑΤΩΝ ΧΡΟΝΙΚΩΝ ΣΕΙΡΩΝ ΣΕ ΔΕΔΟΜΕΝΑ ΚΙΝΗΣΗΣ ΔΙΚΤΥΩΝ 5G 
(ΠΕΡΙΠΤΩΣΗ YOUTUBE)

# Μέλη Ομάδας
ΜΠΑΛΑΣΗ ΔΗΜΗΤΡΑ, 1093440
ΠΑΠΟΥΤΣΗ ΑΓΓΕΛΙΚΗ ΕΙΡΗΝΗ, 1093473

---

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- tkinter

Install dependencies with:
```bash
pip install torch numpy scikit-learn matplotlib
```
(`tkinter` is included with most Python installations.)


## Εκτέλεση Προγράμματος

### 1. Προεπεξεργασία Δεδομένων
```bash
python proepejergasia_multi.py
python proepejergasia_dl.py
```

### 2. Εκμάθηση Μοντέλων

Transformer model:
```bash
python train_transformer.py
```
LSTM models για UL και DL:
```bash
python train_lstm_ul.py
python train_lstm_dl.py
```

### 3. Αξιολόγηση Μοντέλων

Transformer model:
```bash
python transformer_eval.py
```
LSTM models:
```bash
python ul_eval.py
python dl_eval.py
```

### 2. Δοκιμή των Μοντέλων 

Μπορείτε να τρέξετε τα μοντέλα χρησιμοποιώντας το GUI που έχουμε φτιάξει για εύκολο έλεγχο και αξιολόγηση.

```bash
python cmp_models.py
```
- Επιλέγετε το μοντέλο που θέλετε να χρησιμοποιήσετε.
- Επιλέγετε πόσες φορές θέλετε να τρέξει το μοντέλο.
- Πατάτε "OK".
- Το μοντέλο που επιλέξατε θα τρέξει και έπειτα θα εμφανιστούν στην οθόνη σας τα αντίστοιχα διαγράμματα και οι τιμές MAPE και RMSE.

---

### Lisence 

Το Project αυτό υλοποιήθηκε για ακαδημαικό σκοπό με θέμα που μας δώθηκε από τον κ. Νίκο Ράπτη(nraptis@isi.gr).

---