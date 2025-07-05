import pandas as pd
import re
import nltk
import shutil
import lightgbm as lgb
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def print_centered(text, width=None):
    if width is None:
        width = shutil.get_terminal_size((80, 20)).columns
    lines = str(text).split('\n')
    for line in lines:
        print(line.center(width))

groundtruth = pd.read_csv("groundtruth.csv")
groundtruth['user_input'] = groundtruth['user_input'].apply(clean_text)
groundtruth['bandmate_text'] = groundtruth['bandmate_text'].apply(clean_text)
groundtruth['combined'] = groundtruth['user_input'] + " [SEP] " + groundtruth['bandmate_text']

vectorizer = TfidfVectorizer().fit(groundtruth['combined'])
X = vectorizer.transform(groundtruth['combined'])
y = groundtruth['label']

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, groundtruth.index, test_size=0.2, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1}
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title("Evaluation Metrics")
plt.ylim(0, 1)
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    return np.sum((2**relevance_scores - 1) / np.log2(np.arange(2, relevance_scores.size + 2))) if relevance_scores.size else 0.

def ndcg_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    ideal_dcg = dcg_at_k(sorted(y_true, reverse=True), k)
    return dcg_at_k(y_true_sorted, k) / ideal_dcg if ideal_dcg != 0 else 0.

def average_precision(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.take(y_true, order)
    relevant = score = 0.
    for i, rel in enumerate(y_true_sorted, start=1):
        if rel:
            relevant += 1
            score += relevant / i
    return score / relevant if relevant else 0.

def mean_average_precision(y_trues, y_scores):
    return np.mean([average_precision(y_true, y_score) for y_true, y_score in zip(y_trues, y_scores)])

def precision_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    return np.mean(np.take(y_true, order)[:k])

def mean_precision_at_k(y_trues, y_scores, k):
    return np.mean([precision_at_k(y_true, y_score, k) for y_true, y_score in zip(y_trues, y_scores)])

test_df = groundtruth.loc[idx_test].copy()
test_df['y_true'] = y_test.values
test_df['y_score'] = y_pred_proba
grouped = test_df.groupby('query_id') if 'query_id' in test_df.columns else [(0, test_df)]

y_trues = [group['y_true'].values for _, group in grouped]
y_scores = [group['y_score'].values for _, group in grouped]
k = 5
ndcg_scores = [ndcg_at_k(yt, ys, k) for yt, ys in zip(y_trues, y_scores)]
map_score = mean_average_precision(y_trues, y_scores)
prec_at_k = mean_precision_at_k(y_trues, y_scores, k)

plt.figure(figsize=(8, 6))
plt.bar(['Mean NDCG@5', 'Mean Average Precision', f'Mean Precision@{k}'], 
        [np.mean(ndcg_scores), map_score, prec_at_k],
        color=['orange', 'green', 'purple'])
plt.title("Ranking Metrics")
plt.ylim(0, 1)
plt.ylabel("Score")
for i, v in enumerate([np.mean(ndcg_scores), map_score, prec_at_k]):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

bandmates = pd.read_csv("bandmates.csv")
bandmates['full_text'] = (
    bandmates['bio'].fillna('') + " " +
    bandmates['genres'].fillna('') + " " +
    bandmates['skills'].fillna('') + " " +
    bandmates['personality'].fillna('')
)
bandmates['clean_text'] = bandmates['full_text'].apply(clean_text)

print_centered("WELCOME TO DREAM TEAM MATCHMAKER")
print_centered("1. Find Full Band")
print_centered("2. Find N Bandmates")
mode = input("Enter 1 or 2: ").strip()

instrument = input("What instrument/role do you play? ").strip()
genres_input = input("What genres do you love? (e.g., Rock, Pop, Jazz) ").strip()
background = input("Tell us a bit about your musical background: ").strip()
personality_input = input("How would you describe your personality? (comma-separated) ").strip()
band_prefs = input("What do you want in your bandmates? ").strip()
skills_input = input("What are your key skills? ").strip()

user_genres = [g.strip().lower() for g in genres_input.split(',')]
user_personality = [p.strip().lower() for p in personality_input.split(',')]
user_skills = [s.strip().lower() for s in skills_input.split(',')]
user_background_text = background.lower()

user_query = " ".join([instrument, genres_input, background, personality_input, band_prefs])
clean_query = clean_text(user_query)
query_vec = vectorizer.transform([clean_query])
doc_vecs = vectorizer.transform(bandmates['clean_text'])

bandmates['similarity_score'] = (query_vec @ doc_vecs.T).toarray().flatten()
bandmates['combined_query'] = clean_query + " [SEP] " + bandmates['clean_text']
combined_vecs = vectorizer.transform(bandmates['combined_query'])
bandmates['ml_match_prob'] = model.predict(combined_vecs)
bandmates['final_score'] = 0.5 * bandmates['similarity_score'] + 0.5 * bandmates['ml_match_prob']

def generate_compatibility_feedback(user_genres, user_personality, user_skills, user_background, band_row):
    band_genres = [g.strip().lower() for g in str(band_row['genres']).split(';') if g.strip()]
    band_personality = [p.strip().lower() for p in str(band_row['personality']).split(';') if p.strip()]
    band_skills = [s.strip().lower() for s in str(band_row.get('skills', '')).split(';') if s.strip()]
    band_background = band_row.get('bio', '').lower() + " " + band_row.get('background', '').lower()

    genre_overlap = set(user_genres) & set(band_genres)
    personality_overlap = set(user_personality) & set(band_personality)
    skills_overlap = set(user_skills) & set(band_skills)
    background_overlap = any(word in band_background for word in user_background.split())

    reasons = []
    if genre_overlap: reasons.append(f"shared genre(s): {', '.join(genre_overlap)}")
    if personality_overlap: reasons.append(f"similar personality trait(s): {', '.join(personality_overlap)}")
    if skills_overlap: reasons.append(f"common skill(s): {', '.join(skills_overlap)}")
    if background_overlap: reasons.append("related musical background")

    return "match based on " + " and ".join(reasons) if reasons else "unique style match despite different traits"

if mode == '1':
    desired_instruments = ['Singer', 'Drums', 'Bass', 'Guitar', 'Keyboard']
    top_candidates_per_instrument = {}
    print_centered("TOP 3 MATCHES PER INSTRUMENT")
    for inst in desired_instruments:
        print_centered(f"Instrument: {inst}")
        candidates = bandmates[bandmates['instrument'].str.lower() == inst.lower()]
        top3 = candidates.sort_values(by='final_score', ascending=False).head(3)
        top_candidates_per_instrument[inst] = top3
        if not top3.empty:
            for _, row in top3.iterrows():
                print_centered(f"{row['name']} (Score: {row['final_score']:.3f})")
        else:
            print_centered("No candidates found.")
        print_centered("-" * 50)

    print_centered("YOUR DREAM TEAM")
    print_centered("WHY ARE YOU COMPATIBLE?")
    for inst, candidates in top_candidates_per_instrument.items():
        if not candidates.empty:
            best = candidates.iloc[0]
            feedback = generate_compatibility_feedback(user_genres, user_personality, user_skills, user_background_text, best)
            print_centered(f"{inst.upper()}: {best['name']} -> {feedback}")
        else:
            print_centered(f"{inst.upper()}: No suitable candidate found.")

elif mode == '2':
    n = int(input("How many bandmates are you looking for? "))
    chosen_bandmates = []

    for i in range(n):
        inst = input(f"Instrument #{i+1}: ").strip()
        candidates = bandmates[bandmates['instrument'].str.lower().str.contains(inst.lower())].sort_values(
            by='final_score', ascending=False).head(5)

        if candidates.empty:
            print_centered(f"No candidates found for {inst}")
            continue

        accepted = False
        for _, row in candidates.iterrows():
            print_centered(f"\nName: {row['name']}")
            print_centered(f"Genres: {row['genres']}")
            print_centered(f"Skills: {row['skills']}")
            print_centered(f"Personality: {row['personality']}")
            print_centered(f"Final Score: {row['final_score']:.3f}")
            decision = input("Swipe → (accept) or ← (reject)? (type 'r' or 'l'): ").strip().lower()
            if decision == 'r':
                chosen_bandmates.append(row)
                accepted = True
                print_centered(f"You accepted {row['name']}!")
                break
            elif decision == 'l':
                print_centered(f"You rejected {row['name']}.")

        if not accepted:
            print_centered("lower your standards man")

    print_centered("\nYOUR FINAL PICKS")
    for row in chosen_bandmates:
        feedback = generate_compatibility_feedback(user_genres, user_personality, user_skills, user_background_text, row)
        print_centered(f"{row['instrument']}: {row['name']} -> {feedback}")
