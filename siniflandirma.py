from __future__ import annotations  # Tip açıklamalarını Python sürümleri arasında tutarlı çalıştırmak için.

import argparse  # Eğitim ayarlarını ve dosya yollarını komut satırından almak için.
import csv  # Yorumları okumak ve doğruluk sonuçlarını CSV'ye kaydetmek için.
import math  # Yıldız puanlarının geçerli aralıkta ve sonlu olduğunu denetlemek için.
import random  # BERT eğitim sırasını tekrarlanabilir kılmak için.
import re  # Temel Türkçe metin normalizasyonunda URL ve boşlukları düzenlemek için.
import unicodedata  # Türkçe karakterleri tek bir Unicode biçimine getirmek için.
from collections import Counter  # Duygu sınıfı dağılımını raporlamak için.
from pathlib import Path  # Veri ve sonuç dosyalarının yollarını yönetmek için.
from typing import Any, Optional, Sequence  # Model, sonuç ve CLI tiplerini belirtmek için.

import numpy as np  # Stratified split ve model girdileri için sayısal diziler sağlamak için.
from sklearn.ensemble import RandomForestClassifier  # Rastgele orman sınıflandırıcısını eğitmek için.
from sklearn.feature_extraction.text import TfidfVectorizer  # Yorumları TF-IDF özelliklerine çevirmek için.
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon sınıflandırıcısını eğitmek için.
from sklearn.metrics import accuracy_score  # Eğitim ve test doğruluğunu hesaplamak için.
from sklearn.model_selection import train_test_split  # Bütün modeller için ortak stratified veri bölmesi yapmak için.
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes sınıflandırıcısını eğitmek için.
from sklearn.pipeline import Pipeline  # TF-IDF dönüşümünü modelle birlikte sızıntısız eğitmek için.
from sklearn.svm import LinearSVC  # Doğrusal Support Vector Machine sınıflandırıcısını eğitmek için.


RANDOM_SEED = 42
LABELS = ("olumsuz", "notr", "olumlu")
TURKISH_CASE_TRANSLATION = str.maketrans({"I": "ı", "İ": "i"})
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def sentiment_from_rating(rating: float) -> str:
    if not math.isfinite(rating) or not 1 <= rating <= 5:
        raise ValueError(f"Puan 1 ile 5 arasında olmalı: {rating}")
    if rating <= 2:
        return "olumsuz"
    if rating == 3:
        return "notr"
    return "olumlu"


def load_reviews(input_path: Path) -> tuple[list[str], list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    model_texts: list[str] = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = {"review", "rating"} - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "CSV dosyasında gerekli sütunlar eksik: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            review = (row.get("review") or "").strip()
            if not review:
                continue
            try:
                rating = float(row["rating"])
                sentiment = sentiment_from_rating(rating)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{row_number}. satırda geçersiz puan: {row.get('rating')!r}"
                ) from exc

            # Klasik TF-IDF modelleri, önceki Zemberek + TurkishStemmer çıktısını varsa kullanır.
            prepared_text = (row.get("review_clean") or "").strip()
            if not prepared_text:
                prepared_text = normalize_surface_text(review)
            texts.append(prepared_text)
            labels.append(sentiment)
            # BERT'in alt-kelime tokenizer'ı ekleri ve yazım biçimlerini ham metinde öğrenebilir.
            model_texts.append(normalize_bert_text(review))

    if not texts:
        raise ValueError("CSV dosyasında işlenecek yorum bulunamadı.")
    return texts, labels, model_texts


def normalize_surface_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = URL_PATTERN.sub(" ", text)
    text = text.translate(TURKISH_CASE_TRANSLATION).lower()
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_bert_text(text: str) -> str:
    # BERT modeli cased olduğu için özgün büyük/küçük harf bilgisini sakla.
    text = unicodedata.normalize("NFC", text)
    text = URL_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def make_classical_models() -> dict[str, Pipeline]:
    def vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )

    return {
        "Naive Bayes": Pipeline(
            [("tfidf", vectorizer()), ("classifier", MultinomialNB())]
        ),
        "Lojistik Regresyon": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    LinearSVC(class_weight="balanced", random_state=RANDOM_SEED),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def split_indices(labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    return train_indices, test_indices


def train_classical_models(
    models: dict[str, Pipeline],
    texts: Sequence[str],
    labels: Sequence[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> list[dict[str, Any]]:
    text_array = np.asarray(texts, dtype=object)
    label_array = np.asarray(labels, dtype=object)
    results: list[dict[str, Any]] = []

    for name, model in models.items():
        model.fit(text_array[train_indices], label_array[train_indices])
        train_predictions = model.predict(text_array[train_indices])
        test_predictions = model.predict(text_array[test_indices])
        results.append(
            {
                "model": name,
                "train_accuracy": accuracy_score(
                    label_array[train_indices], train_predictions
                ),
                "test_accuracy": accuracy_score(
                    label_array[test_indices], test_predictions
                ),
            }
        )
    return results


def train_turkish_bert(
    texts: Sequence[str],
    labels: Sequence[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    try:
        import torch  # BERT tensörlerini, veri yükleyicilerini ve optimizasyonu çalıştırmak için.
        from torch.utils.data import DataLoader, TensorDataset  # Token'ları mini-batch'lere bölmek için.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Türkçe BERT modelini ve tokenizer'ını yüklemek için.
    except ImportError as exc:
        raise RuntimeError(
            "BERT bağımlılıklarını kurmak için "
            "'py -m pip install -r requirements-ml.txt' komutunu çalıştırın."
        ) from exc

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    model_name = "dbmdz/bert-base-turkish-cased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    numeric_labels = np.asarray([label_to_id[label] for label in labels])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(
        list(texts),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    label_tensor = torch.tensor(numeric_labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_mask, label_tensor)
    train_dataset = torch.utils.data.Subset(dataset, train_indices.tolist())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_class_counts = np.bincount(
        numeric_labels[train_indices], minlength=len(LABELS)
    )
    class_weights = torch.tensor(
        len(train_indices) / (len(LABELS) * train_class_counts),
        dtype=torch.float32,
        device=device,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch_input_ids, batch_mask, batch_labels = (
                tensor.to(device) for tensor in batch
            )
            optimizer.zero_grad()
            output = model(
                input_ids=batch_input_ids,
                attention_mask=batch_mask,
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits, batch_labels, weight=class_weights
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        average_loss = total_loss / max(len(train_loader), 1)
        print(f"BERT epoch {epoch + 1}/{epochs} - train loss: {average_loss:.4f}")

    def predict(indices: np.ndarray) -> np.ndarray:
        model.eval()
        predictions: list[int] = []
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                logits = model(
                    input_ids=input_ids[batch_indices].to(device),
                    attention_mask=attention_mask[batch_indices].to(device),
                ).logits
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        return np.asarray(predictions)

    train_predictions = predict(train_indices)
    test_predictions = predict(test_indices)
    return {
        "model": "Türkçe BERT (dbmdz)",
        "train_accuracy": accuracy_score(
            numeric_labels[train_indices], train_predictions
        ),
        "test_accuracy": accuracy_score(
            numeric_labels[test_indices], test_predictions
        ),
    }


def save_results(results: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=("model", "train_accuracy", "test_accuracy"),
        )
        writer.writeheader()
        writer.writerows(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_input = Path(__file__).with_name("iphone_yorumlar_hazir.csv")
    default_output = Path(__file__).with_name("siniflandirma_sonuclari.csv")
    parser = argparse.ArgumentParser(
        description="Dört klasik model ve Türkçe BERT ile duygu sınıflandırması yapar."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Etiketli yorum CSV'si")
    parser.add_argument("--output", type=Path, default=default_output, help="Doğruluk sonuç CSV'si")
    parser.add_argument("--epochs", type=int, default=3, help="BERT eğitim epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=8, help="BERT mini-batch boyutu")
    parser.add_argument(
        "--skip-bert", action="store_true", help="Yalnızca dört klasik modeli eğit."
    )
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs ve --batch-size pozitif tam sayı olmalı.")

    try:
        texts, labels, bert_texts = load_reviews(args.input)
        label_counts = Counter(labels)
        if len(label_counts) != len(LABELS) or min(label_counts.values()) < 2:
            raise ValueError(
                "Stratified train/test bölmesi için her duygu sınıfında en az 2 yorum gerekli."
            )

        train_indices, test_indices = split_indices(labels)
        print(f"Toplam yorum: {len(labels)}")
        print(f"Sınıf dağılımı: {dict(sorted(label_counts.items()))}")
        print(
            f"Eğitim/test: {len(train_indices)}/{len(test_indices)} "
            f"(stratified, random_state={RANDOM_SEED})"
        )

        results = train_classical_models(
            make_classical_models(), texts, labels, train_indices, test_indices
        )
        if not args.skip_bert:
            results.append(
                train_turkish_bert(
                    bert_texts,
                    labels,
                    train_indices,
                    test_indices,
                    args.epochs,
                    args.batch_size,
                )
            )

        print("\nModel doğrulukları")
        print(f"{'Model':<28} {'Eğitim':>10} {'Test':>10}")
        for result in results:
            print(
                f"{result['model']:<28} "
                f"{result['train_accuracy'] * 100:>9.2f}% "
                f"{result['test_accuracy'] * 100:>9.2f}%"
            )
        save_results(results, args.output)
        print(f"\nSonuç dosyası: {args.output}")
    except (OSError, csv.Error, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
