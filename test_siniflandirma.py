import csv  # Ratingsiz örnek yorum CSV'leri oluşturup okuyabilmek için.
import tempfile  # Test verilerini proje dosyalarına dokunmadan geçici yerde tutmak için.
import unittest  # Duygu etiketleme ve model hazırlama davranışlarını doğrulamak için.
from pathlib import Path  # Geçici CSV dosyalarının yollarını birleştirmek için.

from siniflandirma import (  # Sınıflandırma akışının temel işlevlerini sınamak için.
    LABELS,
    load_reviews,
    make_classical_models,
    sentiment_from_probabilities,
    split_indices,
)


class ClassificationSetupTests(unittest.TestCase):
    def test_neutral_is_based_on_model_confidence(self) -> None:
        self.assertEqual(sentiment_from_probabilities(0.9), ("olumlu", 0.9))
        self.assertEqual(sentiment_from_probabilities(0.1), ("olumsuz", 0.9))
        self.assertEqual(sentiment_from_probabilities(0.55), ("notr", 0.55))
        self.assertEqual(sentiment_from_probabilities(0.4), ("olumsuz", 0.6))

    def test_rejects_invalid_probabilities_and_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            sentiment_from_probabilities(1.1)
        with self.assertRaises(ValueError):
            sentiment_from_probabilities(0.5, 0.5)

    def test_loads_reviews_without_any_rating_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "comments.csv"
            with input_path.open("w", encoding="utf-8-sig", newline="") as source:
                writer = csv.DictWriter(
                    source, fieldnames=("review_id", "review", "review_clean")
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "review_id": "review-1",
                        "review": "Kameralarımız çok güzel.",
                        "review_clean": "kamera güzel",
                    }
                )

            texts, bert_texts, review_ids, raw_reviews = load_reviews(input_path)

        self.assertEqual(texts, ["kamera güzel"])
        self.assertEqual(bert_texts, ["Kameralarımız çok güzel."])
        self.assertEqual(review_ids, ["review-1"])
        self.assertEqual(raw_reviews, ["Kameralarımız çok güzel."])

    def test_shared_split_is_disjoint_and_stratified(self) -> None:
        labels = ["olumsuz"] * 5 + ["notr"] * 7 + ["olumlu"] * 86
        train_indices, test_indices = split_indices(labels)

        self.assertEqual((len(train_indices), len(test_indices)), (78, 20))
        self.assertFalse(set(train_indices) & set(test_indices))
        self.assertEqual(
            {
                label: sum(labels[index] == label for index in test_indices)
                for label in LABELS
            },
            {"olumsuz": 1, "notr": 1, "olumlu": 18},
        )

    def test_classical_model_list_has_requested_algorithms(self) -> None:
        self.assertEqual(
            set(make_classical_models()),
            {
                "Naive Bayes",
                "Lojistik Regresyon",
                "Support Vector Machine",
                "Random Forest",
            },
        )


if __name__ == "__main__":
    unittest.main()
