import unittest  # Veri bölmesi, sınıf etiketleri ve model listesini sınamak için.
from pathlib import Path  # Proje içindeki örnek CSV yolunu kurmak için.

from siniflandirma import (  # Sınıflandırma akışının yeniden kullanılabilir birimlerini doğrulamak için.
    LABELS,
    load_reviews,
    make_classical_models,
    sentiment_from_rating,
    split_indices,
)


class ClassificationSetupTests(unittest.TestCase):
    def test_rating_is_mapped_to_sentiment_without_using_title_text(self) -> None:
        self.assertEqual(sentiment_from_rating(1), "olumsuz")
        self.assertEqual(sentiment_from_rating(3), "notr")
        self.assertEqual(sentiment_from_rating(5), "olumlu")

    def test_prepared_dataset_contains_expected_labels_and_raw_bert_text(self) -> None:
        dataset_path = Path(__file__).with_name("iphone_yorumlar_hazir.csv")
        texts, labels, bert_texts = load_reviews(dataset_path)

        self.assertEqual(len(texts), 98)
        self.assertEqual(set(labels), set(LABELS))
        self.assertEqual(len(bert_texts), len(texts))
        self.assertNotEqual(texts[0], bert_texts[0])
        self.assertIn("orjinal", texts[0])
        self.assertIn("orjinal", bert_texts[0])

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
