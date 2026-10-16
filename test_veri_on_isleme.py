import csv
import tempfile
import unittest
from pathlib import Path

from TurkishStemmer import TurkishStemmer

from veri_on_isleme import (
    clean_review,
    create_zemberek_tokenizer,
    preprocess_csv,
)


class CleanReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenize = create_zemberek_tokenizer()
        cls.stem = TurkishStemmer().stem

    def test_cleans_text_without_removing_turkish_negation(self) -> None:
        # HTML/URL temizliğini sınarken "değil" kelimesinin duygu sinyali olarak kaldığını doğrula.
        result = clean_review(
            "İPHONE çok iyi değil! <b>Fiyatı</b> pahalı. https://example.com",
            type(self).tokenize,
            type(self).stem,
        )
        self.assertEqual(result, "iphone çok iyi değil fiyat paha")

    def test_uses_zemberek_turkish_tokenization(self) -> None:
        # Tokenizer'ın noktalama işaretlerini ayıklayıp sayıları ayrı token'lara böldüğünü doğrula.
        result = clean_review(
            "Kamera çok güzel değil; fiyatı 12.500 TL.",
            type(self).tokenize,
            type(self).stem,
        )
        self.assertEqual(result, "kamera çok güzel değil fiyat 12 500 tl")

    def test_stems_turkish_suffixes(self) -> None:
        # Ek almış farklı biçimlerin aynı köke indirgenmesi model özelliklerini birleştirir.
        result = clean_review(
            "Kameralarımızdan", type(self).tokenize, type(self).stem
        )
        self.assertEqual(result, "kamera")


class PreprocessCsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenize = create_zemberek_tokenizer()
        cls.stem = TurkishStemmer().stem

    def test_processes_comments_without_creating_rating_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # Geçici dosyalar, örnek CSV testinin gerçek kullanıcı verisini etkilemesini önler.
            input_path = Path(directory) / "input.csv"
            output_path = Path(directory) / "nested" / "output.csv"
            with input_path.open("w", encoding="utf-8-sig", newline="") as source:
                writer = csv.DictWriter(
                    source, fieldnames=("review_id", "rating", "review")
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"review_id": "a", "rating": "1.0", "review": "Hiç iyi değil!"},
                        {"review_id": "b", "rating": "not-a-rating", "review": "Çok güzel."},
                        {"review_id": "c", "rating": "5.0", "review": "   "},
                    )
                )

            count, skipped = preprocess_csv(
                input_path, output_path, type(self).tokenize, type(self).stem
            )

            self.assertEqual((count, skipped), (2, 1))
            with output_path.open(encoding="utf-8-sig", newline="") as result:
                rows = list(csv.DictReader(result))
            self.assertEqual(rows[0]["review_clean"], "hiç iyi değil")
            self.assertEqual(rows[1]["review_clean"], "çok güzel")
            self.assertEqual(set(rows[0]), {"review_id", "review", "review_clean"})


if __name__ == "__main__":
    unittest.main()
