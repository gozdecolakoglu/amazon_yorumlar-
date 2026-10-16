from __future__ import annotations  # Tip açıklamalarını çalışma anında çözümlemeden kullanmak için.

import argparse  # Komut satırından girdi ve çıktı dosyalarını seçmek için.
import csv  # Yorumları CSV'den okuyup işlenmiş sonuçları CSV'ye yazmak için.
import re  # URL, HTML, noktalama ve fazla boşlukları temizlemek için.
import sys  # Python sürümünü ve Zemberek uyumluluk modülünü yönetmek için.
import types  # Python 3.13'te Zemberek'in beklediği uyumluluk modülünü oluşturmak için.
import typing  # Uyumluluk modülüne TextIO tipini sağlamak için.
import unicodedata  # Türkçe karakterlerin Unicode biçimini tutarlı hale getirmek için.
from pathlib import Path  # Varsayılan ve kullanıcı tarafından seçilen dosya yollarını yönetmek için.
from typing import Callable, Optional, Sequence, Union  # Fonksiyon arayüzlerini ve türleri belirtmek için.

from TurkishStemmer import TurkishStemmer  # Türkçe sözcükleri eklerinden arındırıp köklerine indirmek için.


URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]*>")
NON_WORD_PATTERN = re.compile(r"[^a-zçğıöşü0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
TURKISH_CASE_TRANSLATION = str.maketrans({"I": "ı", "İ": "i"})
TEXT_TOKEN_TYPES = {"Word", "Number", "Abbreviation"}
ZemberekTokenizer = Callable[[str], Sequence[tuple[str, str]]]
TurkishStem = Callable[[str], str]


def create_zemberek_tokenizer() -> ZemberekTokenizer:
    if sys.version_info >= (3, 13):
        # Zemberek'in sabitlenmiş ANTLR sürümü Python 3.13'te kaldırılan typing.io yolunu bekler.
        typing_io = types.ModuleType("typing.io")
        typing_io.TextIO = typing.TextIO
        sys.modules["typing.io"] = typing_io
        setattr(typing, "io", typing_io)

    try:
        from zemberek import TurkishTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Zemberek bağımlılıklarını kurmak için "
            "'py -m pip install -r requirements.txt' komutunu çalıştırın."
        ) from exc

    tokenizer = TurkishTokenizer.DEFAULT
    return lambda text: [
        (token.content, token.type_.name) for token in tokenizer.tokenize(text)
    ]


def clean_review(
    text: str, tokenize: ZemberekTokenizer, stem: TurkishStem
) -> str:
    # NFC, aynı Türkçe harfin farklı Unicode yazımlarını tek biçime getirir.
    text = unicodedata.normalize("NFC", text)
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    # Türkçe büyük I, genel lower() dönüşümünden önce noktasız ı olmalıdır.
    # Zemberek'in metin sınıflarından noktalama/URL'leri atıp sözcük, sayı ve kısaltmaları tut.
    words = [
        stem(token.translate(TURKISH_CASE_TRANSLATION).lower())
        for token, token_type in tokenize(text)
        if token_type in TEXT_TOKEN_TYPES
    ]
    # Stop-word temizliği yapma: "değil" gibi kelimeler duygu etiketini belirleyebilir.
    cleaned_text = NON_WORD_PATTERN.sub(" ", " ".join(words))
    return WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()


def preprocess_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tokenize: ZemberekTokenizer,
    stem: TurkishStem,
) -> tuple[int, int]:
    input_file = Path(input_path)
    output_file = Path(output_path)

    # utf-8-sig, Türkçe metni korurken Excel kaynaklı UTF-8 BOM başlığını da tolere eder.
    with input_file.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing_columns = {"review"} - set(fieldnames)
        if missing_columns:
            raise ValueError(
                "CSV dosyasında gerekli sütunlar eksik: "
                + ", ".join(sorted(missing_columns))
            )

        processed_rows: list[dict[str, str]] = []
        skipped_rows = 0
        for row in reader:
            review = (row.get("review") or "").strip()
            if not review:
                skipped_rows += 1
                continue

            cleaned_review = clean_review(review, tokenize, stem)
            if not cleaned_review:
                skipped_rows += 1
                continue

            processed_rows.append(
                {
                    "review_id": (row.get("review_id") or "").strip(),
                    # Ham alanları denetim için koru; model girdisinde yalnızca review_clean kullanılmalı.
                    "review": review,
                    "review_clean": cleaned_review,
                }
            )

    if not processed_rows:
        raise ValueError("İşlenecek boş olmayan yorum bulunamadı.")

    # Varsayılan hedef ayrı bir dosyadır; böylece kaynak veri korunur.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=("review_id", "review", "review_clean"),
        )
        writer.writeheader()
        writer.writerows(processed_rows)

    return len(processed_rows), skipped_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_input = Path(__file__).with_name("iphone_yorumlar.csv")
    default_output = Path(__file__).with_name("iphone_yorumlar_hazir.csv")
    parser = argparse.ArgumentParser(
        description="iPhone yorumlarını makine öğrenmesi için temizler ve etiketler."
    )
    parser.add_argument("--input", type=Path, default=default_input, help="Girdi CSV")
    parser.add_argument("--output", type=Path, default=default_output, help="Çıktı CSV")
    args = parser.parse_args(argv)

    try:
        tokenize = create_zemberek_tokenizer()
        stemmer = TurkishStemmer()
        processed, skipped = preprocess_csv(args.input, args.output, tokenize, stemmer.stem)
    except (OSError, csv.Error, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"İşlenen yorum: {processed}")
    print(f"Atlanan boş veya temizlenemeyen yorum: {skipped}")
    print(f"Çıktı: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
