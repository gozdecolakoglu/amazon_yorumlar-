# iPhone yorumları: makine öğrenmesi için ön işleme

`iphone_yorumlar.csv` dosyasını Zemberek ile Türkçe token'lara ayırıp
`TurkishStemmer` ile sözcükleri köklerine indirerek makine öğrenmesinde
kullanılabilir bir CSV üretmek için Python 3.9 veya üzeriyle çalıştırın. Önce
bağımlılıkları yükleyin:

```powershell
py -m pip install -r .\requirements.txt
```

Ardından ön işlemi başlatın:

```powershell
py .\veri_on_isleme.py
```

Varsayılan çıktı `iphone_yorumlar_hazir.csv` dosyasıdır. Farklı dosya yolları
seçilebilir:

```powershell
py .\veri_on_isleme.py --input .\iphone_yorumlar.csv --output .\cikti\yorumlar.csv
```

İşlem, yorum metnini NFC biçimine getirir; URL ve HTML parçalarını temizler;
Zemberek Türkçe tokenizer'ı ile sözcük, sayı ve kısaltmaları ayıklar, Türkçe
büyük/küçük harf dönüşümü uygular ve sözcükleri `TurkishStemmer` ile köklerine
indirir. Olumsuzluk bildiren “değil” gibi kelimeler özellikle korunur; genel
stop-word listesi uygulanmaz. Boş veya temizleme sonrası metinsiz yorumlar
atlanır ve sonuç terminalde raporlanır.

`rating` değerinden `sentiment` etiketi türetilir: 1–2 yıldız `olumsuz`,
3 yıldız `notr`, 4–5 yıldız `olumlu`.

Modelde özellik olarak yalnızca `review_clean`, hedef olarak `sentiment`
kullanın. `rating` etiketin kaynağı olduğundan model girdisine eklenmemelidir.
Başlık da girdiye dahil edilmez; bu veri kümesinde başlık puan metnini
tekrarlayabildiği için hedef bilgisini sızdırabilir. Veri kümesi küçük ve
olumlu yorumlara belirgin biçimde ağırlık verdiğinden, model performansını
değerlendirirken sınıf dengesizliğini ve örnek sayısını göz önünde bulundurun.

Ön işleme testleri:

```powershell
py -m unittest -v .\test_veri_on_isleme.py
```

## Sınıflandırma modelleri

Naive Bayes, lojistik regresyon, Support Vector Machine, Random Forest ve
Türkçe BERT modellerini aynı stratified eğitim/test bölmesinde eğitip doğruluk
puanlarını hesaplamak için:

```powershell
py -m pip install -r .\requirements-ml.txt
py .\siniflandirma.py
```

Türkçe BERT için `dbmdz/bert-base-turkish-cased` önceden eğitilmiş modeli
indirilir. Klasik TF-IDF modelleri `review_clean` sütununu; BERT ise önceden
eğitildiği yüzey biçimlerini korumak için ham `review` metnini kullanır.
Bütün modeller aynı sabit tohumla oluşturulan stratified %80 eğitim / %20 test
bölmesini ve puandan türetilen `sentiment` etiketini paylaşır. Her iki
doğruluk puanı terminale yazılır ve `siniflandirma_sonuclari.csv` dosyasına
kaydedilir.

Varsayılan BERT eğitimi 3 epoch sürer. Ayarlanabilir:

```powershell
py .\siniflandirma.py --epochs 5 --batch-size 8
```

Yalnızca klasik modelleri çalıştırmak için `--skip-bert` kullanılabilir.
Sonuçları değerlendirirken veri kümesinin küçük ve dengesiz olduğunu
(olumlu sınıfı baskın) göz önünde bulundurun; tek bir test bölmesindeki
doğruluk puanları genellenebilir performans garantisi değildir.

Bu veri kümesinde (78 eğitim / 20 test örneği, `random_state=42`) ölçülen
doğruluklar:

| Model | Eğitim doğruluğu | Test doğruluğu |
|---|---:|---:|
| Naive Bayes | %87,18 | %90,00 |
| Lojistik Regresyon | %100,00 | %90,00 |
| Support Vector Machine | %100,00 | %90,00 |
| Random Forest | %100,00 | %90,00 |
| Türkçe BERT (`dbmdz/bert-base-turkish-cased`) | %89,74 | %90,00 |

Test kümesindeki 20 yorumun 18'i olumlu olduğundan, her yorumu olumlu sayan
çoğunluk tabanı da %90 test doğruluğu alır. Bu nedenle bu deneydeki test
doğrulukları, modellerin çoğunluk tabanını aştığını göstermiyor; daha fazla ve
dengeli etiketli veriyle tekrar değerlendirme yapılmalı.
