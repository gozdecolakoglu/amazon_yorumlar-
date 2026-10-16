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

Ön işleme yalnızca metin ve yorum kimliğini çıktı CSV'sine taşır; bu aşamada
puanlardan duygu etiketi üretilmez. Duygu etiketleri sınıflandırma akışında
yorum metninden hazır Türkçe duygu modeliyle alınır.

Klasik modellerde özellik olarak `review_clean`, Türkçe BERT'te ham `review`
kullanılır. Başlık ve puan sütunları modele verilmez.

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

Önce `savasy/bert-base-turkish-sentiment-cased` modeli yorum metnini olumlu
veya olumsuz olarak puanlar. En yüksek sınıf olasılığı %60'ın altındaysa yorum
nötr etiketlenir; böylece 3 sınıfın tamamı yorum içeriği ve model güveninden
üretilir, yıldız puanları kullanılmaz. Etiketler ve güven değerleri
`iphone_yorumlar_duygu_etiketli.csv` dosyasına yazılır. Eşik değiştirilebilir:

```powershell
py .\siniflandirma.py --neutral-threshold 0.65
```

Klasik TF-IDF modelleri `review_clean` sütununu; incelenen Türkçe BERT modeli
ise büyük/küçük harf biçimini koruyan ham `review` metnini kullanır. Etiketleme
modeli ve sınıflandırıcılar yerel makinede çalışır; yorumlar bir hizmete
yüklenmez. Modeller aynı sabit tohumla oluşturulan stratified %80 eğitim /
%20 test bölmesini ve metinden üretilen `sentiment` etiketlerini paylaşır.
Doğruluklar terminale ve `siniflandirma_sonuclari.csv` dosyasına kaydedilir.

Varsayılan BERT eğitimi 3 epoch sürer. Ayarlanabilir:

```powershell
py .\siniflandirma.py --epochs 5 --batch-size 8
```

Yalnızca klasik modelleri çalıştırmak için `--skip-bert` kullanılabilir.
Sonuçları değerlendirirken veri kümesinin küçük ve dengesiz olduğunu
(nötr sınıfı az sayıda örnek içeriyor) göz önünde bulundurun; tek bir test bölmesindeki
doğruluk puanları genellenebilir performans garantisi değildir.

Bu veri kümesinde %60 eşikle hazır modelin ürettiği etiket dağılımı 48 olumlu,
44 olumsuz ve 8 nötr yorumdur. Ölçülen eğitim/test doğrulukları:

| Model | Eğitim doğruluğu | Test doğruluğu |
|---|---:|---:|
| Naive Bayes | %91,25 | %55,00 |
| Lojistik Regresyon | %100,00 | %75,00 |
| Support Vector Machine | %100,00 | %80,00 |
| Random Forest | %100,00 | %65,00 |
| Türkçe BERT (`dbmdz/bert-base-turkish-cased`) | %62,50 | %50,00 |

Önemli: Bu doğruluk puanları bağımsız insan etiketlerine göre değil, hazır
duygu modelinin ürettiği otomatik etiketlere göre hesaplanır. Yani modellerin
öğretmen modelle ne ölçüde uyuştuğunu gösterir; insan yargısına göre doğruluk
garantisi değildir. Testte yalnızca bir nötr örnek olduğundan üç sınıf
başarımını bu sonuçlarla güvenilir biçimde değerlendirmek mümkün değildir.
Bilimsel değerlendirme için yorumlar insan tarafından etiketlenmeli ve
modeller bu bağımsız etiketlerle ölçülmelidir.
