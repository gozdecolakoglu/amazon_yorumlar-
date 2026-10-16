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
