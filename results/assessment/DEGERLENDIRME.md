# Değerlendirme

SEC-BERT-SHAPE modelinin ana test görünümündeki 8,222 gold entity ile örtüşmeyen tahmininden, seed=42 ile geri koymadan seçilen 50 örnek incelendi. Örnekleme birimi tahmin edilen entity spanıdır; bu örneklem 50 ayrı cümle içeriyor. Yanlış tür ve sınır hataları bu örnekleme evrenine dahil değil.

## Sonuç

| Karar | Örnek sayısı |
|---|---:|
| Olası eksik etiket / kapsam farkı | 28 |
| Model hatası adayı | 5 |
| Bağlam yetersiz / kavram ayrımı belirsiz | 17 |

Bu kararlar cümle bağlamı, gold etiketler, model tahminleri ve aynı cümlenin ham verideki kopyalarının incelenmesine dayanıyor. Tek değerlendirmedir; bağımsız ikinci değerlendirme ve değerlendiriciler arası uyum ölçümü yapılmadı. Kaynak SEC raporundaki XBRL fact ve taksonomi kapsamı her örnek için doğrulanmadığından kararlar ön bulgu niteliğindedir.

**Olası eksik etiket**, modelin kesin doğru veya veri etiketinin kesin yanlış olduğu anlamına gelmez. Özel şirket etiketleri, 139 sınıflık kapsam, raporlayan işletme sınırı veya fact/dimension seçimi de O etiketini açıklayabilir. Kategori sayıları tüm veri setinin hata oranı değildir.

## Ham veri kontrolü

- 17 örneğin cümlesi ham veride birden fazla kez bulunuyor.
- 10 örnek için ham eğitim verisinde de aynı cümle bulunuyor. Bu ham kopyalar temizlenmiş eğitim setinde sızıntı olduğu anlamına gelmez; temizlik bunları çıkarmıştı.
- Seçilen spanın ham kopyaları arasında farklı hedef etiketi bulunan örnek sayısı: **0**.
- Bu kontrol yalnız birebir aynı token dizilerini kapsar; benzer cümleleri veya dış kaynak raporları kapsamaz.
- Aynı O etiketinin tekrar edilmesi doğruluğunu kanıtlamaz. Modellerin aynı tahmini yapması da bağımsız doğruluk kanıtı değildir.

## Öne çıkan örnekler

- Q019 ve Q037: LIBOR üzerine eklenen faiz spreadi cümlede açıkça ifade ediliyor; gold O. Kaynak etiket kapsamının incelenmesi için güçlü adaylar.
- Q027: Pay geri alımında pay sayısı ve toplam maliyet etiketli, ortalama birim fiyat O. Aynı cümledeki etiket kapsamı farkını gösteriyor.
- Q030: %50, belirli bir yatırımın payı değil, muhasebe politikasının eşik aralığı. Bağlamı aşan etiketleme için model hatası adayı.
- Q034: Tutar goodwill ve diğer intangible assets toplamı; yalnız Goodwill tahmini kapsamı daraltıyor.
- Q001: Vade tarihi bağlamı var, ancak model yalnız July kelimesini span olarak seçiyor. Olası etiket eksikliği ile tahminin kendi sınır problemi aynı örnekte birlikte bulunabilir.

## Tüm örnekler

[[ ... ]] işaretleri değerlendirilen spanı gösterir. Satır numarası cleaned/test.parquet içinde sıfır tabanlıdır. İpucu satırı kısaltılmış bağlamsal özet olabilir; tam özgün cümle aşağıdadır.

### Q001 — Model hatası adayı

- Test satırı: 43390; span: [30, 31).
- Model etiketi: `DebtInstrumentMaturityDate`; seçili spanın gold etiketi: `O`.
- İpucu: due July 15, 2026

Senior Notes The Company has outstanding 5.375 % senior unsecured notes due December 15 , 2024 ( " 2024 Senior Notes " ) and 5.000 % senior unsecured notes due [[ July ]] 15 , 2026 ( " 2026 Senior Notes " and together with the 2024 Senior Notes , the " Senior Unsecured Notes " ) .

Tahmin yalnızca July kelimesini kapsıyor; cümledeki vade tarihi July 15, 2026 ifadesinin tamamı. Tarihin gold etikette bulunmaması ayrıca incelenebilir, ancak mevcut tek kelimelik tahmin tam tarihi yakalamıyor.

Doğrulama adımı: Kaynak belgedeki tarih span sınırlarını ve etiket kapsamını kontrol et.

### Q002 — Olası eksik etiket / kapsam farkı

- Test satırı: 53171; span: [33, 34).
- Model etiketi: `TreasuryStockSharesAcquired`; seçili spanın gold etiketi: `O`.
- İpucu: acquired 3,476 shares ... shares surrendered by employees

The following table shows shares purchased and value ( $ in millions ) under the respective stock purchase plans : For the three months ended March 31 , 2020 , the Company acquired [[ 3,476 ]] shares of Common Stock , at a cost of $ 0.2 million for shares surrendered by employees to pay taxes due on vested restricted stock awards .

3,476 açıkça şirketin çalışanlardan edindiği pay sayısı; tahmin bu anlama uygun görünüyor, gold ise O. Vergi kesintisi için geri alınan payların ilgili taksonomi kapsamına girip girmediği kaynak belgeden doğrulanmalı.

Doğrulama adımı: Çalışan pay geri alımı ile treasury stock etiketinin kapsamını kaynak XBRL'de doğrula.

### Q003 — Olası eksik etiket / kapsam farkı

- Test satırı: 47284; span: [14, 15).
- Model etiketi: `NumberOfRealEstateProperties`; seçili spanın gold etiketi: `O`.
- İpucu: seven properties ... included in other real estate owned

Foreclosure Proceedings and Balances : As of March 31 , 2020 , there were [[ seven ]] properties secured by real estate included in other real estate owned and there were no consumer mortgage loans collateralized by residential real estate property that were in the process of foreclosure .

seven kelimesi doğrudan elde tutulan gayrimenkul sayısını belirtiyor. NumberOfRealEstateProperties tahmini cümleyle uyumlu; gold O olduğu için olası eksik etiket adayı.

Doğrulama adımı: Kaynak gayrimenkul dipnotunda sayının hangi XBRL kavramıyla raporlandığını kontrol et.

### Q004 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 10150; span: [9, 10).
- Model etiketi: `SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage`; seçili spanın gold etiketi: `O`.
- İpucu: PSUs earned may range from 0% - 150%

Actual PSUs earned may range from 0 % - [[ 150 ]] % of the PSUs allocated to the award recipient , based on the Company 's total shareholder return ( " TSR " ) compared to a peer group based on companies with similar assets and revenue over a three - year performance period that commenced on the respective Grant Date ( as referenced below ) .

150 oranı performansa bağlı kazanılabilecek birim miktarının üst sınırı. Bunun vesting rights percentage mı yoksa ayrı bir payout/performance oranı mı olduğu yalnız bu cümleden kesinleşmiyor.

Doğrulama adımı: PSU sözleşmesi ve ilgili etiketin tanımıyla vesting/payout ayrımını doğrula.

### Q005 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 78691; span: [47, 48).
- Model etiketi: `TreasuryStockAcquiredAverageCostPerShare`; seçili spanın gold etiketi: `O`.
- İpucu: weighted-average market purchase price ... 14.93 per common share

During the first three months of 2020 and 2019 , the Company purchased , under its Share Repurchase Program , approximately 1.5 million and 0.3 million of its common shares , respectively , at a weighted - average market purchase price of approximately $ 9.42 and $ [[ 14.93 ]] per common share , respectively , for an aggregate purchase price , including commissions , of approximately $ 14.3 million and $ 4.1 million , respectively .

14.93 bir geri alımın ortalama pay başı fiyatı. Ancak gold aynı cümledeki payları repurchased and retired olarak etiketliyor; treasury stock için maliyet etiketiyle iptal edilen payların fiyatı arasındaki kapsam ayrımı belirsiz.

Doğrulama adımı: Payların treasury stock olarak tutulup tutulmadığını ve emeklilik/iptal muhasebesini kaynak belgeden kontrol et.

### Q006 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 92818; span: [42, 43).
- Model etiketi: `EquityMethodInvestmentOwnershipPercentage`; seçili spanın gold etiketi: `O`.
- İpucu: 70% ownership ... remaining 30% ownership held by Nanjing

In March 2018 , Xianning Xiangtian formed Xiangtian Zhongdian ( Hubei ) New Energy Co. Ltd. ( “ Xiangtian Zhongdian ” ) , a joint venture in China , in which Xianning Xiangtian holds a 70 % ownership interest with the remaining [[ 30 ]] % ownership held by Nanjing Zhongdian Photovoltaic Co. Ltd. Xiangtian Zhongdian is in the business of manufacturing and sales of PV panels In April 2018 , Xianning Xiangtian formed a wholly owned subsidiary , Jingshan Sanhe Xiangtian New Energy Technology Co. Ltd. ( “ Jingshan Sanhe ” ) , which is engaged in the business of researching , manufacturing and sales of high - grade synthetic fuel products .

30 oranı diğer ortağın payını gösteriyor. Equity method yatırımı mı, konsolide ortaklıktaki kontrol gücü olmayan pay mı olduğu ve raporlayan tarafın perspektifi cümleden kesin belirlenemiyor.

Doğrulama adımı: Ortaklığın muhasebeleştirme yöntemi ve raporlayan işletme sınırını kontrol et.

### Q007 — Model hatası adayı

- Test satırı: 65823; span: [13, 14).
- Model etiketi: `AllocatedShareBasedCompensationExpense`; seçili spanın gold etiketi: `O`.
- İpucu: would have resulted in a decrease of $0.1 million ... if ... had not been changed

The impact of this change would have resulted in a decrease of $ [[ 0.1 ]] million to stock compensation expense for the period if the forfeiture methodology had not been changed .

0.1 gerçekleşmiş dönem hisse bazlı ücret gideri değil, farklı yöntem kullanılsaydı giderde oluşacak varsayımsal azalış. Model bir değişim/senaryo tutarını doğrudan gider tutarı gibi etiketlemiş görünüyor.

Doğrulama adımı: Gerçekleşen gider ile varsayımsal farkın taksonomi kapsamını kontrol et.

### Q008 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 71486; span: [64, 65).
- Model etiketi: `BusinessCombinationConsiderationTransferred1`; seçili spanın gold etiketi: `O`.
- İpucu: acquisition of the remaining ... common units ... became a wholly owned subsidiary

Following the Initial Unit Purchase Closing , on December 31 , 2018 , the Company exercised a limited call right under EQGP 's partnership agreement pursuant to which , on January 10 , 2019 , the Company closed on the acquisition of the remaining 11,097,287 outstanding EQGP common units not owned by the Company or its affiliates for an aggregate purchase price of $ [[ 221.9 ]] million ( such acquisition , together with the EQGP Unit Purchases , the EQGP Buyout ) , and EQGP became a wholly owned subsidiary of the Company .

221.9 kalan payların satın alınma bedeli. İşlem öncesinde kontrol varsa bu satın alma yeni bir business combination olmayabilir; kontrol durumu verilmediğinden model etiketinin doğruluğu belirsiz.

Doğrulama adımı: İşlem öncesi kontrol ve işletme birleşmesi muhasebesini incele.

### Q009 — Olası eksik etiket / kapsam farkı

- Test satırı: 29202; span: [100, 101).
- Model etiketi: `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized`; seçili spanın gold etiketi: `O`.
- İpucu: total number of shares issued under the ESPP may not exceed 14,000,000

Under the ESPP , the Company initially reserved for issuance 1,400,000 shares of common stock , which will increase automatically on July 1 of each fiscal year during the term of the ESPP by the number of shares equal to 1 % of the total number of shares of common stock and preferred stock ( on as - converted basis ) outstanding as of the immediately preceding June 30th , unless the board of directors elects to authorize a lesser number of shares ; provided , that , the total number of shares issued under the ESPP may not exceed [[ 14,000,000 ]] shares of common stock .

14,000,000 çalışan pay alım planının açık toplam ihraç sınırı. Plan kapsamında izin verilen pay sayısı etiketiyle anlamlı biçimde eşleşiyor; gold etiketi yok.

Doğrulama adımı: Planın toplam sınırı ile halen kullanılabilir yetki miktarını ayırarak kaynak etiketi kontrol et.

### Q010 — Olası eksik etiket / kapsam farkı

- Test satırı: 46777; span: [4, 6).
- Model etiketi: `LesseeOperatingLeaseRenewalTerm`; seçili spanın gold etiketi: `O`.
- İpucu: Lease extensions were for five years each

Lease extensions were for [[ five years ]] each and the Company reassessed the ROU asset and lease liability related to these leases .

five years kira uzatmalarının süresini açıkça ifade ediyor. Yenileme süresi tahmini bağlamla uyumlu, ancak operating lease sınıflaması ve kaynak etiketleme kapsamı doğrulanmalı.

Doğrulama adımı: İlgili kiraların operating lease olduğunu ve süre etiketini kaynakta doğrula.

### Q011 — Olası eksik etiket / kapsam farkı

- Test satırı: 38598; span: [24, 25).
- Model etiketi: `AssetImpairmentCharges`; seçili spanın gold etiketi: `O`.
- İpucu: $16.9 million for the impairments of trademarks

Of this total , $ 112.0 million was for impairments of goodwill , $ 42.2 million for impairments of gaming license rights and $ [[ 16.9 ]] million for the impairments of trademarks .

16.9 ticari markalardaki değer düşüklüğü tutarı olarak açıkça tanımlanmış. AssetImpairmentCharges anlamlı bir aday; aynı cümlede yalnız goodwill değer düşüklüğünün gold olması etiket kapsamının ayrıca incelenmesini gerektiriyor.

Doğrulama adımı: Daha özel intangible impairment etiketi veya özel şirket etiketi kullanılıp kullanılmadığını kontrol et.

### Q012 — Olası eksik etiket / kapsam farkı

- Test satırı: 69523; span: [12, 13).
- Model etiketi: `LineOfCreditFacilityMaximumBorrowingCapacity`; seçili spanın gold etiketi: `O`.
- İpucu: maximum amount ... could borrow under Facility A was €160.0 million

The maximum amount Duferdofin Nucor could borrow under Facility A was € [[ 160.0 ]] million ( $ 173.0 million ) at April 4 , 2020 .

160.0 Facility A kapsamında alınabilecek azami borç tutarı. MaximumBorrowingCapacity tahmini doğrudan ifade ile uyumlu ve gold O.

Doğrulama adımı: Facility A türünü ve kaynak XBRL kavramını doğrula.

### Q013 — Olası eksik etiket / kapsam farkı

- Test satırı: 45156; span: [43, 44).
- Model etiketi: `ContractWithCustomerLiability`; seçili spanın gold etiketi: `O`.
- İpucu: contract liabilities ... including $3,536 of deferred revenue related to extended warranties

As of December 31 , 2019 , the Company had contract assets of $ 4,660 , primarily related to billings on one large ( $ 7,249 ) order in the Infrastructure Solutions group , and contract liabilities of $ 6,511 , including $ [[ 3,536 ]] of deferred revenue related to extended warranties .

3,536 açıkça sözleşme yükümlülüklerinin ertelenmiş garanti geliri alt bileşeni. Aynı cümlede toplam yükümlülük etiketli olduğu halde bu tutar O; alt bileşenin ayrı etiketlenme kapsamı incelenmeli.

Doğrulama adımı: Garanti yükümlülüğü için daha özel etiket/dimension bulunup bulunmadığını kontrol et.

### Q014 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 7759; span: [33, 34).
- Model etiketi: `UnrecognizedTaxBenefits`; seçili spanın gold etiketi: `O`.
- İpucu: unrecognized income tax benefits and related interest and penalties ... $52.6 million

As of March 31 , 2020 and December 31 , 2019 , we have recorded a liability for unrecognized income tax benefits and related interest and penalties of $ 47.5 million and $ [[ 52.6 ]] million , respectively .

52.6 yalnız vergi faydasını değil ilgili faiz ve cezaları da içeren birleşik yükümlülük. UnrecognizedTaxBenefits etiketinin bu toplam için uygunluğu cümleden kesinleşmiyor.

Doğrulama adımı: Faiz/ceza hariç tutarı ve birleşik kavramın kaynak etiketini kontrol et.

### Q015 — Olası eksik etiket / kapsam farkı

- Test satırı: 79574; span: [29, 30).
- Model etiketi: `LesseeOperatingLeaseTermOfContract`; seçili spanın gold etiketi: `O`.
- İpucu: operating leases ... remaining lease terms of 1 to 30 years

Operating Leases - Our operating leases , for network assets and equipment , office space , office equipment and real estate , have remaining lease terms of 1 to [[ 30 ]] years , some of which may include one or more options to renew with renewal terms that can extend the lease term from 1 to 10 years or more .

30 mevcut operating lease sözleşmelerinin kalan süresinin üst sınırı. Aynı cümlede yenileme süreleri gold etiketli, mevcut sözleşme süresi O; tutarlı etiketleme açısından inceleme adayı.

Doğrulama adımı: Kalan süre ile toplam sözleşme süresi için etiket tanımı ve kaynak uygulamasını kontrol et.

### Q016 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 54542; span: [15, 16).
- Model etiketi: `RevenueFromContractWithCustomerExcludingAssessedTax`; seçili spanın gold etiketi: `O`.
- İpucu: $5.0 million ... intercompany sales ... Eliminations and Other

( 2 ) Included in the Technical Nonwovens segment and Eliminations and Other is $ [[ 5.0 ]] million and $ 4.7 million in intercompany sales to the Thermal Acoustical Solutions segment for the quarters ended March 31 , 2020 and 2019 , respectively .

5.0 grup içi segment satışını ifade ediyor. Dış müşterilerle sözleşmelerden gelir etiketiyle ilişki; konsolidasyon eliminasyonları, segment boyutları ve kavram kapsamı görülmeden belirlenemiyor.

Doğrulama adımı: Intersegment/intercompany revenue kavramı ve konsolidasyon kapsamını kontrol et.

### Q017 — Olası eksik etiket / kapsam farkı

- Test satırı: 52299; span: [23, 24).
- Model etiketi: `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized`; seçili spanın gold etiketi: `O`.
- İpucu: total unrecognized stock-based compensation expense ... $1.0 million

As of March 31 , 2020 , total unrecognized stock - based compensation expense related to non - vested employee awards was $ [[ 1.0 ]] million which is expected to be recognized over a weighted average period of approximately 2.05 years . Restricted Stock Awards The Restated Plan permits the award of restricted stock awards ( “ RSA ’ s ” ) .

1.0 henüz muhasebeleştirilmemiş, hak edilmemiş çalışan ödüllerine ilişkin toplam gider. Tahminle açıkça uyumlu; aynı cümlede muhasebeleştirme süresi gold etiketli iken tutar O.

Doğrulama adımı: Toplam unrecognized compensation cost için kaynak XBRL etiketini kontrol et.

### Q018 — Olası eksik etiket / kapsam farkı

- Test satırı: 76725; span: [9, 10).
- Model etiketi: `DebtInstrumentFaceAmount`; seçili spanın gold etiketi: `O`.
- İpucu: obtained an $8.4 million loan under the PPP

In April 2020 , the Company obtained an $ [[ 8.4 ]] million loan under the PPP .

8.4 alınan kredinin tutarı; borç aracının anaparası etiketi için anlamlı aday. Kaynakta özel PPP kavramı veya başka borç kavramı kullanımı bilinmiyor.

Doğrulama adımı: PPP kredisi tutarının kaynak kavramını ve face/carrying amount ayrımını kontrol et.

### Q019 — Olası eksik etiket / kapsam farkı

- Test satırı: 19982; span: [17, 18).
- Model etiketi: `DebtInstrumentBasisSpreadOnVariableRate1`; seçili spanın gold etiketi: `O`.
- İpucu: variable rate ... LIBOR plus 1.75%

The $ 425 million Term Loan bears interest at a variable rate , which is LIBOR plus [[ 1.75 ]] % .

1.75 değişken faiz oranının referans üzerine eklenen spread kısmı. Tahmin tam bu anlamı taşıyor; gold etiket yok.

Doğrulama adımı: Kaynak kredinin spread etiketini doğrula.

### Q020 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 83240; span: [0, 1).
- Model etiketi: `CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption`; seçili spanın gold etiketi: `O`.
- İpucu: No cumulative effect adjustment ... upon adoption

[[ No ]] cumulative effect adjustment to retained earnings was recognized upon adoption of the new standard .

No kelimesi parasal etki bulunmadığını belirtiyor. Bu veri setinde metinsel yokluk ifadesinin sıfır değerli finansal fact gibi etiketlenip etiketlenmeyeceği cümleden belirlenemez.

Doğrulama adımı: Sıfır/No/nil ifadeleri için veri üretim ve kaynak etiketleme kuralını kontrol et.

### Q021 — Olası eksik etiket / kapsam farkı

- Test satırı: 57253; span: [5, 6).
- Model etiketi: `StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1`; seçili spanın gold etiketi: `O`.
- İpucu: authorized $50 million remaining under our current stock repurchase program

We have an authorized $ [[ 50 ]] million remaining under our current stock repurchase program for any additional repurchases .

50 mevcut geri alım programında kalan yetkilendirilmiş tutar. Model etiketi cümlede doğrudan destekleniyor, gold O.

Doğrulama adımı: Kalan geri alım yetkisinin kaynak etiketini doğrula.

### Q022 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 82058; span: [25, 26).
- Model etiketi: `BusinessCombinationContingentConsiderationLiability`; seçili spanın gold etiketi: `O`.
- İpucu: contingent liability ... acquisition ... former owners' future period of employment

The $ 0.5 million was paid to the seller in July 2018 . We had a contingent liability at March 31 , 2018 for $ [[ 4.9 ]] million related to the acquisition of Simena in November 2011 , which was based on the ultimate settlement of assets and liabilities acquired as part of the transaction , and the former owners ' future period of employment with us .

4.9 satın alma ile ilişkili koşullu yükümlülük; ancak eski sahiplerin gelecekteki istihdam koşulu da belirtiliyor. Bu, satın alma bedeli ile sonraki hizmete ilişkin ödeme ayrımını belirsiz bırakıyor.

Doğrulama adımı: Koşullu ödeme sözleşmesini ve ilgili muhasebe sınıflamasını kontrol et.

### Q023 — Olası eksik etiket / kapsam farkı

- Test satırı: 16953; span: [3, 5).
- Model etiketi: `SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod`; seçili spanın gold etiketi: `O`.
- İpucu: All options expire ten years after the date of grant

All options expire [[ ten years ]] after the date of grant .

ten years opsiyonların verilişten itibaren sona erme süresi. Tahmin semantik olarak açık eşleşme sunuyor; gold O.

Doğrulama adımı: Opsiyonların ücret planı kapsamını ve süre fact'inin kaynak etiketini doğrula.

### Q024 — Model hatası adayı

- Test satırı: 70689; span: [14, 15).
- Model etiketi: `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod`; seçili spanın gold etiketi: `O`.
- İpucu: Each subsequent independent director ... receives 3,333 shares ... upon election

Each subsequent independent director that joins the Company ’ s board of directors receives [[ 3,333 ]] shares of restricted common stock upon election to the Company ’ s board of directors .

3,333 her yeni yönetici için geçerli ödül politikasındaki miktar. Cümle raporlama döneminde fiilen verilen toplam pay sayısını belirtmiyor; GrantsInPeriod tahmini politika miktarını dönem hareketiyle karıştırıyor görünüyor.

Doğrulama adımı: Dönem içi fiili ödül tablosuyla politika açıklamasını ayır.

### Q025 — Olası eksik etiket / kapsam farkı

- Test satırı: 86883; span: [13, 14).
- Model etiketi: `RestructuringCharges`; seçili spanın gold etiketi: `O`.
- İpucu: we recorded restructuring charges of $37.6 million

During Fiscal 2020 and Fiscal 2019 , we recorded restructuring charges of $ [[ 37.6 ]] million and $ 93.6 million , respectively , in connection with our restructuring plans , primarily consisting of severance and benefits costs , as well as a loss on sale of property during the prior fiscal year period .

37.6 dönem içinde kaydedilen yeniden yapılandırma gideri olarak açıkça sunuluyor. RestructuringCharges tahminiyle uyumlu, gold O.

Doğrulama adımı: Kaynak yeniden yapılandırma dipnotundaki gider fact'ini kontrol et.

### Q026 — Olası eksik etiket / kapsam farkı

- Test satırı: 87044; span: [72, 73).
- Model etiketi: `DebtInstrumentInterestRateStatedPercentage`; seçili spanın gold etiketi: `O`.
- İpucu: fixed-rate 2.625% Senior Notes obligation

Dollar - denominated variable interest rate payments based on the 3 - month LIBOR plus a fixed spread ( as paid under the 2.625 % Interest Rate Swap discussed below ) for Euro - denominated variable interest rate payments based on the 3 - month EURIBOR plus a fixed spread , which , in combination with the 2.625 % Interest Rate Swap , economically converts our $ 300 million fixed - rate [[ 2.625 ]] % Senior Notes obligation to a € 274 million floating - rate Euro - denominated obligation . Additionally , in August 2018 , we entered into pay - fixed rate , receive - fixed rate cross - currency swap contracts with an aggregate notional amount of € 346 million which we designated as hedges of our net investment in certain of our European subsidiaries .

Seçilen 2.625 ifadesi swapın başka tekrarından değil, doğrudan sabit faizli tahvil yükümlülüğünün kuponundan geliyor. DebtInstrumentInterestRateStatedPercentage tahmini bu span için anlamlı.

Doğrulama adımı: Seçilen spanın tahvil kuponu olarak kaynakta etiketlendiğini kontrol et.

### Q027 — Olası eksik etiket / kapsam farkı

- Test satırı: 45582; span: [27, 28).
- Model etiketi: `TreasuryStockAcquiredAverageCostPerShare`; seçili spanın gold etiketi: `O`.
- İpucu: repurchased ... at an average price of $22.34 per share

During the three months ended March 31 , 2020 , the Company repurchased 38,411 shares of the Company 's common stock at an average price of $ [[ 22.34 ]] per share and a total cost of $ 858 thousand under the repurchase plan .

22.34 geri alınan payların ortalama birim fiyatı. Aynı cümlede pay sayısı ve toplam maliyet gold etiketli; birim fiyatın O olması eksik etiket/kapsam farkı adayı.

Doğrulama adımı: Ortalama maliyet fact'inin kaynak etiketini kontrol et.

### Q028 — Olası eksik etiket / kapsam farkı

- Test satırı: 10489; span: [9, 10).
- Model etiketi: `StockIssuedDuringPeriodSharesNewIssues`; seçili spanın gold etiketi: `O`.
- İpucu: Company issued 2,728,513 shares ... upon exercise ... warrant

On October 14 , 2019 , the Company issued [[ 2,728,513 ]] shares of Common Stock to CVI upon exercise by CVI of the Series C warrant .

2,728,513 warrant kullanımı üzerine ihraç edilen pay sayısı. Yeni pay ihracı etiketi bağlamla uyumlu; daha özel warrant egzersiz kavramı olup olmadığı bilinmiyor.

Doğrulama adımı: Warrant kullanımı ve yeni pay ihracı için kaynak etiket/dimension ilişkisini kontrol et.

### Q029 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 84338; span: [8, 9).
- Model etiketi: `RelatedPartyTransactionAmountsOfTransaction`; seçili spanın gold etiketi: `O`.
- İpucu: Texas Lehigh purchased ... $28.4 million ... from HCC

Texas Lehigh purchased $ 29.8 million , $ [[ 28.4 ]] million , and $ 21.3 million of cement from HCC in 2019 , 2018 , and 2017 , respectively , $ 1.0 million of slag from Skyway Cement Company in 2019 , and $ 0.3 million of cement from Central Plains Cement Company in 2017 .

Satın alma tutarı açık; fakat şirketler arasındaki ilişkinin related party niteliği bu cümlede açıklanmıyor. Sadece şirket adlarıyla ilişkili taraf etiketi doğrulanamaz.

Doğrulama adımı: İlişkili taraf açıklamasının önceki paragrafını ve mülkiyet bağını incele.

### Q030 — Model hatası adayı

- Test satırı: 89332; span: [22, 23).
- Model etiketi: `EquityMethodInvestmentOwnershipPercentage`; seçili spanın gold etiketi: `O`.
- İpucu: equity method ... where ... voting interest of 20% to 50%

The equity method of accounting is used for investments in companies where Universal Corporation has a voting interest of 20 % to [[ 50 ]] % .

50 belirli bir yatırımdaki gerçek pay oranı değil, equity method politikasını anlatan aralığın üst sınırı. Model genel muhasebe politikasını şirkete özgü yatırım payı gibi etiketlemiş görünüyor.

Doğrulama adımı: Yatırım bazında gerçek oran ile politika eşiğini ayrı değerlendir.

### Q031 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 64651; span: [31, 32).
- Model etiketi: `BusinessAcquisitionPercentageOfVotingInterestsAcquired`; seçili spanın gold etiketi: `O`.
- İpucu: previously held equity method investment ... price negotiated ... 47.5% equity interest

The Company considered multiple factors in determining the fair value of the previously held equity method investment , including ( i ) the price negotiated with the selling shareholder for the [[ 47.5 ]] % equity interest in Centa China , ( ii ) an income valuation model ( discounted cash flow ) , and ( iii ) current trading multiples for comparable companies .

47.5 değerleme bağlamındaki pay oranı. Bunun yeni edinilen oy hakkı mı, önceki yatırım mı olduğu ve ekonomik pay/oy hakkı eşitliği tek cümlede açık değil.

Doğrulama adımı: Satın alma öncesi/sonrası pay ve oy hakları tablosunu kontrol et.

### Q032 — Olası eksik etiket / kapsam farkı

- Test satırı: 66868; span: [16, 17).
- Model etiketi: `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized`; seçili spanın gold etiketi: `O`.
- İpucu: number of shares that may be issued pursuant to awards under the 2018 Plan is 1,596,753

The number of shares that may be issued pursuant to awards under the 2018 Plan is [[ 1,596,753 ]] . In 2010 , the Company adopted an equity compensation plan , or 2010 Plan , that provides for the grant of stock options to employees , directors , and other persons referred to in Rule 701 under the U.S. Securities Act of 1933 .

1,596,753 ödül planı kapsamında ihraç edilebilecek pay sayısı olarak açıkça verilmiş. Planın yetkilendirilmiş pay sayısı etiketi için olası eşleşme; toplam yetki/kalan kapasite ayrımı doğrulanmalı.

Doğrulama adımı: Planın başlangıç yetkisi ile kalan ihraç kapasitesi ayrımını kaynakta kontrol et.

### Q033 — Olası eksik etiket / kapsam farkı

- Test satırı: 74571; span: [19, 20).
- Model etiketi: `BusinessCombinationContingentConsiderationLiability`; seçili spanın gold etiketi: `O`.
- İpucu: recorded ... earn-out provision as contingent consideration of $77.3 million ... liabilities

The Company recorded the acquisition date fair value of the earn - out provision as contingent consideration of $ [[ 77.3 ]] million within “ Other long - term liabilities ” within the Company ’ s Condensed Consolidated Balance Sheets .

77.3 satın almadaki earn-out için kaydedilmiş koşullu bedel yükümlülüğü. Tahmin doğrudan cümleyle uyumlu, gold O.

Doğrulama adımı: İlgili contingent consideration fact'ini kaynak XBRL'de doğrula.

### Q034 — Model hatası adayı

- Test satırı: 47400; span: [71, 72).
- Model etiketi: `Goodwill`; seçili spanın gold etiketi: `O`.
- İpucu: $1.0 billion of goodwill and identifiable intangible assets

Moreover , expansion into any new industry or channel could result in higher compliance costs as we may become subject to laws and regulations to which we are not currently subject . Goodwill and other identifiable intangible assets represent a significant portion of our total assets , and we may never realize the full value of our intangible assets . As of March 31 , 2020 , we had approximately $ [[ 1.0 ]] billion of goodwill and identifiable intangible assets .

1.0 yalnız goodwill değil, goodwill ile diğer tanımlanabilir maddi olmayan duran varlıkların toplamı. Model bileşik toplamı yalnız Goodwill olarak etiketleyerek kapsamı daraltmış görünüyor.

Doğrulama adımı: Goodwill alt tutarını birleşik intangible assets toplamından ayır.

### Q035 — Olası eksik etiket / kapsam farkı

- Test satırı: 92859; span: [27, 28).
- Model etiketi: `FiniteLivedIntangibleAssetUsefulLife`; seçili spanın gold etiketi: `O`.
- İpucu: PV panel certifications ... finite useful lives of 10 years

Certain PV panel certifications were contributed by the Company ’ s noncontrolling interest shareholders as capital contribution in March 2018 with an estimated finite useful lives of [[ 10 ]] years .

10 sertifikasyon varlıklarının tahmini sonlu faydalı ömrü. Maddi olmayan varlığın faydalı ömrü etiketiyle uyumlu görünüyor; aktifleştirme sınıfı doğrulanmalı.

Doğrulama adımı: Sertifikasyon varlıklarının bilanço sınıfı ve faydalı ömür fact'ini kontrol et.

### Q036 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 73048; span: [120, 121).
- Model etiketi: `SaleOfStockPricePerShare`; seçili spanın gold etiketi: `O`.
- İpucu: $18.10 per share ($17.376 per share net of underwriting discounts and commissions)

December 2019 Offering On December 16 , 2019 , Brigham Minerals completed an offering of 12,650,000 shares of its Class A common stock ( the " December 2019 Offering " ) , including 6,000,000 shares issued and sold by Brigham Minerals and an aggregate of 6,650,000 shares sold by certain shareholders of the Company ( the " Selling Shareholders " ) , of which 5,496,813 represents shares issued upon redemption of an equivalent number of their common units in Brigham LLC ( the " Brigham LLC Units " ) ( together with a corresponding number of shares of Class B common stock in Brigham Minerals ) , at a price to the public of $ 18.10 per share ( $ [[ 17.376 ]] per share net of underwriting discounts and commissions ) .

17.376 kesintiler sonrası net pay başı tutar; aynı cümlede brüt halka arz fiyatı gold etiketli. SaleOfStockPricePerShare etiketinin net tutarı da kapsayıp kapsamadığı kaynak tanımına bağlı.

Doğrulama adımı: Brüt teklif fiyatı ile net proceeds per share kavramlarını karşılaştır.

### Q037 — Olası eksik etiket / kapsam farkı

- Test satırı: 7078; span: [44, 45).
- Model etiketi: `DebtInstrumentBasisSpreadOnVariableRate1`; seçili spanın gold etiketi: `O`.
- İpucu: floating rate of LIBOR + 2.85%

Deferred issuance costs of $ 0.6 million are being amortized on a straight - line basis over the term of the securities . Interest on the debentures issued to First Commonwealth Capital Trust II is paid quarterly at a floating rate of LIBOR + [[ 2.85 ]] % , which is reset quarterly .

2.85 değişken faizli borcun LIBOR üzerine eklenen spread oranı. Tahmin doğrudan destekleniyor; gold O.

Doğrulama adımı: Borç dipnotundaki spread fact'ini doğrula.

### Q038 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 76764; span: [24, 25).
- Model etiketi: `EquityMethodInvestmentOwnershipPercentage`; seçili spanın gold etiketi: `O`.
- İpucu: sole managing member ... 99.2% ... economic interest

PBF Energy Inc. ( “ PBF Energy ” ) is the sole managing member of , and owner of an equity interest representing approximately [[ 99.2 ]] % of the outstanding economic interest in , PBF LLC as of March 31 , 2020 .

99.2 ekonomik sahiplik payı. Cümle equity method uygulandığını göstermiyor; yönetici üyelik ve çok yüksek pay konsolidasyon ihtimalini de doğuruyor. Doğru etiket için muhasebe kapsamı gerekli.

Doğrulama adımı: PBF LLC konsolidasyonu ve economic/voting interest ayrımını kontrol et.

### Q039 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 46225; span: [15, 16).
- Model etiketi: `AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount`; seçili spanın gold etiketi: `O`.
- İpucu: notional units ... either ... contingency period ... or ... anti-dilutive

For the three months ended March 31 , 2020 and 2019 approximately 1.5 million and [[ 1.2 ]] million notional units were excluded from the computation , respectively , because these notional units either would not have been issuable if the end of the reporting period were the end of the contingency period or as they were anti - dilutive .

1.2 milyon birim iki farklı nedenle hariç tutulmuş olabilir: koşulların sağlanmaması veya anti-dilution. Tamamının antidilutive securities olarak etiketlenmesi bu birleşik açıklamadan doğrulanamıyor.

Doğrulama adımı: EPS dipnotunda nedenlere göre hariç bırakılan birimleri ayır.

### Q040 — Olası eksik etiket / kapsam farkı

- Test satırı: 84549; span: [36, 37).
- Model etiketi: `DebtInstrumentFaceAmount`; seçili spanın gold etiketi: `O`.
- İpucu: issued the 2023 Notes with an aggregate principal amount of $1.7 billion

Withholding taxes that would be payable upon remittance of these lower tier earnings are not expected to be material . In July 2018 , we issued the 2023 Notes with an aggregate principal amount of $ [[ 1.7 ]] billion .

1.7 ihraç edilen borcun toplam anaparası olarak açıkça tanımlanmış. DebtInstrumentFaceAmount için doğrudan bağlamsal destek var; gold O.

Doğrulama adımı: Tahvil anapara tutarının kaynak etiketini kontrol et.

### Q041 — Olası eksik etiket / kapsam farkı

- Test satırı: 56258; span: [28, 29).
- Model etiketi: `DebtInstrumentFaceAmount`; seçili spanın gold etiketi: `O`.
- İpucu: Company issued $500 million of unsecured 3.0% senior notes

See Note 14 for information on the Company ’ s fair value hedges . Long - term Debt On April 2 , 2020 , the Company issued $ [[ 500 ]] million of unsecured 3.0 % senior notes due April 2 , 2025 and incurred debt issuance costs of $ 4 million .

500 ihraç edilen tahvillerin tutarı; aynı cümlede kupon oranı gold etiketli. Face amount tahmini anlamlı ve tutarın etiket kapsamı incelenmeli.

Doğrulama adımı: Tahvil ihracı tutarını kaynak XBRL fact'iyle eşleştir.

### Q042 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 23438; span: [28, 29).
- Model etiketi: `RelatedPartyTransactionAmountsOfTransaction`; seçili spanın gold etiketi: `O`.
- İpucu: paid EPCO and its privately held affiliates cash distributions ... $296.0 million

During the three months ended March 31 , 2020 and 2019 , we paid EPCO and its privately held affiliates cash distributions totaling $ 302.8 million and $ [[ 296.0 ]] million , respectively .

296 nakit dağıtım tutarı. İsimler ve affiliates ifadesi tek başına bu tarafların raporlayan şirkete göre related party olduğunu kanıtlamıyor; ayrıca dağıtım türünün uygun etiketi belirsiz.

Doğrulama adımı: EPCO ile raporlayan şirket arasındaki kontrol ilişkisini ve dağıtım kavramını kontrol et.

### Q043 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 9800; span: [28, 29).
- Model etiketi: `RestructuringAndRelatedCostExpectedCost1`; seçili spanın gold etiketi: `O`.
- İpucu: expect ... restructuring charges ... severance benefits (between $8 million and $11 million)

We expect to incur restructuring charges totaling approximately $ 16 million to $ 22 million , consisting primarily of severance benefits ( between $ 8 million and $ [[ 11 ]] million ) and retention benefits ( between $ 8 million and $ 11 million ) primarily associated with the retention of employees impacted by the Spin - Off transaction and certain key research and development employees .

11 beklenen toplam yeniden yapılandırma giderinin kıdem/işten ayrılma faydası alt bileşenine ait üst sınır. Genel expected cost etiketi mi yoksa özel termination benefit etiketi mi gerektiği bu cümleden kesinleşmiyor.

Doğrulama adımı: Alt maliyet bileşeni ve boyutlandırılmış restructuring kavramlarını kontrol et.

### Q044 — Olası eksik etiket / kapsam farkı

- Test satırı: 78828; span: [124, 125).
- Model etiketi: `CommonStockParOrStatedValuePerShare`; seçili spanın gold etiketi: `O`.
- İpucu: Class B common stock, par value $0.01 per share

In the Entertainment Distribution , stockholders of Madison Square Garden Sports Corp. received ( a ) one share of the Company ’ s Class A common stock , par value $ 0.01 per share , for every share of Madison Square Garden Sports Corp. Class A common stock , par value $ 0.01 per share , held of record as of the close of business , New York City time , on April 13 , 2020 ( the “ Record Date ” ) , and ( b ) one share of the Company ’ s Class B common stock , par value $ 0.01 per share , for every share of Madison Square Garden Sports Corp. Class B common stock , par value $ [[ 0.01 ]] per share , held of record as of the close of business , New York City time , on the Record Date . Description of Business The Company is a leader in live experiences comprised of iconic venues ; marquee entertainment content ; popular dining and nightlife offerings ; and a premier music festival .

0.01 açıkça Class B payının nominal birim değeri. Etiket anlamsal olarak uyumlu; sayı başka şirketin payına ait olduğundan raporlama kapsamı ayrıca doğrulanmalı.

Doğrulama adımı: Söz konusu şirket/pay sınıfı için kaynak fact ve entity kapsamını kontrol et.

### Q045 — Olası eksik etiket / kapsam farkı

- Test satırı: 78108; span: [20, 21).
- Model etiketi: `IncomeTaxExpenseBenefit`; seçili spanın gold etiketi: `O`.
- İpucu: Income tax expense ... was $328,000

54 Income Tax Expense Income tax expense for the three months ended March 31 , 2020 and 2019 was $ [[ 328,000 ]] and $ 699,000 , respectively .

328,000 dönem gelir vergisi gideri olarak açıkça belirtilmiş. IncomeTaxExpenseBenefit etiketi bağlamla uyumlu; gold O.

Doğrulama adımı: Vergi giderinin kaynak XBRL fact'ini kontrol et.

### Q046 — Olası eksik etiket / kapsam farkı

- Test satırı: 18852; span: [14, 15).
- Model etiketi: `LineOfCreditFacilityMaximumBorrowingCapacity`; seçili spanın gold etiketi: `O`.
- İpucu: total borrowing capacity ... Credit Facility ... $1.4 billion

The total borrowing capacity of the 2020 Senior Unsecured Credit Facility is approximately $ [[ 1.4 ]] billion USD .

1.4 kredi tesisinin toplam borçlanma kapasitesi. MaximumBorrowingCapacity tahmini için doğrudan bağlamsal destek var.

Doğrulama adımı: Toplam taahhüt/azami kredi kapasitesi kavramını kaynakta doğrula.

### Q047 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 39760; span: [5, 6).
- Model etiketi: `NumberOfOperatingSegments`; seçili spanın gold etiketi: `O`.
- İpucu: report our operations in four business segments

We report our operations in [[ four ]] business segments - Gaming , Lottery , Sci Play and Digital . Basis of Presentation and Principles of Consolidation The accompanying condensed consolidated financial statements have been prepared in accordance with U.S. GAAP .

four segment sayısıdır; ancak operating segments ile reportable segments ayrımı cümlede açık değil. Modellerin de bu iki kavram arasında ayrışması doğruluk kanıtı değil, inceleme gereksinimidir.

Doğrulama adımı: Segment dipnotunda faaliyet segmenti ve raporlanabilir segment sayısını karşılaştır.

### Q048 — Olası eksik etiket / kapsam farkı

- Test satırı: 14716; span: [4, 5).
- Model etiketi: `DebtInstrumentTerm`; seçili spanın gold etiketi: `O`.
- İpucu: The loan has a 10-year term

The loan has a [[ 10 ]] - year term and bears interest on the principal amount outstanding at an annual rate of 1.00 % until April 23 , 2025 , and the applicable Secured Overnight Financing Rate ( " SOFR " ) plus 2.00 % thereafter until April 23 , 2030 .

10 doğrudan kredinin süresini ifade ediyor. DebtInstrumentTerm tahminiyle açık uyum var, gold O.

Doğrulama adımı: Kredi süresi için kaynak etiket ve birimi kontrol et.

### Q049 — Olası eksik etiket / kapsam farkı

- Test satırı: 10655; span: [11, 12).
- Model etiketi: `LineOfCredit`; seçili spanın gold etiketi: `O`.
- İpucu: Company had $298.1 million drawn ... under the Credit Agreement

As of March 31 , 2020 , the Company had $ [[ 298.1 ]] million drawn , letters of credit issued in the amount of $ 24.9 million , and $ 177.0 million of remaining borrowing capacity available under the Credit Agreement .

298.1 kredi anlaşmasından çekilmiş tutar. Aynı cümlede akreditif ve kalan kapasite gold etiketli; kullanılan kredi tutarının O olması inceleme adayı.

Doğrulama adımı: Kredi kullanım bakiyesinin LineOfCredit kavramıyla kaynak eşleşmesini kontrol et.

### Q050 — Bağlam yetersiz / kavram ayrımı belirsiz

- Test satırı: 82260; span: [43, 44).
- Model etiketi: `DebtInstrumentCarryingAmount`; seçili spanın gold etiketi: `O`.
- İpucu: unconsolidated joint ventures had total aggregate debt outstanding of $341.8 million

( 3 ) During the three months ended March 31 , 2020 , the Office Partnership acquired one property from a third party for a purchase price of $ 33.1 million . The unconsolidated joint ventures had total aggregate debt outstanding of $ [[ 341.8 ]] million as of March 31 , 2020 , which is non - recourse to the Company , as discussed in Note 6 - Debt .

341.8 konsolide edilmeyen ortak girişimlerin toplam borcu. Bu toplamın carrying amount olduğu ve raporlayan işletmenin kendi borç etiketiyle kapsamının nasıl belirlendiği cümleden kesinleşmiyor.

Doğrulama adımı: Ortak girişim kapsamını, borcun vadesini ve taşıma/anapara değeri ayrımını kontrol et.

## Kapsam ve tekrar üretim

Ana test etiketleri ve skorlar değiştirilmedi. Bu değerlendirmeden düzeltilmiş bir F1 üretilmedi. Test bulguları üzerinden yeni bir model seçimi yapılırsa bağımsız değerlendirme gerekir.

`python src/assess_results.py --replicates 10000 --sample-size 50 --seed 42` örnekleri ve bootstrap sonuçlarını yeniden üretir. `python src/render_assessment.py` saklanan örnek-bazlı notlardan bu raporu oluşturur. Notlar otomatik sınıflandırma kuralı değildir; örneklem değişirse yeniden değerlendirme gerekir.

FiNER makalesi de eksik/tutarsız kaynak etiketlerine örnek verir; bu, buradaki adayların tek tek doğrulandığı anlamına gelmez. [Loukas et al., 2022, Bölüm 9.4](https://aclanthology.org/2022.acl-long.303.pdf)
