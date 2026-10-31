"""Render the saved, sample-specific qualitative notes. Does not infer verdicts.

python src/render_assessment.py
The versioned review_notes.json contains one assessment per sampled span.
Changed samples require a new review, not automatic reuse of old decisions.
"""
from collections import Counter
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES={'possible_annotation_gap':'Olası eksik etiket / kapsam farkı',
            'model_error':'Model hatası adayı',
            'insufficient_context':'Bağlam yetersiz / kavram ayrımı belirsiz'}


def merge_notes(sample,notes):
    by_id={r['sample_id']:r for r in sample}
    if len(by_id)!=len(sample) or len(notes)!=len(sample):
        raise ValueError('Exactly one assessment per sampled span is required')
    if len({r['sample_id'] for r in notes})!=len(notes):
        raise ValueError('Duplicate assessment ID')
    if {r['sample_id'] for r in notes} != set(by_id):
        raise ValueError('Assessment IDs do not match the sample')
    merged=[]
    for note in notes:
        item=by_id[note['sample_id']]
        for key in ('test_row','start','end','predicted_type'):
            if note[key]!=item[key]:
                raise ValueError(f'Sample changed for {note["sample_id"]}: {key}')
        if note['category'] not in CATEGORIES or not note['rationale'].strip():
            raise ValueError('Valid category and rationale required')
        if note['cue'] not in item['sentence']:
            # Cues can be excerpts joined with ellipses; they are paraphrased cues,
            # not claimed verbatim quotes. Full original sentences are retained.
            pass
        merged.append({**item,**note})
    return merged


def main():
    out=ROOT/'results'/'assessment'
    sample=json.loads((out/'sample.json').read_text(encoding='utf-8'))
    notes=json.loads((out/'review_notes.json').read_text(encoding='utf-8'))
    meta=json.loads((out/'method.json').read_text(encoding='utf-8'))
    rows=merge_notes(sample,notes)
    counts=Counter(r['category'] for r in rows)
    variant_count=sum(len({tuple(m['target_tags']) for m in r['raw_matches']})>1 for r in rows)
    repeated=sum(len(r['raw_matches'])>1 for r in rows)
    train_matches=sum(any(m['split']=='train' for m in r['raw_matches']) for r in rows)
    csv_rows=[]
    for r in rows:
        span=[r['start'],r['end'],r['predicted_type']]
        csv_rows.append({k:r[k] for k in ('sample_id','test_row','start','end','target_text','predicted_type',
                                         'category','cue','rationale','follow_up','marked_sentence')})
        csv_rows[-1]['raw_matches']=len(r['raw_matches'])
        csv_rows[-1]['raw_target_variants']=len({tuple(m['target_tags']) for m in r['raw_matches']})
        csv_rows[-1]['models_agreeing_on_exact_span']=sum(span in ps for ps in r['model_spans'].values())
    pd.DataFrame(csv_rows).to_csv(out/'evaluation.csv',index=False,encoding='utf-8-sig')
    lines=['# Değerlendirme','',
           f'SEC-BERT-SHAPE modelinin ana test görünümündeki {meta["sample_population"]:,} '
           f'gold entity ile örtüşmeyen tahmininden, seed={meta["seed"]} ile geri koymadan seçilen '
           f'{len(rows)} örnek incelendi. Örnekleme birimi tahmin edilen entity spanıdır; '
           f'bu örneklem {meta["distinct_sample_sentences"]} ayrı cümle içeriyor. '
           'Yanlış tür ve sınır hataları bu örnekleme evrenine dahil değil.','',
           '## Sonuç','', '| Karar | Örnek sayısı |','|---|---:|']
    for category,label in CATEGORIES.items():
        lines.append(f'| {label} | {counts[category]} |')
    lines += ['',
              'Bu kararlar cümle bağlamı, gold etiketler, model tahminleri ve aynı cümlenin ham verideki '
              'kopyalarının incelenmesine dayanıyor. Tek değerlendirmedir; bağımsız ikinci değerlendirme '
              've değerlendiriciler arası uyum ölçümü yapılmadı. Kaynak SEC raporundaki XBRL fact ve '
              'taksonomi kapsamı her örnek için doğrulanmadığından kararlar ön bulgu niteliğindedir.', '',
              '**Olası eksik etiket**, modelin kesin doğru veya veri etiketinin kesin yanlış olduğu anlamına gelmez. '
              'Özel şirket etiketleri, 139 sınıflık kapsam, raporlayan işletme sınırı veya fact/dimension '
              'seçimi de O etiketini açıklayabilir. Kategori sayıları tüm veri setinin hata oranı değildir.', '',
              '## Ham veri kontrolü', '',
              f'- {repeated} örneğin cümlesi ham veride birden fazla kez bulunuyor.',
              f'- {train_matches} örnek için ham eğitim verisinde de aynı cümle bulunuyor. '
              'Bu ham kopyalar temizlenmiş eğitim setinde sızıntı olduğu anlamına gelmez; temizlik bunları çıkarmıştı.',
              f'- Seçilen spanın ham kopyaları arasında farklı hedef etiketi bulunan örnek sayısı: **{variant_count}**.',
              '- Bu kontrol yalnız birebir aynı token dizilerini kapsar; benzer cümleleri veya dış kaynak raporları kapsamaz.',
              '- Aynı O etiketinin tekrar edilmesi doğruluğunu kanıtlamaz. Modellerin aynı tahmini yapması da bağımsız doğruluk kanıtı değildir.', '',
              '## Öne çıkan örnekler', '',
              '- Q019 ve Q037: LIBOR üzerine eklenen faiz spreadi cümlede açıkça ifade ediliyor; gold O. '
              'Kaynak etiket kapsamının incelenmesi için güçlü adaylar.',
              '- Q027: Pay geri alımında pay sayısı ve toplam maliyet etiketli, ortalama birim fiyat O. '
              'Aynı cümledeki etiket kapsamı farkını gösteriyor.',
              '- Q030: %50, belirli bir yatırımın payı değil, muhasebe politikasının eşik aralığı. '
              'Bağlamı aşan etiketleme için model hatası adayı.',
              '- Q034: Tutar goodwill ve diğer intangible assets toplamı; yalnız Goodwill tahmini kapsamı daraltıyor.',
              '- Q001: Vade tarihi bağlamı var, ancak model yalnız July kelimesini span olarak seçiyor. '
              'Olası etiket eksikliği ile tahminin kendi sınır problemi aynı örnekte birlikte bulunabilir.', '',
              '## Tüm örnekler', '',
              '[[ ... ]] işaretleri değerlendirilen spanı gösterir. Satır numarası cleaned/test.parquet içinde '
              'sıfır tabanlıdır. İpucu satırı kısaltılmış bağlamsal özet olabilir; tam özgün cümle aşağıdadır.', '']
    for r in rows:
        lines += [f'### {r["sample_id"]} — {CATEGORIES[r["category"]]}','',
                  f'- Test satırı: {r["test_row"]}; span: [{r["start"]}, {r["end"]}).',
                  f'- Model etiketi: `{r["predicted_type"]}`; seçili spanın gold etiketi: `O`.',
                  f'- İpucu: {r["cue"]}', '',r['marked_sentence'],'',r['rationale'],'',
                  'Doğrulama adımı: '+r['follow_up'],'']
    lines += ['## Kapsam ve tekrar üretim', '',
              'Ana test etiketleri ve skorlar değiştirilmedi. Bu değerlendirmeden düzeltilmiş bir F1 üretilmedi. '
              'Test bulguları üzerinden yeni bir model seçimi yapılırsa bağımsız değerlendirme gerekir.', '',
              '`python src/assess_results.py --replicates 10000 --sample-size 50 --seed 42` örnekleri ve bootstrap '
              'sonuçlarını yeniden üretir. `python src/render_assessment.py` saklanan örnek-bazlı notlardan bu '
              'raporu oluşturur. Notlar otomatik sınıflandırma kuralı değildir; örneklem değişirse yeniden değerlendirme gerekir.', '',
              'FiNER makalesi de eksik/tutarsız kaynak etiketlerine örnek verir; bu, buradaki adayların tek tek '
              'doğrulandığı anlamına gelmez. [Loukas et al., 2022, Bölüm 9.4](https://aclanthology.org/2022.acl-long.303.pdf)', '']
    (out/'DEGERLENDIRME.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'evaluation_summary.json').write_text(json.dumps(dict(sample_size=len(rows),categories=dict(counts),
        repeated_raw_sentences=repeated,raw_train_matches=train_matches,raw_target_conflicts=variant_count,
        review_passes=1,independent_second_review=False,gold_labels_changed=False),indent=2),encoding='utf-8')
    print(dict(counts), 'raw target conflicts:',variant_count)


if __name__=='__main__':
    main()
