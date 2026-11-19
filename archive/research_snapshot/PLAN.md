# Критически обоснованный план улучшения scratch-trained Music Source Separation к 23 ноября 2026

## Главное решение

**SINGLE BEST NEXT ACTION: не запускать Exp11/Exp12 сразу. Сначала провести короткий read-only аудит текущего pipeline, а затем первым контролируемым экспериментом сделать `cross-song stem remixing + conservative per-stem gain/EQ augmentation`, не меняя архитектуру, loss, число шагов и dev-panel.**

Причина довольно сильная. Exp10 почти побитово воспроизвела траекторию Exp09 при тех же данных и training recipe; следовательно, EMA/checkpoint selection не являлись главным ограничением. В то же время современная работа MERL специально исследует хронический дефицит размеченных MSS-данных и показывает, что случайное смешивание stems разных песен полезно даже несмотря на музыкально «неправильные» сочетания тональности и ритма. Их последующая реализация добавляет EQ augmentation и multiple sampling stems именно как улучшение этого подхода. citeturn434132view0turn620631search0

Это **не доказательство**, что augmentation даст прирост именно вашей модели. Но среди изменений, которые можно проверить на RTX 3080 Ti за один короткий pilot, оно имеет лучший баланс между научной поддержкой, стоимостью, информативностью и риском. Переключение архитектуры сейчас имеет гораздо худший expected information per GPU-hour: если новая архитектура улучшится или ухудшится, будет трудно понять, сыграли ли роль данные, loss, capacity или optimization. BS-RoFormer сам по себе остаётся конкурентоспособным классом моделей, а новые работы 2026 года продолжают улучшать именно способы представления полос и subband modelling, а не показывают, что идея band-split Transformer исчерпана. citeturn498719academia3turn434132view7turn434132view6

Три наиболее вероятных bottleneck я ранжирую так:

1. **Недостаточная эффективная вариативность обучения — высокая доказательность.** 60 песен / 3.62 часа для 11.66M-параметровой модели — очень узкий supervised corpus; MSS-литература прямо отмечает недостаток multitrack training data и использует random remixing именно для увеличения числа комбинаций. citeturn434132view0
2. **Stereo objective / balance компонентов loss — средне-высокая доказательность проблемы, но причина пока не доказана.** Ваши Side SDR заметно хуже Hybrid, а существует специально разработанный sum/difference MR-STFT loss для stereo audio. Это делает stereo loss хорошей гипотезой, но без исходного `loss.py` нельзя утверждать, что текущая нормализация ошибочна. citeturn453133search0turn453133search2
3. **Слишком узкий corpus / filtering policy — средняя доказательность.** MoisesDB содержит 240 tracks от 45 artists, тогда как у вас используются 60 tracks от 26 artists. До анализа причин исключения остальных 180 нельзя считать увеличение датасета очевидным решением: часть исключений может защищать от bleed, alignment errors или split contamination. citeturn625989academia11turn620631search2

Ниже я отделяю то, что подтверждено источниками, от гипотез, которые необходимо проверить локально.

## Данные, аугментации и лицензии

### Random mixing — самое сильное первое изменение

Работа *Why does music source separation benefit from cacophony?* рассматривает стандартную MSS-аугментацию, при которой stems разных песен комбинируются в новые смеси. Авторы специально исследуют парадокс: такие комбинации нарушают согласованность ритма и тональности, но всё равно улучшают separation. Они также показывают, что просто бесконечное число комбинаций не означает бесконечный рост качества — эффект насыщается. Поэтому «random remix всё исправит» было бы неверным выводом. citeturn434132view0

Более свежий официальный MERL repository для *Embracing Cacophony* содержит training dataloader с random mixing, EQ augmentation и multiple sampling instrumental stems; авторы также отмечают, что их лучший recipe использует эти техники совместно. Репозиторий опубликован под AGPL-3.0-or-later, поэтому **копировать код напрямую в проект с несовместимой лицензией без проверки нельзя**, но алгоритм можно реализовать самостоятельно по статье. citeturn620631search0

MSST также реализует on-the-fly loudness perturbation, same-source mixup, channel operations, pitch shift, EQ, distortion и time stretch. Это полезный инженерный reference, но наличие аугментации в популярном framework само по себе не доказывает, что она улучшит вашу задачу. citeturn434132view2

Для первого pilot я бы **не включал одновременно pitch, time-stretch, reverb, distortion и random remix**. Тогда невозможно будет понять причину результата. Первая гипотеза должна быть минимальной:

**Vocal:** один vocal stem из одной training song.  
**Instrumental:** accompaniment из другой training song.  
**Gain:** независимый умеренный random gain для vocal и instrumental.  
**EQ:** с вероятностью, например, 0.25 — мягкий stereo-linked EQ на каждом source.  
**Stereo:** одинаковое преобразование L/R внутри одного stem, никаких независимых channel effects.

Pitch/time stretch следует оставить отдельным последующим ablation. Demucs действительно использовал pitch/tempo augmentation, а MSST поддерживает эти transformations, поэтому направление разумное, но оно добавляет resampling artifacts и дополнительную переменную. citeturn942083search6turn434132view2

### MoisesDB

Официальный MoisesDB содержит **240 tracks, 45 artists и 12 genres**, а stems и mixture доступны через официальный Python package. Лицензия — **CC BY-NC-SA 4.0**. Hugging Face page существует, но его dataset card прямо отправляет пользователя скачивать данные через официальный research source; поэтому HF listing нельзя считать отдельным новым источником аудио. citeturn625989academia11turn620631search2turn625989search0

Ваши оставшиеся MoisesDB tracks — **первая вещь, которую следует попытаться вернуть в train**, прежде чем смешивать разные datasets. Они ближе всего к вашей текущей domain distribution, taxonomy и acquisition pipeline. Но пока неизвестно, почему 180 tracks были отброшены.

Нужен автоматический audit с категориями:

`length mismatch` / `sample-rate mismatch` / `channel mismatch` / `stem-sum mismatch` / `bleeding annotation` / `missing vocal` / `silent target` / `alignment failure` / `dev or holdout artist` / `other`.

Особенно важно не «починить» dataset путём обрезания tracks до общего span, пока не подтверждено sample-accurate alignment. Common-span crop безопасен только для случаев, где различается длина контейнера, но shared region синхронен.

### MUSDB18-HQ

MUSDB18-HQ — реальный stereo multitrack dataset: **100 training songs + 50 test songs, 44.1 kHz WAV**, с `vocals`, `drums`, `bass`, `other` и mixture. Архив на Zenodo занимает 22.7 GB. Его условия ограничивают использование educational/non-commercial purposes; dataset также композитный и содержит разные исходные лицензии. citeturn434132view3

Для вашей двухstem задачи можно сформировать:

`vocal = vocals.wav`

`instrumental = drums + bass + other`

или использовать `mixture - vocals`, но предпочтительно проверить точность additive reconstruction и не смешивать способы target construction без необходимости.

**MUSDB18-HQ train — наиболее практичный внешний corpus после MoisesDB.** Но его 50-song test должен оставаться полностью вне training и tuning, если вы захотите когда-либо сравниваться с MUSDB literature. Публикуемые MUSDB SDR нельзя сравнивать напрямую с вашими MoisesDB artist-macro SDR: разные песни, targets, aggregation и зачастую evaluation protocol. Официальный MUSDB resource сам подчёркивает предназначение fixed train/test split. citeturn434132view3turn625989search3

### MedleyDB

MedleyDB предоставляет реальный multitrack audio, доступ запрашивается через Zenodo; лицензия — CC BY-NC-SA 4.0 / non-commercial research. Авторы предупреждают, что некоторые записи редактировались, remix-ились или дополнялись при подготовке dataset. citeturn434132view4

Критический риск: MUSDB содержит **46 tracks из MedleyDB**, поэтому наивное использование обоих datasets создаёт duplicate leakage. Независимый aggregation project явно обрабатывает эти 46 overlaps, но для научно чистого pipeline я рекомендую сделать собственный deduplication manifest по canonical artist/title + duration + audio fingerprint, а не просто доверять стороннему агрегатору. citeturn625989search1turn434132view3

MedleyDB имеет смысл только после MUSDB train или как способ расширить diversity, **не как автоматическое прибавление всех файлов**.

### Slakh2100

Slakh2100 содержит 2100 multitrack songs, но audio **синтезирован из MIDI** с sample-based virtual instruments; stems исходно mono, затем автоматически loudness-normalized и mixed. Это отличный источник разнообразных instrumental structures, но не источник реальных vocal recordings. Официальная страница указывает CC BY 4.0. citeturn855962search2

Поэтому Slakh может когда-нибудь служить instrumental-domain regularizer, но **не должен быть Exp11**: ваша основная задача vocal-vs-instrumental, а добавление большого synthetic instrumental domain одновременно изменит timbre distribution и source balance. Сначала есть более близкие real-audio datasets.

Hugging Face сейчас действительно содержит сторонний mirror Slakh audio, но это не официальный источник, и его dataset card/licensing metadata отличается от официальной Slakh page. Для исследовательского provenance лучше использовать официальный Slakh distribution. citeturn625989search4turn855962search2

## Loss, stereo и архитектура

### Почему stereo-aware loss — сильный второй эксперимент

`auraloss` реализует `SumAndDifferenceSTFTLoss`: stereo waveform преобразуется в sum и difference signals, после чего на каждом рассчитывается multi-resolution STFT loss. Реализация позволяет раздельно задавать веса sum и difference components. citeturn453133search0turn453133search2

Для вашей задачи это логично: обычный L/R waveform или complex STFT loss может дать низкую общую ошибку, при этом плохо реконструировать относительно слабую Side component. Если Side energy значительно ниже Mid energy, не-нормализованный absolute error легко доминируется mid-channel content. **Это математически правдоподобная гипотеза, а не установленная причина вашего результата.**

Поэтому перед Exp stereo надо измерить:

\[
M=(L+R)/2,\qquad S=(L-R)/2
\]

Для каждого training crop:

- RMS Mid;
- RMS Side;
- текущий loss contribution Mid;
- current loss contribution Side;
- gradient norm каждого loss component;
- Side/Mid target-energy ratio.

Если `side_l1` уже нормализован относительно energy Side, новый loss может оказаться почти дублирующим. Если он считается absolute L1 и Side энергетически слаб, гипотеза становится гораздо сильнее.

MR-STFT полезен потому, что разные FFT resolutions одновременно оценивают transient и longer-harmonic structures; `auraloss` предоставляет multi-resolution и stereo sum/difference варианты. citeturn453133search1

Я **не рекомендую сразу добавлять explicit ILD/IPD loss**. Phase differences нестабильны около spectral nulls и требуют masking/weighting по энергии. Это более рискованно, чем Mid/Side MR-STFT, и легко создать плохо обусловленный objective.

### Mixture consistency

У вас она уже есть, и это разумно сохранять. TF-GridNet также использует complex spectral mapping и дополнительный objective, стимулирующий separated sources суммироваться обратно в mixture. Однако TF-GridNet — прежде всего speaker-separation architecture; его результаты нельзя переносить на музыкальную двухstem stereo задачу. citeturn855962academia8

### Архитектура

Ваш BS-RoFormer-inspired подход остаётся технически оправданным. Оригинальный BS-RoFormer использует band-split complex spectrogram representation, hierarchical Transformer processing и RoPE. citeturn498719academia3

**Mel-Band RoFormer** имеет смысл как future architecture ablation, но переключаться сейчас нерационально: вы измените одновременно frequency partitioning и model capacity.

**SCNet** использует sparse compression of frequency representations и имеет официальный training implementation; это хороший low-compute architecture candidate, но опять же требует нового scratch baseline и не отвечает на вопрос, ограничены ли вы данными. citeturn855962search4

**Hybrid Transformer Demucs** объединяет waveform и spectrogram paths и transformer interaction. Авторы сами сообщают, что вариант Transformer Demucs оказался слабым при обучении только на MUSDB, а выигрыш возник с дополнительными 800 songs — ещё один сигнал, что более сложная архитектура не компенсирует ограниченный training corpus автоматически. citeturn498719academia4

**BandIt** — bandsplit architecture с официальным Apache-2.0 implementation; его design интересен как более parameter-efficient bandsplit family, но пока нет причины считать его заменой, которая исправит именно ваши stereo/bleed tradeoffs. citeturn855962search3

Две очень свежие работы 2026 года заслуживают наблюдения, но не немедленной миграции. **Dynamic Band Splitting** обучает content-dependent frequency boundaries и показывает преимущество над static versions нескольких backbones; **SubRoformer** сочетает subband time-domain representation с residual RoFormer и complex masks. Обе идеи появились слишком недавно, чтобы считать их low-risk deadline experiments, и опубликованные gains получены на других setups. citeturn434132view7turn434132view6

**Итого: архитектуру до 23 ноября не менять, если augmentation/data/loss pilots не окажутся полностью отрицательными.**

### Optimization и inference

EMA у вас уже проверена и не дала систематического выигрыша в конце Exp10, поэтому дополнительный полный EMA experiment не имеет смысла.

Checkpoint averaging можно проверить **без retraining**: усреднить несколько собственных nearby Raw checkpoints в parameter space и прогнать ровно тот же dev evaluation. Это дешёвый inference-side diagnostic, но результат не стоит смешивать с основной data hypothesis.

4 s → 8 s context потенциально добавляет musical context, однако Transformer memory cost может расти сильно. Demucs documentation также отмечает зависимость quality/memory от segment length, хотя это другая architecture. citeturn855962search5 Поэтому сначала нужен 200–500-update memory/throughput probe, не full run.

PyTorch SDPA автоматически выбирает CUDA implementations между fused FlashAttention, memory-efficient attention и math backend при совместимых inputs. Проверять backend полезно для увеличения context, но **не следует менять pinned PyTorch/CUDA stack одновременно с quality experiment**, иначе нарушается reproducibility. citeturn453133search4turn453133search6

## Decision matrix и последовательность экспериментов

| Гипотеза | Evidence | Вероятная область эффекта | Основной риск | Pilot информативность | GPU cost | Scratch run? | Приоритет |
|---|---|---|---|---|---|---|---|
| Cross-song vocal/accompaniment remix + gain | **Высокий** | Generalization, vocal/instrumental SDR, bleed | Unnatural mixtures / loudness distribution | **Очень высокая** | Низкий–средний | Да для чистого сравнения | **1** |
| Conservative EQ augmentation | Высокий/средний | Robustness к timbre, SDR | Excess EQ creates unrealistic spectra | Высокая | Низкий | Да | **1, вместе с controlled augmentation arm либо отдельная ablation** |
| Restore valid remaining MoisesDB | **Высокий потенциал, evidence unknown locally** | Все метрики через diversity | Bad alignment/bleed/leakage | Очень высокая после audit | Низкий training overhead | Да | **2** |
| MUSDB18-HQ train addition | Высокий | Diversity/generalization | Domain shift, license, split overlap | Высокая | Средний | Да | **2–3** |
| Mid/Side MR-STFT loss | Средне-высокий | Vocal/Instr Side SDR, artifacts | Может ухудшить SDR/bleed | **Очень высокая** | Низкий | Да | **2** |
| 8 s context | Средний | Long phrases / harmonic disambiguation | VRAM, throughput, smaller effective batch | Средняя | Высокий | Да | 3 |
| Checkpoint averaging | Средний | Small stability gain | May blur minima | Высокая | Только eval | Нет | 3 |
| Pitch/time augmentation | Средний | Generalization | Artifacts, domain distortion | Средняя | Средний | Да | 3 |
| Architecture: SCNet/Mel-RoFormer | Средний | Potential broad changes | Confounded comparison, implementation bugs | Низкая коротким pilot | Высокий | **Да** | 4 |
| DBS/SubRoformer adaptation | Promising but very recent | Frequency representation | Engineering/reproduction risk | Низкая до implementation | Высокий | **Да** | 5 |

Первый эксперимент я бы назвал **Exp11-Diversity**.

**Hypothesis:** при неизменной model/loss/optimizer увеличение эффективного сочетательного разнообразия sources уменьшит зависимость separator от correlations конкретных 60 songs и улучшит artist-macro generalization.

**Изменить только dataloader.** На каждом crop с заранее определённой вероятностью строить synthetic training pair из vocal одного разрешённого training track и accompaniment другого training track. Target instrumental должен быть именно соответствующим accompaniment, а mixture должен создаваться как их сумма. Original aligned mixes оставлять заметной долей minibatches, чтобы модель не потеряла distribution настоящей музыки. MERL evidence поддерживает сам принцип random cross-song mixing, но точную probability для вашей модели необходимо выбрать заранее и не подбирать по dev десятки раз. citeturn434132view0turn620631search0

Independent gain можно применить к sources перед summation, но с фиксированным разумным range и peak control. EQ — только stereo-linked и умеренный. Я бы **не включал pitch/time stretch в Exp11**, чтобы результат оставался интерпретируемым.

Не менять: architecture, seed policy, optimizer, LR schedule, crop 4 s, batch 4, number of optimizer updates, checkpoint cadence, inference, evaluation metrics, dev songs, guardrails.

### GO / NO-GO

Pilot не должен использовать правило «Vocal SDR вырос на 0.03 dB — запускаем full». На маленьком dev это приводит к selection noise.

Предварительно зафиксировать GO, например:

- artist-macro Vocal SDR не хуже matched Exp10 checkpoint более чем на 0.10 dB;
- Instrumental SDR не хуже более чем на 0.10 dB;
- ни один Side SDR не хуже более чем на 0.15 dB;
- bleed не ухудшается более чем на 1 percentage point;
- не менее 4 из 7 artist groups двигаются в правильную сторону по основному composite comparison;
- никаких NaN, clipping distribution shift или mixture-consistency regressions.

Числа выше — **decision tolerances**, а не ожидаемые gains. Их надо зафиксировать до просмотра pilot result.

Pilot лучше сравнивать **на том же update count с контрольным Exp10 trajectory**, например на 4k или 6k updates, а не с Exp10 final 24k. Поскольку у вас deterministic pipeline, matched-step comparison особенно информативно.

Если pilot проходит — full 24k scratch run. Если нет — **не пытаться спасать его подбором пяти probabilities**; перейти к stereo-loss pilot.

### Следующие контролируемые experiments

**Exp12-StereoLoss.** Только если audit показывает, что Side objective энергетически недовзвешен. Добавить normalized Mid/Side MR-STFT term поверх существующего objective, с отдельным логированием всех components и gradient norms. Не менять augmentation. Full run только если короткий pilot улучшает Side SDR без явной потери vocal/instrumental SDR. Stereo sum/difference MR-STFT имеет готовую математически прозрачную reference implementation. citeturn453133search0turn453133search2

**Exp13-Data.** После audit remaining MoisesDB добавить только tracks, прошедшие alignment/bleed/split checks. Если usable прирост мал, использовать MUSDB18-HQ **train only**. Artist/track dedup обязательны. MUSDB test не использовать. citeturn434132view3

**Exp14-Context**, только если предыдущие изменения дают plateau. Сделать 8 s memory probe и matched pilot с уменьшенным batch + gradient accumulation для сохранения effective batch. Если throughput падает настолько, что за фиксированное wall-clock время модель видит существенно меньше independent examples, 8 s может оказаться невыгодным.

**Architecture experiment я бы до дедлайна не делал**, кроме ситуации, где все первые три гипотезы провалены уже к 19–20 ноября.

## Диагностика Exp10 и правильная model selection

Exp10 почти воспроизвела Exp09 потому, что — согласно вашему run audit — architecture, dataset, crop schedule, seed/input addresses и optimizer trajectory были сохранены. Это сильное подтверждение воспроизводимости, а не свидетельство, что BS-RoFormer достиг теоретического потолка.

Но **четыре вещи сейчас невозможно проверить удалённо и их нельзя выдумывать**. До Exp11 Codex должен открыть:

- файл реализации composite loss и точные coefficients;
- код stereo conversion / STFT / reconstruction;
- eligibility manifest всех 240 MoisesDB tracks с reason codes;
- split manifest с artist IDs и UUID;
- evaluation implementation, включая SDR/SI-SDR aggregation;
- training config Exp09/10;
- несколько hundred per-step component losses и ideally gradient norms;
- verification того, что `mixture == vocal_target + instrumental_target` в том exact domain, который получает модель.

Особенно я бы проверил **scale каждого loss term**. Наличие коэффициента `side_l1` ничего не говорит о реальном влиянии: важны его expected magnitude и gradient contribution. Гипотеза о конфликте bleed-vs-retention должна проверяться correlations по crop/song между loss components и validation metrics, а не интуитивно.

### Multi-objective selection

`best_overall = max Vocal SDR` больше не подходит как единственный scientific selector.

Нужно сохранить **Pareto frontier** checkpoints по:

Vocal SDR ↑, Vocal SI-SDR ↑, Instrumental SDR ↑, bleed ↓, projected bleed energy ↓, retention ↑, Vocal Side SDR ↑, Instrumental Side SDR ↑.

Не надо превращать их сразу в произвольный weighted scalar. Используйте:

1. hard guardrails против существенного regression;
2. Pareto dominance;
3. заранее выбранный primary metric;
4. paired artist-level uncertainty;
5. blind listening среди 2–3 nondominated candidates.

Семь artist groups — маленькая единица bootstrap. Поэтому confidence intervals будут информативны, но не чрезвычайно точны. А использование одной 16-song panel для десятков architecture/checkpoint decisions постепенно превращает её в de-facto training signal. Поэтому после нескольких следующих experiments нужно **прекратить ручной checkpoint fishing**.

Final holdout должен быть открыт **один раз**, после freezing model, inference parameters и selection rule.

Blind listening должен отдельно оценивать: residual vocals в instrumental, missing accompaniment, vocal warbling, stereo collapse, transients/reverb tails. Не подменять listening одним SDR.

Checkpoint averaging можно оценить до нового обучения как дешёвый side experiment, но не выбирать десятки averaging windows по той же dev panel.

## Реалистичный график до 23 ноября

**15 ноября вечером:** никаких новых full runs. Read-only forensic audit Exp09/10: loss normalization, Mid/Side energy, data exclusions, split integrity, mixture consistency. Сделать manifest всех 240 MoisesDB tracks и exact reason каждого exclusion. Это самый важный незавершённый технический шаг.

**16 ноября:** реализовать Exp11-Diversity в новой isolated experiment directory. Unit tests: deterministic sample addressing; never draw dev/final tracks; channel alignment; source sum; gain bounds; clipping; repeatability after resume. Затем 4k–6k pilot. По вашей текущей скорости это должно укладываться в несколько часов GPU; exact estimate нужно получить по первым 500 updates, потому что augmentation увеличит CPU/data-loader overhead.

**17 ноября:** если GO — full 24k Exp11 scratch run и paired evaluation. Ваш historical baseline — 4.4 h Exp09 / 6–7 h Exp10, поэтому один обычный full run укладывается в рабочий день. Если NO-GO — не rerun; запускается Exp12 stereo pilot.

**18 ноября:** audit remaining MoisesDB eligibility и clearance семи common-span candidates. Если можно безопасно увеличить real corpus — собрать Exp13 data manifest. Если нет — запросить/подготовить MUSDB18-HQ train; официальный архив 22.7 GB и содержит 100 training tracks. citeturn434132view3

**19 ноября:** full run лучшей из двух подтверждённых гипотез: augmentation либо data expansion/stereo loss. Не комбинировать две непроверенные идеи.

**20 ноября:** максимум один ablation — вероятнее всего stereo-aware loss поверх лучшего data recipe либо data expansion поверх подтверждённого augmentation recipe. Это первый момент, когда комбинация факторов оправдана: каждый компонент уже проверен отдельно.

**21 ноября:** freeze candidate set. Checkpoint averaging и inference overlap можно исследовать как inference-only changes, но зафиксировать их до final test. 8 s context запускать только если предыдущие experiments дали неожиданно слабый результат и memory probe показывает разумную скорость.

**22 ноября:** blind listening, per-song failure audit, paired artist bootstrap, Pareto selection. Выбрать один primary candidate и один fallback. Больше training не начинать, кроме исправления явного воспроизводимого bug.

**23 ноября:** открыть sealed holdout один раз для frozen primary candidate; подготовить final table, listening notes, hashes, configs, provenance, limitations.

Что **нереалистично** сделать качественно до дедлайна: полноценно воспроизвести с нуля SCNet + Mel-Band RoFormer + DBS + SubRoformer, оптимизировать каждую architecture и провести multi-seed statistical comparison. Это превратится в architecture lottery, а не исследование.

## Готовое задание Codex

Передавать Codex я рекомендую **сначала именно аудит + подготовку Exp11, без разрешения training**:

```text
PROJECT: Music Source Separation
TASK: Forensic audit + prepare Exp11-Diversity
MODE: READ-ONLY for Exp01–Exp10. DO NOT START GPU TRAINING.

Project root:
C:\Users\YOUR_USER\music_source_separation

Baseline:
Exp10 is the frozen reproducible baseline.
Architecture, optimizer, splits, evaluation and checkpoints from Exp01–Exp10 must not be modified.

GOAL

Prepare one controlled experiment testing a single main hypothesis:

Cross-song source remixing increases effective training diversity and improves
generalization of the existing two-stem scratch-trained separator.

Do NOT change architecture, model capacity, optimizer, LR schedule,
validation set, checkpoint selection implementation or evaluation metrics.

PART A — FORENSIC AUDIT

1. Locate the exact Exp09/Exp10:
   - model implementation
   - loss implementation
   - training config
   - data loader
   - source reconstruction code
   - split manifests
   - validation code
   - eligibility/filtering code

2. Produce a report of every current loss term:
   - exact mathematical expression
   - configured coefficient
   - mean magnitude during Exp10
   - relative contribution to total loss
   - if practical, sampled gradient norm contribution
   - normalization denominator

Pay special attention to:
   complex spectral loss
   log magnitude loss
   waveform loss
   side loss
   leakage loss
   mixture consistency

Do not claim that a component is over/under-weighted without measurement.

3. Stereo audit.

For a reproducible sample of training crops calculate:
   Mid = (L + R) / 2
   Side = (L - R) / 2

Report:
   target Mid RMS
   target Side RMS
   Side/Mid energy ratio
   prediction error Mid
   prediction error Side
   current side-loss normalization

Determine whether low Side energy can mathematically cause the current
objective to underweight stereo reconstruction.

Do not modify the loss in Exp11.

4. Dataset eligibility audit.

Enumerate all 240 MoisesDB tracks.

For every track output exactly one primary status:
   train_eligible
   dev
   final_holdout
   missing_target
   length_mismatch
   alignment_failure
   bleed_rejected
   channel_problem
   silence_problem
   corrupted
   artist_overlap
   other

Include exact machine-readable reason data.

Confirm artist-disjointness of train/dev/final.

For the seven previously identified common-span candidates,
verify sample rate, channels, exact alignment in shared span,
stem-sum consistency and bleed metadata.
DO NOT add them to training.

5. Verify mixture construction for the existing 60 training tracks.

Check numerically whether:
   mixture == vocal_target + instrumental_target

Report residual RMS, peak and relative energy distributions.

PART B — PREPARE EXP11-DIVERSITY

Create a new isolated experiment directory.
Do not alter Exp01–Exp10.

The only substantive training change versus Exp10 must be the data sampler.

Implement deterministic cross-song source remix augmentation:

- all source tracks must come only from the existing approved TRAIN split
- never use dev or final holdout audio
- select vocal crop from track A
- select instrumental crop from track B
- A and B must be independently addressable and logged
- build mixture exactly as augmented_vocal + augmented_instrumental
- retain a predefined proportion of original same-song mixtures
- deterministic RNG state must be checkpointed and exactly restorable

Implement conservative independent source gain augmentation.

Implement optional stereo-linked EQ augmentation:
- identical filter parameters applied to L/R of a given source
- do not alter stereo geometry independently per channel
- parameters and random draw logged
- EQ can be disabled independently for ablation

DO NOT implement in Exp11:
- pitch shifting
- time stretching
- reverb
- compression
- distortion
- architecture changes
- loss changes
- new external datasets

PART C — TESTS

Add CPU tests proving:

1. no dev/final UUID can be sampled
2. deterministic replay reproduces identical crop addresses and augmentations
3. resume reproduces RNG state
4. target vocal + target instrumental equals synthetic mixture within float tolerance
5. no clipping or non-finite samples
6. stereo channel relationship is preserved by linked transformations
7. augmentation can be fully disabled and then reproduces Exp10 input crops
8. all existing Exp10 tests still pass

Run a tiny GPU preflight only if necessary for correctness.
Do not start the pilot training.

PART D — PILOT PLAN

Prepare, but DO NOT RUN, a matched 6,000-update scratch pilot.

Same as Exp10:
- same model
- same initialization policy
- same optimizer
- same LR schedule
- same 4 s crop duration
- same effective batch
- same validation songs
- same evaluation code
- same validation cadence

Before seeing results, encode GO/NO-GO criteria:

At matched update count versus Exp10:
- Vocal SDR regression no worse than 0.10 dB
- Instrumental SDR regression no worse than 0.10 dB
- Vocal Side SDR regression no worse than 0.15 dB
- Instrumental Side SDR regression no worse than 0.15 dB
- absolute vocal bleed regression no worse than 1 percentage point
- inspect all 7 artist groups, not only aggregate
- no numeric instability or mixture-consistency regression

These thresholds are decision tolerances, not expected gains.

Do not evaluate the sealed final holdout.

PART E — OUTPUT

Return a Russian report containing:

1. exact Exp10 loss formulation and measured scale of every component
2. stereo Mid/Side diagnostic
3. complete MoisesDB eligibility counts by reason
4. status of the seven common-span tracks
5. split contamination audit
6. Exp11 file diff versus Exp10
7. all test results
8. estimated pilot GPU time from measured preflight throughput
9. exact command that WOULD launch the pilot
10. explicit statement:
   READY_FOR_AUTHORIZATION
   or
   NOT_READY

Do not start training.
Do not download external datasets.
Do not modify or delete any Exp01–Exp10 file.
```

Этот prompt важнее немедленной команды `train`. Если аудит обнаружит, например, что большая часть оставшихся 180 MoisesDB tracks была исключена из-за слишком консервативного, но технически исправимого length rule, **приоритет Exp11 может измениться**: увеличение реального corpus окажется более чистым intervention, чем synthetic remixing.

И наоборот, если exclusions обоснованы bleed/alignment problems, первый augmentation experiment остаётся лучшим следующим шагом.

Самая важная смена стратегии после Exp10 — **перестать считать каждый полный run прогрессом**. Следующие runs должны либо улучшать модель, либо давать однозначный ответ на конкретную гипотезу. К 23 ноября реалистичная цель — не гарантировать прирост всех метрик, а получить один Pareto-лучший independently trained separator с доказуемой provenance, sealed evaluation и понятным объяснением того, какие изменения действительно помогли.