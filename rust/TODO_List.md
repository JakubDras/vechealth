# VecHealth — TODO / Roadmap do pierwszej realnej wersji biblioteki

Zapis rzeczy zidentyfikowanych podczas przeglądu projektu jako "biblioteki dla
AI engineerów / MLOps", żeby nie zgubić ich po drodze. Nieuporządkowane
chronologicznie — priorytety oznaczone tagiem.

Kontekst: `rust/` = produkcyjny kod biblioteki (to co się buduje i publikuje
jako pakiet `vechealth`). `python/` = eksperymenty, benchmarki, artykuły —
NIE część dystrybuowanego pakietu.

Stan na dziś (punkt odniesienia): 8 metryk geometrycznych zaimplementowanych
w `rust/core/src/metrics/` (hubness, fragmentation/dispersion, anisotropy,
outliers, duplicates, intrinsic_dim, qmas, snc) + orkiestrator `compute_all`
(`rust/core/src/metrics/all.rs`, wystawiony w bindings jako
`VecHealthEvaluator.compute_all()`). Wejście: macierz w pamięci (dowolny
numeryczny array-like — `float32`/`float64`/int/listy Pythona, patrz
"Wygodne loadery wejścia" niżej) albo plik `.npy`/`.csv`/`.parquet` przez
`from_local(...)`.

---

## 🔴 Krytyczne — bez tego narzędzie nie nadaje się do realnego użycia na produkcyjnym vectorstorze

- [ ] **Konektory do vectorstore'ów** (`python/vechealth/connectors` to dziś
      pusty stub). Dziś użytkownik musi sam wyciągnąć wektory do
      `numpy.float32` z Pinecone/Qdrant/Weaviate/Milvus/pgvector zanim
      cokolwiek policzy. To przeciwieństwo "monitorowania jakości embeddingów
      w vectorstorze" — minimalna wersja: `from_qdrant(...)`,
      `from_pinecone(...)` itd. zwracające gotową macierz.

- [ ] **Sampling dla dużej skali** (`core/src/sampling` i
      `python/vechealth/sampling` puste). Silnik KNN to brute-force
      O(n²·d) — OK dla dziesiątek/setek tysięcy wektorów, nieużywalne na
      milionach wektorów typowych dla produkcyjnego RAG. Potrzebny losowy /
      stratyfikowany sampling do reprezentatywnej podpróbki przed
      liczeniem metryk.

- [ ] **Warstwa interpretacji wyników.** `compute_hubness()` zwraca
      `hubness_skewness=0.42` — bez progów/heurystyk ("kiedy to jest
      problem?"), bez klasyfikacji zdrowia (`healthy`/`warning`/`critical`),
      bez rekomendacji. To dokładnie to, co README obiecuje
      (`report.health_score`, `report.pathologies`, `report.recommendations`)
      i czego brakuje w kodzie. Nie chodzi o serwer/dashboard — tylko o
      logikę interpretacji w Pythonie nad wynikami `compute_all()`.

- [ ] **Hub Inspector — identyfikacja i inspekcja konkretnych punktów
      odpowiedzialnych za wysoką hubness, nie tylko zbiorcza statystyka.**
      Wynika bezpośrednio z eksperymentu na danych rzeczywistych
      (ClinicalTrials.gov, `python/benchmarks/evaluation/experiments/
      exp07_wild_data/clinicaltrials_wild.py`, funkcje
      `identify_hub_occurrences` + `inspect_hubs`) — pokazał się tam
      realny, powtarzalny wzorzec: `hubness_max` samo w sobie nie mówi,
      *czy* to problem, ani *dlaczego* — dopiero ręczne sprawdzenie, które
      konkretnie punkty są hubami, ujawniło że to nie szum geometryczny,
      tylko sensowna (choć nieoczekiwana) struktura semantyczna. Bez tej
      warstwy użytkownik dostaje liczbę i musi sam odtwarzać tę analizę
      ręcznie w Pythonie, tak jak myśmy zrobili w skrypcie badawczym.

      **Napięcie architektoniczne do świadomego rozstrzygnięcia przed
      implementacją:** oryginalny dokument projektu (`00_VecHealth_Overview`,
      sekcja "Bezpieczeństwo i prywatność danych") celowo zakłada, że
      `analyze(vectors: np.ndarray)` nigdy nie przyjmuje treści/metadanych —
      wyłącznie surowe wektory, żeby strukturalnie uniemożliwić przypadkowy
      wyciek treści źródłowej. Hub Inspector z natury chce łączyć indeks
      wektora z czymś, co pozwala go zidentyfikować "na zewnątrz" — to nie
      wymaga łamania tej zasady, jeśli zaprojektowane świadomie:

      - Rdzeń (Rust, `core`): `identify_hubs(vectors, k, top_n) ->
        Vec<(usize, u32)>` — zwraca WYŁĄCZNIE indeksy + liczbę wystąpień w
        k-NN. Zero zmiany filozofii bezpieczeństwa, to nadal czysta funkcja
        na macierzy.
      - Warstwa Python (opcjonalna, jawnie oddzielna): jeśli użytkownik
        SAM dostarczy równoległą tablicę identyfikatorów (`id_array` — te
        same ID, które już ma po swojej stronie w Qdrant/Pinecone/etc.,
        nie treść), biblioteka może zwrócić `(id, occurrence_count)`
        zamiast surowych indeksów — użytkownik i tak musi sam odpytać
        swój własny system, żeby zobaczyć treść. VecHealth nigdy nie
        przechowuje ani nie przetwarza samej treści.
      - Opcjonalne wzbogacenie (jawnie opt-in, osobna metoda): jeśli
        użytkownik dostarczy dodatkowo LEKKIE etykiety kategorii/tagów
        (nie pełną treść — dokładnie jak `category` w eksperymencie
        ClinicalTrials.gov), biblioteka może automatycznie policzyć
        rozproszenie hubów między kategoriami — dokładnie ten test, który
        w exp07 trzeba było robić ręcznie, a który realnie odróżnił
        "zdrowe klastrowanie tematyczne" od "prawdziwej patologii
        geometrycznej" (huby skupione w 1 kategorii → prawdopodobnie
        zdrowe; huby rozproszone między wieloma kategoriami → prawdopodobna
        patologia).

      Proponowane API w Pythonie:
      ```python
      report = evaluator.inspect_hubs(k=10, top_n=20, id_array=my_ids)
      # report: lista (id, occurrence_count), posortowana malejąco

      report = evaluator.inspect_hubs(k=10, top_n=20, id_array=my_ids,
                                       category_labels=my_categories)
      # dodatkowo: report.category_dispersion, report.likely_pathology (bool)
      ```

---

## 🟠 Ważne — mocno ograniczają realną użyteczność

- [x] **Śledzenie w czasie / porównanie z baseline.** Nowy crate
      `vechealth-report` (`rust/report/`, sąsiad `core`/`connectors`/
      `bindings` w workspace — `core` zostaje bez zmian, zero zależności od
      serde). `Report` = wersjonowany snapshot (`schema_version`,
      `generated_at`, `vechealth_version`, `dataset` — fingerprint z
      `content_hash`, `config`, `tags`, `metrics`) z `save()`/`load()`
      (JSON) i `flatten()` (płaski widok `"{grupa}.{pole}"`, gotowy pod
      przyszły metric store / eksporter Prometheus). `compare(baseline,
      current)` zwraca deltę per metrykę (`baseline`/`current`/`delta`/
      `delta_pct`) + `warnings` przy niekompatybilnym configu (`k`,
      `k_intrinsic_dim`, `duplicate_epsilon`) lub innym wymiarze —
      **świadomie bez progów pass/fail**, to należy do osobnego zadania
      "warstwa interpretacji" (🔴 wyżej), które jeszcze nie istnieje.
      W Pythonie: `evaluator.compute_report(...)` (jak `compute_all`, plus
      `tags`), `Report.save/load/compare/flatten/to_dict/to_json`,
      `Comparison`, `MetricDelta`, wyjątek `ReportError`. Testy: 9 w
      `rust/report/src/*.rs` (`cargo test --workspace`) + smoke test
      `python/tests/test_report.py`. CI-gating z przykładu ("hubness wzrósł
      o 30%") jest teraz możliwy do zrobienia w Pythonie ręcznie po
      `delta_pct`; automatyczny pass/fail nadal czeka na warstwę
      interpretacji.

- [ ] **Filtrowanie szumu w `compare()` według kategorii niezawodności
      metryki (nawiązanie do już zaimplementowanego `Report.compare()`
      wyżej).** exp02 (`python/benchmarks/evaluation/experiments/
      exp02_subsampling_stability`) ustalił cztery kategorie niezawodności
      metryk pod subsamplingiem: (A) stabilna — `anisotropy_mean_norm`;
      (B) korygowalna analitycznie — `ndds_fraction`; (C) zbieżna
      monotonicznie — `dispersion_*`, `intrinsic_dim_mean`, `qmas_*`,
      `snc_score`; (D) genuinely niestabilna — `hubness_max`,
      `hubness_skewness` (wariancja ROŚNIE z N, nie maleje). Dziś
      `compare()` raportuje `delta_pct` identycznie dla wszystkich metryk —
      użytkownik nie ma jak odróżnić "hubness_max skoczył o 40%, bo to
      zwykły szum tej konkretnej metryki" od "hubness_max skoczył o 40%, bo
      coś się realnie popsuło". Propozycja: opcjonalny próg istotności per
      kategoria (D wymaga większej zmiany, żeby zasygnalizować alarm, niż
      A), oparty wprost o zmierzone w exp02 poziomy wariancji, nie o
      arbitralne wartości.

- [ ] **Kontekst wdrożenia jako opcjonalny parametr `Report`
      (`deployment_context`).** Dwa niezależne wyniki eksperymentalne
      pokazują, że ta sama wartość metryki wymaga innej interpretacji
      zależnie od kontekstu produkcyjnego, którego VecHealth dziś w ogóle
      nie zna:
      1. **Typ indeksu ANN** (`python/benchmarks/evaluation/experiments/
         exp05_ann_vs_exact`) — hubness/noise dewastują HNSW i IVF-PQ
         68-90× silniej niż sugerowałby recall mierzony na exact search;
         anisotropy/voids/collapse są dla obu architektur praktycznie
         nieszkodliwe. Odczyt `hubness_max` bez wiedzy "czy produkcja
         używa HNSW" może być mylący w obie strony.
      2. **Charakter domeny danych** (exp07, Komponent 2,
         ClinicalTrials.gov) — wysoka hubness rozproszona między
         kategoriami okazała się częściowo odzwierciedlać prawdziwą,
         wartościową strukturę semantyczną (wspólna terminologia terapii
         onkologicznych), nie czysty szum geometryczny.

      Propozycja: `Report` przyjmuje opcjonalne pole `deployment_context`
      (np. `index_type: "hnsw"|"ivfpq"|"exact"`, `domain_hint: str`) —
      czysto informacyjne w samym rdzeniu, ale wykorzystywane przez
      przyszłą warstwę interpretacji (🔴 wyżej) do kalibrowania
      wag/ostrzeżeń bez zmiany samych metryk.

- [ ] **Wyjaśnienia gruntowane w danych walidacyjnych projektu, nie tylko w
      statycznym opisie metryki.** Uzupełnienie "Dokumentacji
      interpretacyjnej" niżej (🟡) o coś mocniejszego niż statyczny opis:
      biblioteka ma już (z Fazy 1 i dalszych eksperymentów) skalibrowane
      krzywe dose-response — wiadomo np. jak `hubness_skewness` konkretnej
      wielkości przekładało się na spadek recall@10 w kontrolowanych
      testach. Zamiast generycznego "wysoka hubness może szkodzić
      retrievalowi", `evaluator.explain(metric_name, value)` mogłoby
      zwrócić coś w rodzaju "wartości hubness_skewness w tym zakresie
      odpowiadały w naszej walidacji spadkowi recall@10 rzędu X-Y%" —
      bezpośredni most między materiałem badawczym (`python/benchmarks/`)
      a produktową użytecznością biblioteki, nie dwa osobne światy.

- [ ] **CLI.** `vechealth analyze vectors.parquet --output report.json` —
      do szybkich sprawdzeń ad-hoc i wpięcia w pipeline CI/CD bez pisania
      kleju w Pythonie.

---

## 🟡 Drobniejsze tarcia UX

- [x] **Wygodne loadery wejścia.** Przy realizacji okazało się, że część
      tego punktu była już zrobiona wcześniej i lista była nieaktualna:
      `.npy`/`.csv`/`.parquet` przez `VecHealthEvaluator.from_local(...)`
      (`connectors/src/local.rs`) już obsługiwały automatyczną konwersję
      `float64`→`float32` (patrz test `npy_f64_is_cast_down`). Prawdziwa
      luka była gdzie indziej: bezpośredni, najczęściej używany konstruktor
      `VecHealthEvaluator(vectors)` oraz parametr `queries` w
      `compute_qmas`/`compute_all`/`compute_report` przyjmowały wyłącznie
      dokładnie `numpy.float32` (`PyReadonlyArray2<f32>`) — `float64` albo
      zwykła lista Pythona kończyły się nieczytelnym błędem pyo3
      (`TypeError: 'ndarray' object cannot be converted to 'PyArray<T, D>'`),
      zmuszając do ręcznego `.astype(np.float32)` dokładnie tak, jak opisano
      w tym punkcie. Naprawione zamianą typu parametru na
      `PyArrayLike2<'_, f32, AllowTypeChange>` (`numpy` crate, ten sam
      mechanizm co `numpy.asarray(..., dtype=float32)`) w
      `rust/bindings/src/lib.rs` — akceptuje teraz dowolny numeryczny
      array-like (`float64`, `int32/64`, listy zagnieżdżone Pythona);
      wejście już w `float32` i C-contiguous idzie tą samą ścieżką bez
      kopiowania co wcześniej (zero regresji wydajności dla obecnych
      użytkowników). Smoke test: `python/tests/test_flexible_input.py`.

- [x] **Serializacja wyników** — wszystkie typowane klasy wyników
      (`HubnessResult`, ..., `AllMetricsResult`, w `rust/bindings/src/lib.rs`)
      mają teraz `.to_dict()` / `.to_json()`. Schemat JSON pochodzi z jednego
      źródła prawdy — DTO w nowym crate `vechealth-report` (patrz pozycja
      "Śledzenie w czasie" wyżej) — więc JSON pojedynczej metryki ma
      dokładnie ten sam kształt co jej odpowiednik zagnieżdżony w pełnym
      `Report`. Konwersja `serde` → obiekt Pythona przez `pythonize`
      (jedyna nowa zewnętrzna zależność dodana w ramach tego zadania).

- [ ] **Dokumentacja interpretacyjna** — docstringi opisują *co* liczy
      metryka, nie *jak ją czytać* / kiedy się martwić (powiązane z pozycją
      "warstwa interpretacji" wyżej).

---

## Dojrzałość packagingu / inżynierska (v0.1 release readiness)

- [x] **Wheele na macOS (Intel + Apple Silicon), Windows (x64) i Linux
      (x86_64 + aarch64/Graviton).** `.github/workflows/ci.yml`, job
      `build-wheels` (5-elementowa macierz, oparta o `maturin generate-ci`).
      Zweryfikowane lokalnie: poprawność YAML, brak natywnych/systemowych
      zależności w drzewie Cargo (żadnych `-sys`/OpenSSL/BLAS-linking, które
      mogłyby złamać cross-kompilację). **Nie zweryfikowane realnym CI run**
      (to środowisko nie ma dostępu do Dockera/crossów dla macOS/Windows/
      aarch64) — pierwszy push/PR na GitHubie jest pierwszym prawdziwym
      testem tej macierzy, obserwuj wynik.
- [ ] **Testy warstwy `bindings`** — dziś jedyna weryfikacja Python-facing
      API to ręczny smoke-test, nie zautomatyzowany `pytest` w CI.
- [ ] **`maturin publish` workflow** (ręcznie wyzwalany, osobny od CI) — dziś
      jedyna droga na PyPI to ręczne `maturin publish` z laptopa.
- [ ] **README aktualne względem realnego API** — obiecuje `vh.analyze(...)`
      zwracające `report.health_score` / `pathologies` / `recommendations`,
      czego nie ma w kodzie (najbliższy odpowiednik dziś:
      `evaluator.compute_all()`, ale bez interpretacji — patrz sekcja 🔴).
- [ ] **`CHANGELOG.md` / polityka wersjonowania** przed pierwszym publicznym
      wydaniem.

---

## Świadomie odłożone (nie teraz)

- Integracja w stylu Prometheus/Grafana (eksporter metryk, serwer, dashboard)
  — zgodnie z ustaleniem: najpierw core samej biblioteki i jej
  funkcjonalność, integracje później.
- **Koła `musllinux` (Alpine Linux)** — dziś budujemy tylko `manylinux`
  (glibc: Ubuntu/Debian/RHEL/Fedora), co pokrywa zdecydowaną większość
  obrazów Docker używanych do serwowania ML. Alpine ma słabą kompatybilność
  z ekosystemem numpy/scipy/torch, więc ryzyko trafienia na realny przypadek
  użycia jest niskie — do rewizji, jeśli konkretny user faktycznie tego
  zapyta.
- **Publikacja na PyPI** — nazwa `vechealth` już zarezerwowana, więc
  formalnie wystarczy `maturin publish` (ręcznie albo przez workflow z
  punktu "`maturin publish` workflow" wyżej) w momencie, gdy zdecydujemy się
  wypuścić aktualizację. Nie jest to blokowane niczym technicznym z tej
  listy — czysto kwestia "kiedy jesteśmy gotowi to zrobić".