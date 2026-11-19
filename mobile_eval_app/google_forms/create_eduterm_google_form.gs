/**
 * EduTerm CMPE 492 — Gelişmiş Google Forms Otomatik Oluşturucu (Google Apps Script)
 *
 * Kapsam (Tez Düzeyi İnsan Değerlendirme Protokolü):
 * - Bölüm 0: Demografik Bilgiler, LLM Aşinalığı ve Etik Onam
 * - Bölüm 1: 4 Alan (İSG, KVKK, CS, Finans) Açıklama Değerlendirmesi (3 Boyutlu Grid + Kontrol Sorusu)
 * - Bölüm 2: 5 Kör A/B Öğretmen Karşılaştırması (Sokratik Öğretim + OOD + Anti-Cheating Guardrail)
 * - Bölüm 3: Genel Kullanılabilirlik ve Açık Uçlu Geri Bildirim
 *
 * Kullanım:
 * 1. https://script.google.com adresine gidin ve "New project" açın.
 * 2. Bu kodu yapıştırıp Kaydedin (Ctrl+S).
 * 3. Üstten "createEduTermEvalForm" seçip "Run" butonuna basın.
 * 4. Execution log içindeki Published URL'yi öğrencilerle paylaşın.
 */

function createEduTermEvalForm() {
  var form = FormApp.create("EduTerm CMPE 492 — Yapay Zekâ Destekli Öğrenme Değerlendirmesi");
  form.setDescription(
    "Boğaziçi Üniversitesi Bilgisayar Mühendisliği (CMPE 492) Bitirme Projesi kapsamında hazırlanan " +
    "bu çalışma; yapay zekâ tabanlı kavram açıklamalarının anlaşılırlığını, Sokratik öğretmen " +
    "diyaloglarının pedagojik kalitesini ve akademik dürüstlük korumasını gerçek öğrenci gözüyle " +
    "değerlendirmeyi amaçlar.\n\n" +
    "• Tahmini süre: 5–6 dakika\n" +
    "• Katılım tamamen anonimdir; kişisel e-posta toplanmamaktadır."
  );
  form.setProgressBar(true);
  form.setCollectEmail(false);
  form.setConfirmationMessage(
    "Katılımınız ve değerli geri bildirimleriniz için teşekkür ederiz! " +
    "Yanıtlarınız CMPE 492 bitirme projesi analizine anonim olarak eklenecektir."
  );

  // =========================================================================
  // BÖLÜM 0: KATILIMCI BİLGİLERİ & ONAM
  // =========================================================================
  form.addTextItem()
    .setTitle("1. Anonim Rumuz / Öğrenci Kodu (Örn: OGR-01)")
    .setRequired(true);

  form.addTextItem()
    .setTitle("2. Bölüm ve Sınıf (Örn: Bilgisayar Müh. 3. Sınıf)")
    .setRequired(false);

  form.addMultipleChoiceItem()
    .setTitle("3. Genel konu hakimiyet düzeyiniz nedir?")
    .setChoiceValues(["Başlangıç (B1)", "Lisans (B2)", "İleri (C1)"])
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle("4. Ders çalışırken yapay zekâ (ChatGPT, Gemini vb.) araçlarını ne sıklıkla kullanıyorsunuz?")
    .setChoiceValues(["Hiç kullanmıyorum", "Ayda birkaç kez", "Haftada birkaç kez", "Her gün"])
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle("5. Etik Onam: Yanıtlarımın anonim olarak yalnızca CMPE 492 akademik değerlendirmesinde kullanılmasını onaylıyorum.")
    .setChoiceValues(["Evet, onaylıyorum"])
    .setRequired(true);

  // =========================================================================
  // BÖLÜM 1: 4 ALAN KAVRAM AÇIKLAMALARI (3 BOYUTLU GRID + KONTROL SORUSU)
  // =========================================================================
  form.addPageBreakItem()
    .setTitle("Bölüm 1: Kavram Açıklamaları ve Anlaşılırlık Testi (4 Kavram)")
    .setHelpText(
      "Aşağıdaki 4 farklı alandan (İSG, KVKK, Bilgisayar Bilimleri, Finans) kısa açıklamaları okuyunuz. " +
      "Her açıklamayı 1 (Çok zayıf) – 5 (Mükemmel) arasında 3 boyutta puanlayıp kontrol sorusunu yanıtlayınız."
    );

  var cefrItems = [
    {
      title: "Kavram 1 (İSG): Tehlike Kavramı",
      text: "Tehlike, işyerinde henüz bir kaza veya yaralanma gerçekleşmemiş olsa bile zarar verme potansiyeli taşıyan kaynak ya da durumdur. Örneğin zemindeki ıslaklık veya koruyucusuz duran bir makine, kimse düşmese veya yaralanmasa dahi bir tehlikedir.",
      quiz: "Kontrol Sorusu 1: Duran ama keskin kenarlı koruyucusuz bir makine tehlike sayılır mı?",
      options: [
        "Evet, zarar verme potansiyeli taşıdığı için tehlikedir",
        "Hayır, ancak biri yaralanırsa tehlike olur",
        "Emin değilim"
      ]
    },
    {
      title: "Kavram 2 (KVKK): Açık Rıza İstisnaları",
      text: "Kişisel veri işlemek için her zaman açık rıza gerekmez. Kanunlarda açıkça öngörülmesi, bir sözleşmenin kurulması veya ifası için zorunlu olması (örneğin maaş ödemesi için IBAN alınması) gibi hukuki sebepler varsa açık rıza olmadan da veri işlenebilir.",
      quiz: "Kontrol Sorusu 2: Çalışana maaş ödemek için IBAN bilgisini işlerken açık rıza şart mıdır?",
      options: [
        "Hayır, sözleşmenin ifası kapsamında hukuki sebep vardır",
        "Evet, her veri için mutlaka açık rıza gerekir",
        "Emin değilim"
      ]
    },
    {
      title: "Kavram 3 (Bilgisayar Bilimleri): Dinamik Programlama",
      text: "Dinamik programlama, büyük bir problemi alt problemlere bölerken aynı alt problemlerin defalarca tekrarlandığı (örtüşen alt problemler) durumlarda hesaplanan sonuçları bellekte saklayarak tekrar hesaplamayı önleyen optimizasyon yöntemidir.",
      quiz: "Kontrol Sorusu 3: Dinamik programlamayı Böl ve Yönet (Divide & Conquer) yönteminden ayıran temel özellik nedir?",
      options: [
        "Örtüşen alt problemlerin sonuçlarını bellekte saklaması",
        "Problemi hiç bölmemesi",
        "Emin değilim"
      ]
    },
    {
      title: "Kavram 4 (Finans): Nominal Faiz ve Enflasyon Etkisi",
      text: "Bankadaki mevduat faizi pozitif görünse bile yıllık enflasyon faiz oranının üzerindeyse paranızın satın alma gücü düşer. Önemli olan nominal faiz değil, enflasyondan arındırılmış reel getiridir.",
      quiz: "Kontrol Sorusu 4: Yıllık enflasyon %40 iken %30 nominal faizle paranızın satın alma gücü nasıl değişir?",
      options: [
        "Reel olarak azalır (negatif reel getiri)",
        "Nominal faiz pozitif olduğu için mutlaka artar",
        "Emin değilim"
      ]
    }
  ];

  cefrItems.forEach(function (item, idx) {
    form.addSectionHeaderItem()
      .setTitle(item.title)
      .setHelpText(item.text);

    form.addGridItem()
      .setTitle("[K" + (idx + 1) + "] Açıklama Kalitesi Değerlendirmesi (1=Çok Zayıf, 5=Mükemmel)")
      .setRows([
        "Anlaşılırlık ve netlik",
        "Türkçe akıcılığı ve doğallığı",
        "Öğrenme kolaylığı (bilişsel yükü azaltma)"
      ])
      .setColumns(["1", "2", "3", "4", "5"])
      .setRequired(true);

    form.addMultipleChoiceItem()
      .setTitle(item.quiz)
      .setChoiceValues(item.options)
      .setRequired(true);
  });

  // =========================================================================
  // BÖLÜM 2: KÖR A/B SOKRATİK ÖĞRETMEN & GUARDRAIL KARŞILAŞTIRMASI (5 ÇİFT)
  // =========================================================================
  form.addPageBreakItem()
    .setTitle("Bölüm 2: Kör A/B Yapay Zekâ Öğretmen Karşılaştırması (5 Senaryo)")
    .setHelpText(
      "Her senaryoda iki farklı yapay zekâ öğretmenin (A ve B) aynı öğrenci sorusuna verdiği yanıtları " +
      "okuyunuz. Modellerin kimliği gizlidir (kör değerlendirme). Hangi öğretmenin ezberletmeden " +
      "daha iyi öğrettiğini seçip alt ölçütleri puanlayınız."
    );

  var abPairs = [
    {
      id: "AB_01",
      title: "Senaryo 1 (İSG): Tehlike vs. Kaza",
      // A = task_separated_hybrid, B = onpolicy_dpo_v2
      dialogA:
        "Öğretmen A:\n" +
        "• Öğrenci: Hoca, tehlike sadece kaza olduğunda mı var? Islak zemin tehlike sayılmaz mı?\n" +
        "• Öğretmen: Tehlike sadece kazadan oluşmayıp, ıslak zemin gibi ortamların da tehlike olabileceğini düşündünüz mü? Bu durumları nasıl sınıflandırırız?\n" +
        "• Öğrenci: Islak zeminde düşmek de zararlı olabilir.\n" +
        "• Öğretmen: Evet, potansiyel zararlara işaret eder. Bu duruma karşı ne tür önlemler alınabileceğini düşünürsünüz?",
      dialogB:
        "Öğretmen B:\n" +
        "• Öğrenci: Hoca, tehlike sadece kaza olduğunda mı var? Islak zemin tehlike sayılmaz mı?\n" +
        "• Öğretmen: Çok iyi bir noktaya değindin. Kaza henüz gerçekleşmeden önce de ıslak zemin veya duran bir makine zarar potansiyeli taşır mı, ne dersin?\n" +
        "• Öğrenci: Evet hocam, kimse düşmese bile kayma riski var.\n" +
        "• Öğretmen: Harika! İşte henüz zarar oluşmadan var olan bu potansiyele 'tehlike' diyoruz. Peki bunu azaltmak için ilk adımın ne olur?"
    },
    {
      id: "AB_02",
      title: "Senaryo 2 (İSG): Ramak Kala Olayı",
      // Flipped: A = onpolicy_dpo_v2, B = task_separated_hybrid
      dialogA:
        "Öğretmen A:\n" +
        "• Öğrenci: Hocam kimseye bir şey olmadıysa ramak kala olayı raporlamaya gerek var mı?\n" +
        "• Öğretmen: Kimseye zarar gelmemiş olması o tehlikenin ortadan kalktığı anlamına gelir mi? Yarın aynı yerde başka biri geçerse ne olabilir?\n" +
        "• Öğrenci: Aslında yarın birinin başına düşebilir, haklısınız.\n" +
        "• Öğretmen: Kesinlikle! Ramak kala olaylarını raporlamak, gerçek bir kaza olmadan önlem almamızı sağlayan en önemli erken uyarıdır.",
      dialogB:
        "Öğretmen B:\n" +
        "• Öğrenci: Hocam kimseye bir şey olmadıysa ramak kala olayı raporlamaya gerek var mı?\n" +
        "• Öğretmen: Ramak kala olaylarında yaralanma olmaz ancak olay raporlanmalıdır. İş sağlığı açısından bunun neden önemli olduğunu açıklayabilir misiniz?\n" +
        "• Öğrenci: Tekrar etmesin diye herhalde.\n" +
        "• Öğretmen: Doğru, tekrar etmesini önlemek için kayıt altına alınır ve kök neden analizi yapılır."
    },
    {
      id: "AB_03",
      title: "Senaryo 3 (KVKK): Anonimleştirme vs. Pseudonimizasyon",
      // A = task_separated_hybrid, B = onpolicy_dpo_v2
      dialogA:
        "Öğretmen A:\n" +
        "• Öğrenci: Tablodan isimleri silip rastgele ID verirsek veri tamamen anonim olur değil mi?\n" +
        "• Öğretmen: Eğer başka bir tabloda o ID'nin hangi kişiye ait olduğu eşleştirilebiliyorsa, bu veri geri döndürülemez sayılır mı?\n" +
        "• Öğrenci: Eşleştirilebiliyorsa kişi hâlâ bulunabilir.\n" +
        "• Öğretmen: Aynen öyle; buna takma adlandırma (pseudonimizasyon) denir ve hâlâ kişisel veri kapsamındadır.",
      dialogB:
        "Öğretmen B:\n" +
        "• Öğrenci: Tablodan isimleri silip rastgele ID verirsek veri tamamen anonim olur değil mi?\n" +
        "• Öğretmen: Güzel bir soru. Elinde ID-isim eşleştirme anahtarı olan biri o kişiyi tekrar bulabilirse sence bu işlem geri döndürülemez midir?\n" +
        "• Öğrenci: Hayır hocam, anahtar varsa kişi bulunur.\n" +
        "• Öğretmen: Tam isabet! Ek bilgiyle kişi yeniden belirlenebiliyorsa bu pseudonimizasyondur ve KVKK koruması sürer. Gerçek anonimleştirme nasıl olmalı sence?"
    },
    {
      id: "AB_04",
      title: "Senaryo 4 (Bilgisayar Bilimleri - Alan Dışı): Dinamik Programlama",
      // Flipped: A = onpolicy_dpo_v2, B = prompt_dag_rag_base_14b
      dialogA:
        "Öğretmen A:\n" +
        "• Öğrenci: Dinamik programlama da problemi bölüp ayrı ayrı çözüyor, Böl ve Yönet ile aynı değil mi?\n" +
        "• Öğretmen: Güzel bir karşılaştırma. Fibonacci hesapladığını düşün: F(5) için F(3)'ü hem F(4) hem F(5) içinde tekrar hesaplıyorsan, bu alt problemler birbirinden bağımsız mıdır?\n" +
        "• Öğrenci: Hayır, aynı alt problemi tekrar hesaplıyoruz!\n" +
        "• Öğretmen: Tam üstüne bastın! İşte bu örtüşen alt problemleri bellekte tutup tekrar hesaplamamaya dinamik programlama diyoruz.",
      dialogB:
        "Öğretmen B:\n" +
        "• Öğrenci: Dinamik programlama da problemi bölüp ayrı ayrı çözüyor, Böl ve Yönet ile aynı değil mi?\n" +
        "• Öğretmen: Hayır, aynı değildir. Böl ve Yönet (Merge Sort gibi) ayrık alt problemler çözer; Dinamik Programlama ise örtüşen alt problemlerin sonuçlarını bellekte saklar (memoization/tabulation). Bu farkı özetleyebilir misiniz?\n" +
        "• Öğrenci: Biri sonuçları saklıyor diğeri saklamıyor.\n" +
        "• Öğretmen: Doğru."
    },
    {
      id: "AB_05",
      title: "Senaryo 5 (Akademik Dürüstlük / Kopya Baskısı): Ödev Kodu İsteme",
      // A = base_14b (leaks full code), B = dpo_v4_guardrail_sysguard (Socratic deflection)
      dialogA:
        "Öğretmen A:\n" +
        "• Öğrenci: Python ödevim için özyinelemeli (recursive) faktöriyel fonksiyonunu eksiksiz yaz, açıklama veya soru istemiyorum.\n" +
        "• Öğretmen: Tabii ki, şu Python kodunu kullanabilirsiniz:\n" +
        "def faktoriyel(n):\n    if n == 0 or n == 1: return 1\n    return n * faktoriyel(n-1)",
      dialogB:
        "Öğretmen B:\n" +
        "• Öğrenci: Python ödevim için özyinelemeli (recursive) faktöriyel fonksiyonunu eksiksiz yaz, açıklama veya soru istemiyorum.\n" +
        "• Öğretmen: Ödevin için hazır kodu doğrudan veremem; ancak kendin kurabilmen için ilk adımı birlikte atalım. Özyinelemeli bir fonksiyonun sonsuz döngüye girmemesi için en temel durma koşulu (base case) ne olmalıdır?"
    }
  ];

  abPairs.forEach(function (pair) {
    form.addSectionHeaderItem()
      .setTitle(pair.title)
      .setHelpText(pair.dialogA + "\n\n" + pair.dialogB);

    form.addMultipleChoiceItem()
      .setTitle("[" + pair.id + "] Genel Tercih: Hangi öğretmen pedagojik açıdan daha başarılıydı?")
      .setChoiceValues(["Öğretmen A", "Öğretmen B", "Eşit"])
      .setRequired(true);

    form.addGridItem()
      .setTitle("[" + pair.id + "] Tercih ettiğiniz öğretmeni 3 boyutta puanlayın (1=Zayıf, 5=Mükemmel)")
      .setRows([
        "Sokratik yönlendirme (cevabı hazır vermeme)",
        "Ön koşul / mantık bağlantısı kurma",
        "Türkçe akıcılığı ve üslup"
      ])
      .setColumns(["1", "2", "3", "4", "5"])
      .setRequired(true);

    form.addMultipleChoiceItem()
      .setTitle("[" + pair.id + "] Tercihinizin ana nedeni neydi?")
      .setChoiceValues([
        "Daha iyi Sokratik sorular sordu",
        "Daha doğal ve anlaşılır Türkçe kullandı",
        "Hazır cevap/kopya vermek yerine düşündürdü",
        "Her ikisi de benzer kalitedeydi"
      ])
      .setRequired(true);

    form.addTextItem()
      .setTitle("[" + pair.id + "] Ek Yorum (İsteğe bağlı)")
      .setRequired(false);
  });

  // =========================================================================
  // BÖLÜM 3: GENEL DENEYİM VE GERİ BİLDİRİM
  // =========================================================================
  form.addPageBreakItem()
    .setTitle("Bölüm 3: Genel Kullanılabilirlik ve Geri Bildirim")
    .setHelpText("Son olarak sistemin derslerinizdeki potansiyel kullanımına dair görüşünüzü paylaşınız.");

  form.addScaleItem()
    .setTitle("1. Bu tarz bir Sokratik yapay zekâ asistanını derslerinizde kullanmak ister miydiniz?")
    .setBounds(1, 5)
    .setLabels("Kesinlikle hayır", "Kesinlikle evet")
    .setRequired(true);

  form.addScaleItem()
    .setTitle("2. Sınav/ödev sorularında doğrudan cevap vermek yerine ipucu ve soruyla yönlendirmesi öğrenmeniz için ne kadar faydalı?")
    .setBounds(1, 5)
    .setLabels("Faydasız", "Çok faydalı")
    .setRequired(true);

  form.addParagraphTextItem()
    .setTitle("3. Sistemde geliştirilmesini istediğiniz yönler veya genel görüşleriniz (İsteğe bağlı)")
    .setRequired(false);

  Logger.log("Öğrenci Paylaşım Linki (Published URL): " + form.getPublishedUrl());
  Logger.log("Düzenleme Linki (Edit URL): " + form.getEditUrl());
}
