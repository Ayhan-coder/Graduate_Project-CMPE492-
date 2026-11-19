/**
 * EduTerm CMPE 492 — Google Forms Otomatik Oluşturucu (Google Apps Script)
 *
 * Kullanım:
 * 1. https://script.google.com adresine gidin ve "New project" (Yeni proje) açın.
 * 2. Bu dosyanın tüm içeriğini yapıştırıp Kaydedin (Ctrl+S).
 * 3. Üstten "createEduTermEvalForm" fonksiyonunu seçip "Run" (Çalıştır) butonuna basın.
 * 4. İzin onayını verin; "Execution log" içinde paylaşılabilir öğrenci linki (Published URL)
 *    ve düzenleme linki (Edit URL) yazdırılacaktır.
 */

function createEduTermEvalForm() {
  var form = FormApp.create("EduTerm CMPE 492 — Öğrenci Değerlendirme Anketi");
  form.setDescription(
    "Boğaziçi Üniversitesi CMPE 492 Bitirme Projesi kapsamında hazırlanan bu anket, " +
    "yapay zekâ destekli Türkçe eğitim açıklamalarının anlaşılırlığını ve Sokratik öğretmen " +
    "diyaloglarının kalitesini değerlendirmeyi amaçlar. Yanıtlarınız tamamen anonimdir (yaklaşık 3–4 dk)."
  );
  form.setProgressBar(true);
  form.setCollectEmail(false);

  // =========================================================================
  // BÖLÜM 1: KATILIMCI BİLGİLERİ & ONAM
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
    .setTitle("4. Etik Onam: Yanıtlarımın anonim olarak yalnızca CMPE 492 akademik değerlendirmesinde kullanılmasını onaylıyorum.")
    .setChoiceValues(["Evet, onaylıyorum"])
    .setRequired(true);

  // =========================================================================
  // BÖLÜM 2: AÇIKLAMA & ANLAŞILIRLIK TESTİ (3 KAVRAM)
  // =========================================================================
  form.addPageBreakItem()
    .setTitle("Bölüm 1: Kavram Açıklamaları ve Anlaşılırlık Testi")
    .setHelpText("Aşağıdaki 3 kısa açıklamayı okuyup anlaşılırlık, Türkçe doğallığı ve kontrol sorusunu yanıtlayınız.");

  var cefrItems = [
    {
      title: "Kavram 1: Tehlike Kavramı",
      text: "Tehlike, işyerinde henüz bir kaza veya yaralanma gerçekleşmemiş olsa bile zarar verme potansiyeli taşıyan kaynak ya da durumdur. Örneğin zemindeki ıslaklık veya koruyucusuz duran bir makine, kimse düşmese veya yaralanmasa dahi bir tehlikedir.",
      quiz: "Kontrol Sorusu 1: Duran ama keskin kenarlı koruyucusuz bir makine tehlike sayılır mı?",
      options: [
        "Evet, zarar verme potansiyeli taşıdığı için tehlikedir",
        "Hayır, ancak biri yaralanırsa tehlike olur",
        "Emin değilim"
      ]
    },
    {
      title: "Kavram 2: KVKK Açık Rıza İstisnaları",
      text: "Kişisel veri işlemek için her zaman açık rıza gerekmez. Kanunlarda açıkça öngörülmesi, bir sözleşmenin kurulması veya ifası için zorunlu olması (örneğin maaş ödemesi için IBAN alınması) gibi hukuki sebepler varsa açık rıza olmadan da veri işlenebilir.",
      quiz: "Kontrol Sorusu 2: Çalışana maaş ödemek için IBAN bilgisini işlerken açık rıza şart mıdır?",
      options: [
        "Hayır, sözleşmenin ifası kapsamında hukuki sebep vardır",
        "Evet, her veri için mutlaka açık rıza gerekir",
        "Emin değilim"
      ]
    },
    {
      title: "Kavram 3: Dinamik Programlama",
      text: "Dinamik programlama, büyük bir problemi alt problemlere bölerken aynı alt problemlerin defalarca tekrarlandığı (örtüşen alt problemler) durumlarda hesaplanan sonuçları bellekte saklayarak tekrar hesaplamayı önleyen optimizasyon yöntemidir.",
      quiz: "Kontrol Sorusu 3: Dinamik programlamayı Böl ve Yönet (Divide & Conquer) yönteminden ayıran temel özellik nedir?",
      options: [
        "Örtüşen alt problemlerin sonuçlarını bellekte saklaması",
        "Problemi hiç bölmemesi",
        "Emin değilim"
      ]
    }
  ];

  cefrItems.forEach(function (item, idx) {
    form.addSectionHeaderItem()
      .setTitle(item.title)
      .setHelpText(item.text);

    form.addScaleItem()
      .setTitle("[" + (idx + 1) + ".1] Bu açıklama ne kadar anlaşılırdı?")
      .setBounds(1, 5)
      .setLabels("Çok zor", "Çok net")
      .setRequired(true);

    form.addScaleItem()
      .setTitle("[" + (idx + 1) + ".2] Türkçe akıcılığı ve doğallığı nasıldı?")
      .setBounds(1, 5)
      .setLabels("Bozuk", "Çok doğal")
      .setRequired(true);

    form.addMultipleChoiceItem()
      .setTitle(item.quiz)
      .setChoiceValues(item.options)
      .setRequired(true);
  });

  // =========================================================================
  // BÖLÜM 3: KÖR A/B SOKRATİK ÖĞRETMEN KARŞILAŞTIRMASI
  // =========================================================================
  form.addPageBreakItem()
    .setTitle("Bölüm 2: Kör A/B Sokratik Öğretmen Karşılaştırması")
    .setHelpText("Her soruda iki farklı yapay zekâ öğretmenin öğrenciyle diyaloğunu okuyup hangisinin kavramı ezberletmeden daha iyi kavrattığını seçiniz.");

  var abPairs = [
    {
      id: "AB_01",
      title: "Karşılaştırma 1: Tehlike vs. Kaza (İSG)",
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
      title: "Karşılaştırma 2: Ramak Kala Olayı",
      // Kör çevrilmiş (flipped): A = onpolicy_dpo_v2, B = task_separated_hybrid
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
      title: "Karşılaştırma 3: KVKK — Anonimleştirme vs. Pseudonimizasyon",
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
    }
  ];

  abPairs.forEach(function (pair, idx) {
    form.addSectionHeaderItem()
      .setTitle(pair.title)
      .setHelpText(pair.dialogA + "\n\n" + pair.dialogB);

    form.addMultipleChoiceItem()
      .setTitle("[" + pair.id + "] Hangi öğretmen kavramı ezberletmeden daha iyi kavrattı?")
      .setChoiceValues(["Öğretmen A", "Öğretmen B", "Eşit"])
      .setRequired(true);

    form.addScaleItem()
      .setTitle("[" + pair.id + "] Seçtiğiniz öğretmenin Sokratik yönlendirmesi nasıldı?")
      .setBounds(1, 5)
      .setLabels("Zayıf", "Mükemmel")
      .setRequired(true);

    form.addTextItem()
      .setTitle("[" + pair.id + "] Kısa Yorum (İsteğe bağlı)")
      .setRequired(false);
  });

  Logger.log("Öğrenci Paylaşım Linki (Published URL): " + form.getPublishedUrl());
  Logger.log("Düzenleme Linki (Edit URL): " + form.getEditUrl());
}
