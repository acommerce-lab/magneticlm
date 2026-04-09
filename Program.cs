using ACommerce.MagneticLM;
using ACommerce.MagneticLM.Graph;
using ACommerce.MagneticLM.Training;
using ACommerce.MagneticLM.Generation;
using System.Diagnostics;

// === PTB Benchmark mode ===
if (args.Length >= 2 && args[0] == "--benchmark")
{
    var trainPath = args[1];
    var testPath = args.Length >= 3 ? args[2] : args[1].Replace("train", "test");
    Benchmark.Run(trainPath, testPath);
    return;
}


Console.OutputEncoding = System.Text.Encoding.UTF8;
Console.WriteLine("╔═══════════════════════════════════════════╗");
Console.WriteLine("║   MagneticLM - نموذج لغوي مغناطيسي       ║");
Console.WriteLine("║   مبني على AccountingKernel               ║");
Console.WriteLine("╚═══════════════════════════════════════════╝\n");

// === 1. بناء الرسم البياني ===
var graph = new WordGraph
{
    SemanticThreshold = 0.05,
    TransitiveDecay = 0.5
};

var trainer = new Trainer(graph);

// === 2. بيانات التدريب (100 جملة عربية) ===
var trainingData = new[]
{
    // التعليم
    "الطالب ذهب إلى المدرسة في الصباح",
    "المعلم شرح الدرس للطلاب في الفصل",
    "الطالب درس الرياضيات في المكتبة",
    "المعلم أعطى الطالب واجباً صعباً",
    "المدرسة نظمت رحلة للطلاب",
    "الطالب نجح في الامتحان بتفوق",
    "المعلم ساعد الطالب في حل المسائل",
    "الطلاب يحبون معلم الرياضيات",
    "المكتبة مليئة بالكتب المفيدة",
    "الطالب قرأ كتاباً في المكتبة",

    // الطب
    "الطبيب فحص المريض في العيادة",
    "المريض ذهب إلى المستشفى",
    "الطبيب كتب الدواء للمريض",
    "الممرضة ساعدت المريض في الغرفة",
    "المستشفى استقبل حالات كثيرة اليوم",
    "الطبيب أجرى العملية بنجاح",
    "المريض تحسنت حالته بعد العلاج",
    "الطبيب نصح المريض بالراحة",
    "العيادة مفتوحة كل يوم",
    "الدواء يساعد المريض على الشفاء",

    // الطعام
    "الطباخ أعد الطعام في المطبخ",
    "الأم طبخت الأرز مع الدجاج",
    "الطفل أكل الطعام بشهية",
    "المطعم يقدم أطباقاً لذيذة",
    "الطباخ استخدم البهارات في الطبخ",
    "العائلة تناولت العشاء معاً",
    "الطعام الصحي مفيد للجسم",
    "المطبخ نظيف ومرتب",
    "الأم تحب الطبخ كثيراً",
    "الدجاج مع الأرز وجبة مشهورة",

    // التجارة
    "التاجر باع البضاعة في السوق",
    "المحل مليء بالبضائع الجديدة",
    "الزبون اشترى ملابس من المحل",
    "السوق مزدحم في نهاية الأسبوع",
    "التاجر خفض الأسعار لجذب الزبائن",
    "المحل يقدم عروضاً مميزة",
    "الزبون دفع الحساب نقداً",
    "البضاعة وصلت من المصنع",
    "التجارة الإلكترونية تنمو بسرعة",
    "المحل فتح فرعاً جديداً في المدينة",

    // السفر
    "المسافر حجز تذكرة الطائرة",
    "الطائرة أقلعت في الموعد المحدد",
    "الفندق يقع في وسط المدينة",
    "السائح زار المعالم التاريخية",
    "الرحلة كانت ممتعة ومفيدة",
    "المطار مزدحم في الصيف",
    "المسافر وصل إلى المدينة مساءً",
    "الفندق يوفر خدمات ممتازة",
    "السائح التقط صوراً كثيرة",
    "الرحلة استغرقت ساعتين بالطائرة",

    // التكنولوجيا
    "المبرمج كتب الكود على الحاسوب",
    "التطبيق يعمل على الهاتف الذكي",
    "الشركة أطلقت منتجاً جديداً",
    "الإنترنت غير حياة الناس",
    "المبرمج صمم موقعاً إلكترونياً",
    "الذكاء الاصطناعي يتطور بسرعة",
    "الهاتف الذكي أصبح ضرورياً",
    "الشركة توظف مبرمجين محترفين",
    "التطبيق حصل على تقييمات عالية",
    "الحاسوب يساعد في إنجاز العمل",

    // الرياضة
    "اللاعب سجل هدفاً في المباراة",
    "الفريق فاز بالمباراة بثلاثة أهداف",
    "المدرب وضع خطة للمباراة",
    "الجمهور شجع الفريق بحماس",
    "اللاعب تدرب كل يوم في الملعب",
    "المباراة انتهت بالتعادل",
    "الفريق صعد إلى الدوري الممتاز",
    "المدرب اختار اللاعبين بعناية",
    "الملعب امتلأ بالجماهير",
    "اللاعب حصل على جائزة أفضل لاعب",

    // الطبيعة
    "الشمس تشرق في الصباح الباكر",
    "المطر هطل بغزارة على المدينة",
    "الأشجار تزين الحديقة الكبيرة",
    "البحر هادئ في هذا الوقت",
    "الجبال تحيط بالقرية من كل جانب",
    "الحديقة مليئة بالأزهار الجميلة",
    "الطيور تغني في الصباح",
    "النهر يجري بين الجبال",
    "السماء صافية واللون أزرق",
    "الربيع أجمل فصول السنة",

    // العمل
    "الموظف يعمل في المكتب كل يوم",
    "المدير عقد اجتماعاً مع الموظفين",
    "الشركة حققت أرباحاً كبيرة",
    "الموظف أنهى المشروع في الوقت المحدد",
    "المدير كافأ الموظفين المتميزين",
    "العمل الجاد يؤدي إلى النجاح",
    "الشركة وفرت بيئة عمل مريحة",
    "الموظف حصل على ترقية",
    "الاجتماع ناقش خطة العمل الجديدة",
    "المكتب يقع في الطابق العاشر",

    // العائلة
    "الأب يحب أطفاله كثيراً",
    "الأم اهتمت بتربية الأبناء",
    "العائلة سافرت في الإجازة",
    "الأطفال يلعبون في الحديقة",
    "الجد يروي قصصاً للأحفاد",
    "العائلة تجتمع كل أسبوع",
    "الأب عمل بجد لإسعاد عائلته",
    "الأم تساعد الأطفال في الدراسة",
    "الأطفال يحبون الحديقة كثيراً",
    "العائلة احتفلت بالعيد معاً",
};

// === 3. التدريب ===
Console.WriteLine($"بيانات التدريب: {trainingData.Length} جملة");
var sw = Stopwatch.StartNew();

trainer.TrainBatch(trainingData);

sw.Stop();
var (nodes, cEdges, sEdges, _) = graph.GetStats();
Console.WriteLine($"التدريب اكتمل في {sw.ElapsedMilliseconds}ms");
Console.WriteLine($"الرسم البياني: {nodes} عقدة, {cEdges} حافة سياقية, {sEdges} حافة معنوية");
Console.WriteLine();

// === 4. التوليد ===
var generator = new Generator(graph)
{
    ContextualAlpha = 0.6,
    SemanticBeta = 0.35,
    ExcitationDecay = 0.85,
    RepulsionStrength = 0.5,
    Temperature = 0.3
};

var prompts = new[]
{
    "الطالب ذهب إلى",
    "الطبيب فحص",
    "المبرمج كتب",
    "الأم طبخت",
    "التاجر باع",
    "الفريق فاز",
    "المسافر وصل إلى",
    "الموظف أنهى",
    "المعلم شرح",
    "الطفل أكل",
};

Console.WriteLine("════════════════════════════════════════════");
Console.WriteLine("                  التوليد                  ");
Console.WriteLine("════════════════════════════════════════════\n");

foreach (var prompt in prompts)
{
    var result = generator.Generate(prompt, maxTokens: 15);
    Console.WriteLine($"المطالبة: {prompt}");
    Console.WriteLine($"التوليد:  {result.FullText}");

    // عرض أول 3 خطوات بالتفصيل
    if (result.Steps.Count > 0)
    {
        Console.Write("  الخطوات: ");
        foreach (var step in result.Steps.Take(5))
        {
            Console.Write($"[{step.ChosenWord} c:{step.ContextualScore:F2} s:{step.SemanticScore:F2}] ");
        }
        Console.WriteLine();
    }
    Console.WriteLine();
}

// === 5. اختبار العلاقات المعنوية ===
Console.WriteLine("════════════════════════════════════════════");
Console.WriteLine("         العلاقات المعنوية المكتشفة         ");
Console.WriteLine("════════════════════════════════════════════\n");

var testWords = new[] { "المدرسة", "المستشفى", "الطعام", "المبرمج", "الفريق" };
foreach (var word in testWords)
{
    var neighbors = graph.GetSemanticNeighbors(word)
        .OrderByDescending(n => n.Weight)
        .Take(8);
    Console.Write($"{word}: ");
    foreach (var (neighbor, weight) in neighbors)
        Console.Write($"{neighbor}({weight:F2}) ");
    Console.WriteLine("\n");
}

// === 6. مقارنة: مع vs بدون الطبقة المعنوية ===
Console.WriteLine("════════════════════════════════════════════");
Console.WriteLine("   مقارنة: مع vs بدون الطبقة المعنوية      ");
Console.WriteLine("════════════════════════════════════════════\n");

// بدون معنوي (سياقي فقط)
var genNoSemantic = new Generator(graph) { ContextualAlpha = 1.0, SemanticBeta = 0.0, Temperature = 0.2 };
// مع معنوي
var genWithSemantic = new Generator(graph) { ContextualAlpha = 0.6, SemanticBeta = 0.35, Temperature = 0.2 };

var comparisonPrompts = new[] { "المعلم ساعد", "الطبيب نصح", "الشركة أطلقت" };
foreach (var prompt in comparisonPrompts)
{
    var withoutSem = genNoSemantic.Generate(prompt, 12);
    var withSem = genWithSemantic.Generate(prompt, 12);
    Console.WriteLine($"المطالبة: {prompt}");
    Console.WriteLine($"  سياقي فقط: {withoutSem.FullText}");
    Console.WriteLine($"  سياقي+معنوي: {withSem.FullText}");
    Console.WriteLine();
}
