using ACommerce.MagneticLM.Graph;
using ACommerce.MagneticLM.Training;
using System.Diagnostics;

namespace ACommerce.MagneticLM;

public static class Benchmark
{
    public static void Run(string trainPath, string testPath)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("╔═══════════════════════════════════════════════════╗");
        Console.WriteLine("║  MagneticLM v3: Modified KN + Cache + Semantic   ║");
        Console.WriteLine("╚═══════════════════════════════════════════════════╝\n");

        var trainLines = File.ReadAllLines(trainPath).Where(l => l.Trim().Length > 0).ToArray();
        var testLines = File.ReadAllLines(testPath).Where(l => l.Trim().Length > 0).ToArray();
        Console.WriteLine($"Train: {trainLines.Length:N0} sentences");
        Console.WriteLine($"Test:  {testLines.Length:N0} sentences");

        var graph = new WordGraph { MaxNgramOrder = 5, SemanticThreshold = 0.05, TransitiveDecay = 0.5 };
        var trainer = new Trainer(graph);

        var sw = Stopwatch.StartNew();
        trainer.TrainBatch(trainLines);

        // بناء كل الهياكل بعد التدريب
        graph.BuildPostTrainingStructures();
        sw.Stop();

        var (nodes, ngramEntries, sEdges, circles) = graph.GetStats();
        Console.WriteLine($"\nTraining: {sw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"Graph: {nodes:N0} nodes, {ngramEntries:N0} n-gram, {sEdges:N0} semantic, {circles:N0} circles");
        Console.WriteLine($"Tokens: {graph.TotalTokens:N0}");
        Console.WriteLine($"Discounts: D1={graph.D1:F3} D2={graph.D2:F3} D3+={graph.D3Plus:F3}\n");

        // === Perplexity ===
        Console.Write("Computing...");

        var pplBigram = ComputePerplexity(graph, testLines, mode: "bigram");
        Console.Write($"\r  Bigram:                      PPL = {pplBigram:F1}\n");

        var pplKN = ComputePerplexity(graph, testLines, mode: "kn");
        Console.Write($"  Modified KN-5gram:           PPL = {pplKN:F1}\n");

        var pplCache = ComputePerplexity(graph, testLines, mode: "cache");
        Console.Write($"  Modified KN + Cache:         PPL = {pplCache:F1}\n");

        var pplMagnetic = ComputePerplexity(graph, testLines, mode: "magnetic");
        Console.Write($"  MagneticLM (KN+Cache+Sem):   PPL = {pplMagnetic:F1}\n");

        // === Comparison ===
        Console.WriteLine("\n════════════════════════════════════════════════════");
        Console.WriteLine("  Model                          | Perplexity");
        Console.WriteLine("  ───────────────────────────────┼───────────");
        Console.WriteLine($"  Our Bigram                     | {pplBigram:F1}");
        Console.WriteLine($"  Our Modified KN-5gram          | {pplKN:F1}");
        Console.WriteLine($"  Our KN + Cache                 | {pplCache:F1}");
        Console.WriteLine($"  Our MagneticLM (KN+Cache+Sem)  | {pplMagnetic:F1}");
        Console.WriteLine("  ───────────────────────────────┼───────────");
        Console.WriteLine("  5-gram KN (published)          | ~141");
        Console.WriteLine("  LSTM (Zaremba 2014)            | ~78");
        Console.WriteLine("  AWD-LSTM (Merity 2018)         | ~57");
        Console.WriteLine("  Transformer-XL (Dai 2019)      | ~54");
        Console.WriteLine("════════════════════════════════════════════════════");

        var knImprove = (pplBigram - pplKN) / pplBigram * 100;
        var cacheImprove = (pplKN - pplCache) / pplKN * 100;
        var semImprove = (pplCache - pplMagnetic) / pplCache * 100;
        Console.WriteLine($"\n  KN vs Bigram:      {knImprove:F1}% better");
        Console.WriteLine($"  Cache vs KN:       {cacheImprove:F1}% better");
        Console.WriteLine($"  Semantic vs Cache: {semImprove:F1}% better");
        Console.WriteLine($"  Total vs Bigram:   {(pplBigram - pplMagnetic) / pplBigram * 100:F1}% better");

        if (pplMagnetic < 78) Console.WriteLine("\n  >>> BEAT LSTM! <<<");
        else if (pplMagnetic < 100) Console.WriteLine("\n  Between LSTM and 5-gram KN");
        else if (pplMagnetic < 141) Console.WriteLine("\n  BETTER than 5-gram KN!");
    }

    private static double ComputePerplexity(WordGraph graph, string[] testLines, string mode)
    {
        double totalLogProb = 0;
        int totalTokens = 0;
        double floor = 1e-10;

        // Cache يتراكم عبر الأسطر (مفيد لـ WikiText/مقالات طويلة)
        var cacheEntries = new List<(string Word, string[] Context)>();

        int done = 0;
        foreach (var line in testLines)
        {
            done++;
            if (done % 1000 == 0)
                Console.Write($"\r  [{mode}] {done}/{testLines.Length}...          ");

            var words = Trainer.Tokenize(line);
            if (words.Length < 2) continue;

            bool isNewSentence = (cacheEntries.Count < 20);
            // لا نصفّر الـ cache - يتراكم عبر الجمل (مفيد لـ WikiText)

            for (int i = 1; i < words.Length; i++)
            {
                var currentWord = words[i];
                var contextStart = Math.Max(0, i - graph.MaxNgramOrder);
                var fullContext = words[contextStart..i];

                double prob;
                switch (mode)
                {
                    case "bigram":
                        var biKey = words[i - 1];
                        if (graph.NgramTotals.TryGetValue(biKey, out var biTotal) && biTotal > 0)
                        {
                            graph.NgramCounts[biKey].TryGetValue(currentWord, out var biCount);
                            prob = biCount / biTotal;
                        }
                        else prob = 0;
                        break;

                    case "kn":
                        prob = graph.GetInterpolatedProbability(fullContext, currentWord);
                        break;

                    case "cache":
                        prob = graph.GetMagneticProbability(fullContext, currentWord,
                            cacheEntries, isNewSentence: isNewSentence);
                        break;

                    case "magnetic":
                        prob = graph.GetMagneticProbability(fullContext, currentWord,
                            cacheEntries, isNewSentence: isNewSentence);
                        break;

                    default:
                        prob = 0;
                        break;
                }

                prob = Math.Max(prob, floor);
                totalLogProb += Math.Log(prob);
                totalTokens++;
                cacheEntries.Add((currentWord, fullContext));
                // حد أقصى للـ cache (يحتفظ بآخر 500 فقط)
                if (cacheEntries.Count > 4000)
                    cacheEntries.RemoveRange(0, cacheEntries.Count - 4000);
            }
        }

        return totalTokens > 0 ? Math.Exp(-totalLogProb / totalTokens) : double.MaxValue;
    }
}
