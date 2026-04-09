using ACommerce.MagneticLM.Graph;

namespace ACommerce.MagneticLM.Training;

public class Trainer
{
    private readonly WordGraph _graph;
    public int SentencesTrained { get; private set; }

    public Trainer(WordGraph graph) => _graph = graph;

    public void TrainSentence(string sentence)
    {
        var words = Tokenize(sentence);
        if (words.Length < 2) return;

        // === 1. تسجيل n-grams بكل الأعماق ===
        for (int i = 0; i < words.Length; i++)
        {
            var node = _graph.GetOrCreateNode(words[i]);
            node.Frequency++;
            _graph.TotalTokens++;

            if (i < words.Length - 1)
            {
                var nextWord = words[i + 1];

                // unigram→bigram→trigram→...→MaxNgramOrder
                for (int order = 1; order <= _graph.MaxNgramOrder && order <= i + 1; order++)
                {
                    var context = words[(i + 1 - order)..(i + 1)];
                    _graph.AddNgram(context, nextWord);
                }
            }
        }

        // === 2. اكتشاف معنوي + مكافأة/عقوبة (Reward/Penalty) ===
        // كل جملة تدريب هي أيضاً عملية استدلال:
        //   - نتوقع الكلمة التالية
        //   - إذا نجح التوقع → مكافأة (تقوية الوزن المعنوي)
        //   - إذا فشل → عقوبة (إضعاف الأوزان الخاطئة)
        for (int i = 0; i < words.Length; i++)
        {
            // 2a. أوزان معنوية من نافذة ±2
            for (int d = -2; d <= 2; d++)
            {
                if (d == 0) continue;
                int j = i + d;
                if (j < 0 || j >= words.Length) continue;
                var weight = Math.Abs(d) <= 1 ? 1.0 : _graph.TransitiveDecay;
                _graph.StrengthSemanticEdge(words[i], words[j], weight * 0.1);
            }

            // 2b. مكافأة/عقوبة: هل الطبقة المعنوية تتوقع الكلمة التالية بشكل صحيح؟
            if (i < words.Length - 1)
            {
                var actual = words[i + 1];
                var neighbors = _graph.GetSemanticNeighbors(words[i]).Take(20).ToList();

                foreach (var (neighbor, w) in neighbors)
                {
                    if (neighbor == actual)
                    {
                        // مكافأة: التوقع المعنوي كان صحيحاً!
                        _graph.StrengthSemanticEdge(words[i], actual, 0.05);
                    }
                    else if (w > 0.5)
                    {
                        // عقوبة: وزن معنوي عالي لكلمة لم تظهر فعلاً
                        _graph.StrengthSemanticEdge(words[i], neighbor, -0.02);
                    }
                }
            }
        }

        // الأوزان الضعيفة لا تُحذف - تُتجاهل عند الاستعلام فقط
        // الأوزان السالبة القوية = معرفة طرد مفيدة تبقى

        // === 3. إعادة التقييم العابر ===
        SentencesTrained++;
        if (SentencesTrained % 200 == 0)
            PropagateTransitiveRelations(words);
        if (SentencesTrained % 1000 == 0)
            Console.Write($"\r  Training: {SentencesTrained:N0} sentences...");
    }

    private void PropagateTransitiveRelations(string[] words)
    {
        foreach (var word in words)
        {
            var neighbors = _graph.GetSemanticNeighbors(word)
                .OrderByDescending(n => n.Weight).Take(8).ToList();

            foreach (var (neighbor, w1) in neighbors)
            {
                if (w1 < _graph.SemanticThreshold) continue;

                foreach (var (transitive, w2) in _graph.GetSemanticNeighbors(neighbor)
                    .OrderByDescending(n => n.Weight).Take(8))
                {
                    if (transitive == word || w2 < _graph.SemanticThreshold) continue;

                    var tw = w1 * w2 * _graph.TransitiveDecay * 0.01;
                    if (tw > 0.001)
                        _graph.StrengthSemanticEdge(word, transitive, tw);
                }
            }
        }
    }

    // لا نحذف أوزاناً - السالبة القوية = طرد مفيد، الضعيفة تُتجاهل عند الاستعلام

    public void TrainBatch(IEnumerable<string> sentences)
    {
        foreach (var s in sentences) TrainSentence(s);
        if (SentencesTrained >= 1000) Console.WriteLine();
    }

    public static string[] Tokenize(string sentence)
    {
        return sentence
            .Replace(".", " ").Replace("،", " ").Replace(",", " ")
            .Replace("!", " ").Replace("؟", " ").Replace("?", " ")
            .Replace("\"", " ").Replace("(", " ").Replace(")", " ")
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(w => w.Length > 0)
            .ToArray();
    }
}
