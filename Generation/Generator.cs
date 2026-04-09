using ACommerce.MagneticLM.Graph;
using ACommerce.MagneticLM.Training;

namespace ACommerce.MagneticLM.Generation;

/// <summary>
/// محرك التوليد المغناطيسي.
///
/// 1. المطالبة تُثير كلماتها في الرسم البياني (Excitation)
/// 2. من آخر كلمة، المؤشر ينظر لكل الرسم البياني
/// 3. كل عقدة تسحبه أو تدفعه بحسب:
///    - القوة السياقية (هل تأتي بعد الكلمة الحالية عادةً؟)
///    - القوة المعنوية (هل هي مرتبطة بالكلمات المُثارة؟)
///    - قوة الطرد (هل استُخدمت مؤخراً؟)
/// 4. المحصلة الأعلى = الكلمة التالية
/// 5. تحديث الإثارة والطرد، والتكرار
/// </summary>
public class Generator
{
    private readonly WordGraph _graph;

    /// <summary>
    /// α: وزن القوة السياقية (transition probability)
    /// </summary>
    public double ContextualAlpha { get; set; } = 0.6;

    /// <summary>
    /// β: وزن القوة المعنوية (semantic attraction)
    /// </summary>
    public double SemanticBeta { get; set; } = 0.35;

    /// <summary>
    /// γ: معامل اضمحلال الإثارة بعد كل خطوة
    /// </summary>
    public double ExcitationDecay { get; set; } = 0.85;

    /// <summary>
    /// R: قوة الطرد الأولية (لمنع التكرار)
    /// </summary>
    public double RepulsionStrength { get; set; } = 0.5;

    /// <summary>
    /// معامل اضمحلال الطرد
    /// </summary>
    public double RepulsionDecay { get; set; } = 0.5;

    /// <summary>
    /// حرارة العشوائية (0 = deterministic, 1 = random)
    /// </summary>
    public double Temperature { get; set; } = 0.3;

    private readonly Random _rng = new(42);

    public Generator(WordGraph graph)
    {
        _graph = graph;
    }

    /// <summary>
    /// توليد نص من مطالبة
    /// </summary>
    public GenerationResult Generate(string prompt, int maxTokens = 30)
    {
        var result = new GenerationResult { Prompt = prompt };
        var promptWords = Trainer.Tokenize(prompt);
        if (promptWords.Length == 0) return result;

        // === 1. إثارة كلمات المطالبة ===
        ResetExcitations();
        for (int i = 0; i < promptWords.Length; i++)
        {
            var word = promptWords[i];
            if (_graph.Nodes.TryGetValue(word, out var node))
            {
                // الكلمات الأخيرة في المطالبة أكثر إثارة
                node.Excitation = 1.0 * (0.5 + 0.5 * ((double)i / promptWords.Length));
                result.ExcitedWords.Add((word, node.Excitation));

                // إثارة الجيران المعنويين (أضعف)
                foreach (var (neighbor, weight) in _graph.GetSemanticNeighbors(word))
                {
                    if (_graph.Nodes.TryGetValue(neighbor, out var nNode))
                    {
                        nNode.Excitation = Math.Max(nNode.Excitation, weight * 0.3);
                    }
                }
            }
        }

        // === 2. التوليد من آخر كلمة ===
        var currentWord = promptWords[^1];
        var generated = new List<string>();
        var usedWords = new HashSet<string>(promptWords);

        for (int step = 0; step < maxTokens; step++)
        {
            var candidates = ScoreCandidates(currentWord, usedWords);
            if (candidates.Count == 0) break;

            // اختيار الكلمة التالية
            var chosen = SelectWithTemperature(candidates);
            if (chosen == null) break;

            generated.Add(chosen.Word);
            result.Steps.Add(new GenerationStep
            {
                StepNumber = step + 1,
                ChosenWord = chosen.Word,
                ContextualScore = chosen.ContextualScore,
                SemanticScore = chosen.SemanticScore,
                RepulsionScore = chosen.RepulsionScore,
                TotalScore = chosen.TotalScore,
                TopCandidates = candidates.Take(5).Select(c => (c.Word, c.TotalScore)).ToList()
            });

            // === 3. تحديث الحالة ===
            // إثارة الكلمة المختارة
            if (_graph.Nodes.TryGetValue(chosen.Word, out var chosenNode))
            {
                chosenNode.Excitation = 0.8;
                chosenNode.Repulsion = RepulsionStrength;
            }

            // اضمحلال الإثارة لكل العقد
            foreach (var node in _graph.Nodes.Values)
            {
                node.Excitation *= ExcitationDecay;
                node.Repulsion *= RepulsionDecay;
            }

            usedWords.Add(chosen.Word);
            currentWord = chosen.Word;
        }

        result.GeneratedText = string.Join(" ", generated);
        result.FullText = prompt + " " + result.GeneratedText;
        return result;
    }

    /// <summary>
    /// حساب النتيجة لكل كلمة مرشحة - المحصلة المغناطيسية الكاملة
    /// </summary>
    private List<ScoredCandidate> ScoreCandidates(string currentWord, HashSet<string> usedWords)
    {
        var candidates = new List<ScoredCandidate>();

        // كل الكلمات التي لها حافة n-gram من الكلمة الحالية
        var contextKey = currentWord;
        var contextNeighbors = _graph.NgramCounts.TryGetValue(contextKey, out var edges)
            ? edges.Keys.ToList() : new List<string>();

        // إضافة كلمات مُثارة حتى لو ليست جاراً سياقياً مباشراً
        var excitedWords = _graph.Nodes.Values
            .Where(n => n.Excitation > 0.1 && n.Word != currentWord)
            .Select(n => n.Word);

        var allCandidates = contextNeighbors.Union(excitedWords).Distinct();

        foreach (var candidate in allCandidates)
        {
            if (!_graph.Nodes.TryGetValue(candidate, out var node)) continue;

            // القوة السياقية: احتمال n-gram مع backoff
            var contextual = _graph.GetInterpolatedProbability(new[] { currentWord }, candidate);

            // القوة المعنوية: مجموع جذب كل الكلمات المُثارة
            var semantic = 0.0;
            foreach (var excitedNode in _graph.Nodes.Values.Where(n => n.Excitation > 0.05))
            {
                var sWeight = _graph.GetSemanticWeight(excitedNode.Word, candidate);
                semantic += sWeight * excitedNode.Excitation;
            }

            // تطبيع المعنوي
            var maxSemantic = _graph.Nodes.Values.Max(n => n.Excitation);
            if (maxSemantic > 0) semantic /= (maxSemantic * 10);

            // قوة الطرد (سالبة)
            var repulsion = -node.Repulsion;

            // المحصلة
            var total = ContextualAlpha * contextual
                      + SemanticBeta * semantic
                      + repulsion;

            if (total > 0)
            {
                candidates.Add(new ScoredCandidate
                {
                    Word = candidate,
                    ContextualScore = contextual,
                    SemanticScore = semantic,
                    RepulsionScore = repulsion,
                    TotalScore = total
                });
            }
        }

        return candidates.OrderByDescending(c => c.TotalScore).ToList();
    }

    /// <summary>
    /// اختيار مع حرارة: أعلى نتيجة مع بعض العشوائية
    /// </summary>
    private ScoredCandidate? SelectWithTemperature(List<ScoredCandidate> candidates)
    {
        if (candidates.Count == 0) return null;
        if (Temperature <= 0.01) return candidates[0];

        // Softmax مع حرارة
        var maxScore = candidates[0].TotalScore;
        var weights = candidates.Select(c => Math.Exp((c.TotalScore - maxScore) / Temperature)).ToList();
        var totalWeight = weights.Sum();

        var roll = _rng.NextDouble() * totalWeight;
        var cumulative = 0.0;
        for (int i = 0; i < candidates.Count; i++)
        {
            cumulative += weights[i];
            if (roll <= cumulative) return candidates[i];
        }
        return candidates[0];
    }

    private void ResetExcitations()
    {
        foreach (var node in _graph.Nodes.Values)
        {
            node.Excitation = 0;
            node.Repulsion = 0;
        }
    }
}

public class ScoredCandidate
{
    public string Word { get; set; } = default!;
    public double ContextualScore { get; set; }
    public double SemanticScore { get; set; }
    public double RepulsionScore { get; set; }
    public double TotalScore { get; set; }
}

public class GenerationStep
{
    public int StepNumber { get; set; }
    public string ChosenWord { get; set; } = default!;
    public double ContextualScore { get; set; }
    public double SemanticScore { get; set; }
    public double RepulsionScore { get; set; }
    public double TotalScore { get; set; }
    public List<(string Word, double Score)> TopCandidates { get; set; } = new();
}

public class GenerationResult
{
    public string Prompt { get; set; } = default!;
    public string GeneratedText { get; set; } = default!;
    public string FullText { get; set; } = default!;
    public List<GenerationStep> Steps { get; set; } = new();
    public List<(string Word, double Excitation)> ExcitedWords { get; set; } = new();
}
