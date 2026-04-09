using System.Runtime.CompilerServices;

namespace ACommerce.MagneticLM.Graph;

/// <summary>
/// MagneticLM v6: Modified KN + Physics-Based Word Positions + Multi-Force System
///
/// يدمج أفكار من SemanticForce:
/// - كل كلمة لها موقع فيزيائي (x,y,z) يتحرك بقوى حتى يستقر
/// - 5 قوى: سياقية + ترددية + جذب + تنافر + تباطؤ
/// - Cosine Similarity بين المواقع النهائية = تشابه دلالي
/// - المواقع المستقرة = emergent embeddings ديناميكية
/// </summary>
public class WordGraph
{
    // === N-gram (unchanged) ===
    public Dictionary<string, Dictionary<string, double>> NgramCounts { get; } = new();
    public Dictionary<string, double> NgramTotals { get; } = new();
    public Dictionary<string, HashSet<string>> ContinuationContexts { get; } = new();
    public Dictionary<string, int> UniqueFollowers { get; } = new();
    public Dictionary<string, int> NgramsWithCount1 { get; } = new();
    public Dictionary<string, int> NgramsWithCount2 { get; } = new();
    public int TotalUniqueBigrams { get; set; }
    public int MaxNgramOrder { get; set; } = 5;
    public double D1 { get; set; } = 0.5;
    public double D2 { get; set; } = 0.75;
    public double D3Plus { get; set; } = 0.9;

    // === العقد ===
    public Dictionary<string, WordNode> Nodes { get; } = new();
    public long TotalTokens { get; set; }

    // === الطبقة المعنوية (أوزان العلاقات) ===
    public Dictionary<string, Dictionary<string, double>> SemanticEdges { get; } = new();
    public double SemanticThreshold { get; set; } = 0.1;
    public double TransitiveDecay { get; set; } = 0.5;

    // === معاملات القوى الفيزيائية (من SemanticForce) ===
    public float K_context { get; set; } = 2.0f;
    public float K_frequency { get; set; } = 1.5f;
    public float K_attraction { get; set; } = 0.5f;
    public float K_repulsion { get; set; } = 0.3f;
    public float DampingCoeff { get; set; } = 0.15f;
    public float PhysicsLearningRate { get; set; } = 0.02f;
    public float OptimalDistance { get; set; } = 3.0f;
    public float MaxRadius { get; set; } = 15.0f;

    // === Node Importance ===
    private Dictionary<string, double> _importance = new();

    // === Conceptual Circles ===
    private List<HashSet<string>> _circles = new();

    // =========================================================================
    // N-gram registration
    // =========================================================================
    public void AddNgram(string[] context, string nextWord)
    {
        var key = string.Join("|", context);
        if (!NgramCounts.ContainsKey(key)) NgramCounts[key] = new();
        var isNew = !NgramCounts[key].ContainsKey(nextWord) || NgramCounts[key][nextWord] == 0;
        NgramCounts[key].TryGetValue(nextWord, out var current);
        NgramCounts[key][nextWord] = current + 1.0;
        NgramTotals.TryGetValue(key, out var total);
        NgramTotals[key] = total + 1.0;
        var nc = current + 1.0;
        if (nc == 1) { NgramsWithCount1.TryGetValue(key, out var v); NgramsWithCount1[key] = v + 1; }
        if (nc == 2) { NgramsWithCount1[key]--; NgramsWithCount2.TryGetValue(key, out var v); NgramsWithCount2[key] = v + 1; }
        if (nc == 3) { NgramsWithCount2[key]--; }
        if (isNew)
        {
            if (!ContinuationContexts.ContainsKey(nextWord)) ContinuationContexts[nextWord] = new();
            ContinuationContexts[nextWord].Add(key);
            UniqueFollowers.TryGetValue(key, out var uf); UniqueFollowers[key] = uf + 1;
            if (context.Length == 1) TotalUniqueBigrams++;
        }
    }

    public WordNode GetOrCreateNode(string word)
    {
        if (!Nodes.TryGetValue(word, out var node))
        {
            node = new WordNode(word);
            Nodes[word] = node;
        }
        return node;
    }

    // =========================================================================
    // Post-training: Discounts + Physics Simulation + Importance + Circles
    // =========================================================================
    public void BuildPostTrainingStructures(int physicsIterations = 50)
    {
        ComputeOptimalDiscounts();
        Console.Write($"  Running physics simulation ({physicsIterations} iterations)...");
        RunPhysicsSimulation(physicsIterations);
        Console.WriteLine(" done.");
        ComputeNodeImportance();
        DetectConceptualCircles();
    }

    private void ComputeOptimalDiscounts()
    {
        long n1 = 0, n2 = 0, n3 = 0;
        foreach (var (_, counts) in NgramCounts)
            foreach (var (_, c) in counts) { if (c == 1) n1++; else if (c == 2) n2++; else if (c == 3) n3++; }
        if (n1 == 0 || n2 == 0) return;
        var Y = (double)n1 / (n1 + 2.0 * n2);
        D1 = Math.Clamp(1.0 - 2.0 * Y * n2 / n1, 0.1, 0.95);
        D2 = Math.Clamp(2.0 - 3.0 * Y * n3 / n2, 0.1, 0.95);
        D3Plus = Math.Clamp(3.0 - 4.0 * Y * (n3 > 0 ? (double)(n3 + 1) / n3 : 1.0), 0.1, 0.95);
    }

    // =========================================================================
    // المحاكاة الفيزيائية الكاملة (من SemanticForce)
    // كل كلمة تتحرك بـ 5 قوى حتى تستقر في موقعها الدلالي
    // =========================================================================
    private void RunPhysicsSimulation(int iterations)
    {
        var nodeList = Nodes.Values.ToArray();
        var edgeLookup = new Dictionary<string, List<(string Target, double Weight, string Type)>>();

        // بناء فهرس سريع للعلاقات
        foreach (var (from, edges) in SemanticEdges)
        {
            if (!edgeLookup.ContainsKey(from)) edgeLookup[from] = new();
            foreach (var (to, weight) in edges)
            {
                if (Math.Abs(weight) < SemanticThreshold) continue;
                edgeLookup[from].Add((to, weight, weight > 1.0 ? "contextual" : "frequent"));
            }
        }

        for (int iter = 0; iter < iterations; iter++)
        {
            foreach (var node in nodeList)
            {
                var force = CalculateForces(node, nodeList, edgeLookup);

                // قانون نيوتن + تباطؤ
                node.VX += force.fx * PhysicsLearningRate;
                node.VY += force.fy * PhysicsLearningRate;
                node.VZ += force.fz * PhysicsLearningRate;

                node.VX *= (1 - DampingCoeff);
                node.VY *= (1 - DampingCoeff);
                node.VZ *= (1 - DampingCoeff);

                node.PX += node.VX * PhysicsLearningRate;
                node.PY += node.VY * PhysicsLearningRate;
                node.PZ += node.VZ * PhysicsLearningRate;

                // حدود الفضاء
                var mag = MathF.Sqrt(node.PX * node.PX + node.PY * node.PY + node.PZ * node.PZ);
                if (mag > MaxRadius)
                {
                    var scale = MaxRadius / mag;
                    node.PX *= scale; node.PY *= scale; node.PZ *= scale;
                    node.VX *= 0.5f; node.VY *= 0.5f; node.VZ *= 0.5f;
                }
            }
        }
    }

    private (float fx, float fy, float fz) CalculateForces(
        WordNode target, WordNode[] allNodes,
        Dictionary<string, List<(string Target, double Weight, string Type)>> edgeLookup)
    {
        float fx = 0, fy = 0, fz = 0;

        // === قوى من العلاقات المعنوية ===
        if (edgeLookup.TryGetValue(target.Word, out var relations))
        {
            foreach (var (otherWord, weight, type) in relations)
            {
                if (!Nodes.TryGetValue(otherWord, out var other)) continue;

                float dx = other.PX - target.PX;
                float dy = other.PY - target.PY;
                float dz = other.PZ - target.PZ;
                float dist = MathF.Max(MathF.Sqrt(dx * dx + dy * dy + dz * dz), 0.1f);
                float ux = dx / dist, uy = dy / dist, uz = dz / dist;

                float k = type == "contextual" ? K_context : K_frequency;
                float f = k * (float)Math.Min(weight, 10) / dist; // cap weight

                fx += f * ux; fy += f * uy; fz += f * uz;
            }
        }

        // === قوى عامة: تنافر + جذب (عيّنة عشوائية لتوفير الوقت) ===
        var sampleSize = Math.Min(allNodes.Length, 100);
        var step = Math.Max(1, allNodes.Length / sampleSize);

        for (int i = 0; i < allNodes.Length; i += step)
        {
            var other = allNodes[i];
            if (other == target) continue;

            float dx = other.PX - target.PX;
            float dy = other.PY - target.PY;
            float dz = other.PZ - target.PZ;
            float dist = MathF.Max(MathF.Sqrt(dx * dx + dy * dy + dz * dz), 0.1f);
            float ux = dx / dist, uy = dy / dist, uz = dz / dist;

            // تنافر (يمنع التكدس)
            float repF = -K_repulsion / (dist * dist + 1);
            fx += repF * ux; fy += repF * uy; fz += repF * uz;

            // جذب (يحافظ على التماسك)
            if (dist > OptimalDistance)
            {
                float attF = K_attraction * (dist - OptimalDistance) * 0.01f;
                fx += attF * ux; fy += attF * uy; fz += attF * uz;
            }
        }

        // قوة مركزية (تمنع الانجراف)
        fx -= 0.01f * target.PX;
        fy -= 0.01f * target.PY;
        fz -= 0.01f * target.PZ;

        return (fx, fy, fz);
    }

    /// <summary>
    /// Cosine Similarity بين موقعي كلمتين في الفضاء الفيزيائي
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double PositionSimilarity(string word1, string word2)
    {
        if (!Nodes.TryGetValue(word1, out var n1) || !Nodes.TryGetValue(word2, out var n2)) return 0;

        float dot = n1.PX * n2.PX + n1.PY * n2.PY + n1.PZ * n2.PZ;
        float mag1 = MathF.Sqrt(n1.PX * n1.PX + n1.PY * n1.PY + n1.PZ * n1.PZ);
        float mag2 = MathF.Sqrt(n2.PX * n2.PX + n2.PY * n2.PY + n2.PZ * n2.PZ);

        if (mag1 < 0.001f || mag2 < 0.001f) return 0;
        return Math.Clamp(dot / (mag1 * mag2), -1.0, 1.0);
    }

    // =========================================================================
    // Node Importance + Circles
    // =========================================================================
    private void ComputeNodeImportance()
    {
        _importance.Clear();
        foreach (var (word, node) in Nodes)
        {
            var conn = SemanticEdges.TryGetValue(word, out var e) ? e.Count : 0;
            _importance[word] = Math.Log(1 + conn) * Math.Log(1 + node.Frequency);
        }
    }

    public double GetImportance(string w) => _importance.TryGetValue(w, out var v) ? v : 0;

    private void DetectConceptualCircles()
    {
        _circles.Clear();
        var strongThreshold = SemanticThreshold * 3;
        var neighbors = new Dictionary<string, HashSet<string>>();
        foreach (var (word, edges) in SemanticEdges)
            foreach (var (target, weight) in edges)
                if (weight >= strongThreshold &&
                    SemanticEdges.TryGetValue(target, out var rev) &&
                    rev.TryGetValue(word, out var rw) && rw >= strongThreshold)
                {
                    if (!neighbors.ContainsKey(word)) neighbors[word] = new();
                    neighbors[word].Add(target);
                }

        var processed = new HashSet<string>();
        foreach (var (word, nbrs) in neighbors)
        {
            if (processed.Contains(word)) continue;
            var clique = new HashSet<string> { word };
            foreach (var cand in nbrs)
            {
                if (!neighbors.ContainsKey(cand)) continue;
                if (clique.All(m => neighbors.ContainsKey(m) && neighbors[m].Contains(cand)))
                { clique.Add(cand); if (clique.Count >= 5) break; }
            }
            if (clique.Count >= 3) { _circles.Add(clique); foreach (var c in clique) processed.Add(c); }
        }
    }

    public double GetCircleBoost(string w1, string w2)
    {
        foreach (var circle in _circles)
            if (circle.Contains(w1) && circle.Contains(w2)) return 1.5;
        return 1.0;
    }

    // =========================================================================
    // Modified Kneser-Ney (unchanged)
    // =========================================================================
    public double GetInterpolatedProbability(string[] ctx, string word)
        => ModKN(ctx, word, Math.Min(ctx.Length, MaxNgramOrder));

    private double ModKN(string[] ctx, string word, int order)
    {
        if (order == 0) return ContProb(word);
        var start = Math.Max(ctx.Length - order, 0);
        var key = string.Join("|", ctx[start..]);
        if (!NgramTotals.TryGetValue(key, out var total) || total == 0) return ModKN(ctx, word, order - 1);
        NgramCounts[key].TryGetValue(word, out var count);
        var d = count <= 0 ? 0 : count == 1 ? D1 : count == 2 ? D2 : D3Plus;
        var disc = Math.Max(count - d, 0) / total;
        NgramsWithCount1.TryGetValue(key, out var n1);
        NgramsWithCount2.TryGetValue(key, out var n2);
        UniqueFollowers.TryGetValue(key, out var uf);
        var n3p = Math.Max(uf - n1 - n2, 0);
        var lambda = (D1 * n1 + D2 * n2 + D3Plus * n3p) / total;
        return disc + lambda * ModKN(ctx, word, order - 1);
    }

    private double ContProb(string word)
    {
        if (TotalUniqueBigrams == 0) return 1.0 / Math.Max(Nodes.Count, 1);
        if (!ContinuationContexts.TryGetValue(word, out var c)) return 0.5 / TotalUniqueBigrams;
        return (double)c.Count / TotalUniqueBigrams;
    }

    // =========================================================================
    // MagneticLM v6: KN + Physics Position Similarity + Importance + Circles + Cache
    //
    // الاحتمال النهائي = مزج:
    //   (1-λ) * P_KN                           ← n-gram الأساسي
    //   + λ_pos * P_position_similarity          ← تشابه الموقع الفيزيائي
    //   + λ_cache * P_cache                      ← ذاكرة مستمرة
    //
    // كل مكون مضمون أن يكون [0,1] → النتيجة لا تتجاوز 1
    // =========================================================================
    public double GetMagneticProbability(string[] fullContext, string word,
        List<(string Word, string[] Context)>? cacheEntries = null,
        bool isNewSentence = false)
    {
        var knProb = GetInterpolatedProbability(fullContext, word);

        // === 1. Position Similarity: تشابه الموقع الفيزيائي ===
        // كلمات قريبة في الفضاء 3D → احتمال أعلى (حتى بدون علاقة مباشرة)
        double posScore = 0;
        int posCount = 0;
        foreach (var ctx in fullContext)
        {
            var sim = PositionSimilarity(ctx, word);
            if (sim > 0.05) // فقط تشابهات ذات معنى
            {
                // Node Importance تُرجّح المساهمة
                var imp = 1.0 + GetImportance(ctx) * 0.05;
                // Circle Boost
                var boost = GetCircleBoost(ctx, word);
                posScore += sim * imp * boost;
                posCount++;
            }
        }
        // تطبيع ليكون [0, ~0.3]
        var posProb = posCount > 0 ? Math.Min(posScore / (posCount * 3.0), 0.3) : 0;

        // === 2. Continuous Cache with Logarithmic Decay + Dynamic Theta ===
        // مستوحى من اقتراح Gemini:
        // - Logarithmic Decay: الكلمات القديمة = صدى توجيهي وليس إشارة كاملة
        // - Dynamic Theta: الحدة تتناقص مع المسافة لمنع التداخل الموجي
        double cacheProb = 0;
        if (!isNewSentence && cacheEntries != null && cacheEntries.Count > 10)
        {
            double cacheScore = 0, totalCacheWeight = 0;
            int window = 3785;
            int startIdx = Math.Max(0, cacheEntries.Count - window);
            int windowLen = cacheEntries.Count - startIdx;

            for (int ci = startIdx; ci < cacheEntries.Count; ci++)
            {
                var (pastWord, pastContext) = cacheEntries[ci];
                var sim = ContextSimilarity(fullContext, pastContext);
                if (sim <= 0) continue;

                // المسافة من الحاضر (0 = الأحدث)
                int age = cacheEntries.Count - 1 - ci;

                // Logarithmic Decay: وزن = 1 / log(2 + age)
                // الكلمات الأخيرة (age=0): وزن=1.44
                // age=10: وزن=0.40  age=100: وزن=0.22  age=1000: وزن=0.14
                double decay = 1.0 / Math.Log(2.0 + age);

                // Dynamic Theta: الحدة تقل مع المسافة
                // قريب: theta=2.0 (حاد جداً) → بعيد: theta=0.5 (منتشر)
                double dynamicTheta = 2.0 / (1.0 + age * 0.01);
                double w = Math.Pow(sim, dynamicTheta) * decay;

                totalCacheWeight += w;
                if (pastWord == word) cacheScore += w;
            }
            cacheProb = totalCacheWeight > 0 ? cacheScore / totalCacheWeight : 0;
        }

        // === 3. المزج التكيّفي (مضمون [0,1]) ===
        // الأوزان تعتمد على ثقة KN
        double posLambda, cacheLambda;
        if (knProb > 0.05) // KN واثق جداً
        { posLambda = 0.02; cacheLambda = 0.01; }
        else if (knProb > 0.005) // KN متوسط
        { posLambda = 0.06; cacheLambda = 0.03; }
        else // KN ضعيف → الطبقات الأخرى تساهم أكثر
        { posLambda = 0.12; cacheLambda = 0.05; }

        if (isNewSentence || cacheEntries == null || cacheEntries.Count < 10)
            cacheLambda = 0;

        var knLambda = 1.0 - posLambda - cacheLambda;

        var result = knLambda * knProb + posLambda * posProb + cacheLambda * cacheProb;

        // ضمان [floor, 0.999]
        return Math.Clamp(result, 1e-10, 0.999);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ContextSimilarity(string[] ctx1, string[] ctx2)
    {
        if (ctx1.Length == 0 || ctx2.Length == 0) return 0;
        double score = 0, max = 0;
        for (int i = 0; i < ctx1.Length; i++)
        {
            var pw = 1.0 + (double)i / ctx1.Length;
            max += pw;
            for (int j = 0; j < ctx2.Length; j++)
                if (ctx1[i] == ctx2[j]) { score += pw * (i == j ? 1.5 : 1.0); break; }
        }
        return max > 0 ? score / max : 0;
    }

    // === Semantic layer ===
    public void StrengthSemanticEdge(string w1, string w2, double amount)
    {
        if (w1 == w2) return;
        AddSW(w1, w2, amount); AddSW(w2, w1, amount);
    }
    private void AddSW(string a, string b, double amount)
    {
        if (!SemanticEdges.ContainsKey(a)) SemanticEdges[a] = new();
        SemanticEdges[a].TryGetValue(b, out var c); SemanticEdges[a][b] = c + amount;
    }
    public double GetSemanticWeight(string w1, string w2)
    {
        if (!SemanticEdges.TryGetValue(w1, out var e)) return 0;
        e.TryGetValue(w2, out var w); return w;
    }

    public IEnumerable<(string Word, double Weight)> GetSemanticNeighbors(string word)
    {
        if (!SemanticEdges.TryGetValue(word, out var edges)) yield break;
        foreach (var (to, w) in edges) if (Math.Abs(w) >= SemanticThreshold) yield return (to, w);
    }

    public (int Nodes, long Ngrams, int Semantic, int Circles) GetStats()
    {
        var ng = NgramCounts.Values.Sum(e => (long)e.Count);
        var se = SemanticEdges.Values.Sum(e => e.Count);
        return (Nodes.Count, ng, se, _circles.Count);
    }
}

public class WordNode
{
    public string Word { get; }
    public int Frequency { get; set; }
    // موقع فيزيائي 3D
    public float PX, PY, PZ;
    // سرعة
    public float VX, VY, VZ;
    // إثارة مؤقتة (للتوليد)
    public double Excitation { get; set; }
    public double Repulsion { get; set; }

    private static readonly Random _rng = new(42);

    public WordNode(string word)
    {
        Word = word;
        PX = (float)(_rng.NextDouble() * 10 - 5);
        PY = (float)(_rng.NextDouble() * 10 - 5);
        PZ = (float)(_rng.NextDouble() * 10 - 5);
    }
}
