using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System.Collections.Concurrent;

namespace OfflineAi.Api.Services
{
    /// <summary>
    /// Embeddings via ONNX Runtime with a BERT-style encoder (e.g., all-MiniLM-L6-v2).
    /// Place your model at: ./models/embeddings/model.onnx and tokenizer.json (HF tokenizer JSON).
    /// </summary>
    public class EmbeddingService
    {
        private readonly InferenceSession? _session;
        private readonly Tokenizer? _tokenizer;
        private readonly int _maxLen = 256;
        private readonly int _hidden;
        public bool ModelLoaded => _session != null && _tokenizer != null;

        public EmbeddingService(ILogger<EmbeddingService> logger)
        {
            var baseDir = AppContext.BaseDirectory;
            var modelPath = Path.Combine(baseDir, "models", "embeddings", "model.onnx");
            var tokPath = Path.Combine(baseDir, "models", "embeddings", "tokenizer.json");
            if (File.Exists(modelPath) && File.Exists(tokPath))
            {
                // Load tokenizer
                using var fs = File.OpenRead(tokPath);
                _tokenizer = Tokenizer.Create(fs);

                // Load ONNX
                var opts = new SessionOptions();
                // CPU EP by default; add DirectML or CUDA EPs here if installed
                _session = new InferenceSession(modelPath, opts);

                // Try to infer hidden size from the model output shape
                try
                {
                    var out0 = _session.OutputMetadata.First().Value.Dimensions;
                    _hidden = (int)(out0.Last() ?? 384);
                }
                catch { _hidden = 384; }
            }
            else
            {
                // Not fatal; service will return errors on use
            }
        }

        public async Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
        {
            if (_session == null || _tokenizer == null)
                throw new InvalidOperationException("Embedding model/tokenizer not found. Put model.onnx and tokenizer.json under ./models/embeddings/");

            // Tokenize
            var enc = _tokenizer.Encode(text);
            var ids = enc.Ids.Select(i => (long)i).ToList();
            // Truncate + add special tokens if needed
            if (ids.Count > _maxLen) ids = ids.Take(_maxLen).ToList();
            var padLen = _maxLen - ids.Count;
            var attn = Enumerable.Repeat(1L, ids.Count).ToList();
            if (padLen > 0)
            {
                ids.AddRange(Enumerable.Repeat(0L, padLen));        // [PAD]=0 in BERT vocab
                attn.AddRange(Enumerable.Repeat(0L, padLen));
            }

            var idsTensor = new DenseTensor<long>(ids.ToArray(), new[] { 1, _maxLen });
            var attnTensor = new DenseTensor<long>(attn.ToArray(), new[] { 1, _maxLen });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", idsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attnTensor)
            };

            // Some models use token_type_ids
            if (_session.InputMetadata.ContainsKey("token_type_ids"))
            {
                var tt = new DenseTensor<long>(new long[_maxLen], new[] { 1, _maxLen });
                inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tt));
            }

            // Run
            var results = _session.Run(inputs);
            // Find 3D output [1, seq, hidden]
            DenseTensor<float>? last = null;
            foreach (var r in results)
            {
                if (r.AsTensor<float>().Rank == 3)
                {
                    last = (DenseTensor<float>)r.AsTensor<float>();
                    break;
                }
            }
            last ??= (DenseTensor<float>)results.First().AsTensor<float>();

            // Mean pool using attention mask
            var seq = last.Dimensions[1];
            var hid = last.Dimensions[2];
            Span<float> sum = new float[hid];
            var valid = 0;
            for (int t = 0; t < seq; t++)
            {
                if (attn[t] == 0) continue;
                valid++;
                for (int h = 0; h < hid; h++)
                    sum[h] += last[0, t, h];
            }
            if (valid == 0) valid = 1;
            var vec = sum.ToArray();
            for (int h = 0; h < hid; h++) vec[h] /= valid;

            // L2 normalize
            var norm = MathF.Sqrt(vec.Sum(x => x * x));
            if (norm > 0)
            {
                for (int h = 0; h < hid; h++) vec[h] /= norm;
            }
            return await Task.FromResult(vec);
        }
    }
}