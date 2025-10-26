using Microsoft.Data.Sqlite;
using System.Buffers.Binary;

namespace OfflineAi.Api.Services
{
    public class VectorStore
    {
        private readonly string _dbPath;
        public bool DbReady { get; private set; }

        public VectorStore(ILogger<VectorStore> logger)
        {
            _dbPath = Path.Combine(AppContext.BaseDirectory, "data", "offlineai.db");
            try
            {
                using var conn = new SqliteConnection($"Data Source={_dbPath}");
                conn.Open();
                var cmd = conn.CreateCommand();
                cmd.CommandText = @"
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL,
  source TEXT,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
";
                cmd.ExecuteNonQuery();
                DbReady = true;
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Failed to init SQLite");
                DbReady = false;
            }
        }

        public async Task InsertChunkAsync(string docId, string text, float[] embedding, string source)
        {
            using var conn = new SqliteConnection($"Data Source={_dbPath}");
            await conn.OpenAsync();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "INSERT INTO chunks(doc_id, source, text, embedding) VALUES($d,$s,$t,$e)";
            cmd.Parameters.AddWithValue("$d", docId);
            cmd.Parameters.AddWithValue("$s", source);
            cmd.Parameters.AddWithValue("$t", text);
            cmd.Parameters.AddWithValue("$e", FloatArrayToBlob(embedding));
            await cmd.ExecuteNonQueryAsync();
        }

        public async Task<List<(long Id, string DocId, string Text, double Score)>> FindTopKAsync(float[] query, int k)
        {
            var results = new List<(long, string, string, double)>();
            using var conn = new SqliteConnection($"Data Source={_dbPath}");
            await conn.OpenAsync();
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT id, doc_id, text, embedding FROM chunks";
            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var id = reader.GetInt64(0);
                var doc = reader.GetString(1);
                var text = reader.GetString(2);
                var embBlob = (byte[])reader["embedding"];
                var emb = BlobToFloatArray(embBlob);
                var score = Cosine(query, emb);
                results.Add((id, doc, text, score));
            }
            return results.OrderByDescending(r => r.Item4).Take(k).ToList();
        }

        private static double Cosine(float[] a, float[] b)
        {
            var n = Math.Min(a.Length, b.Length);
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            if (na == 0 || nb == 0) return 0;
            return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        }

        private static byte[] FloatArrayToBlob(float[] arr)
        {
            var b = new byte[arr.Length * 4];
            int o = 0;
            foreach (var f in arr)
            {
                var bytes = BitConverter.GetBytes(f);
                if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
                bytes.CopyTo(b, o);
                o += 4;
            }
            return b;
        }

        private static float[] BlobToFloatArray(byte[] blob)
        {
            var n = blob.Length / 4;
            var arr = new float[n];
            for (int i = 0; i < n; i++)
            {
                var bytes = new byte[4];
                Array.Copy(blob, i * 4, bytes, 0, 4);
                if (!BitConverter.IsLittleEndian) Array.Reverse(bytes);
                arr[i] = BitConverter.ToSingle(bytes, 0);
            }
            return arr;
        }
    }
}