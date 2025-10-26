using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OfflineAi.Api.Services;
using OfflineAi.Api.Models;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// Logging
builder.Host.UseSerilog((ctx, lc) => lc
    .ReadFrom.Configuration(ctx.Configuration)
    .WriteTo.Console());

builder.Services.AddSingleton<EmbeddingService>();
builder.Services.AddSingleton<VectorStore>();

builder.Services.Configure<HostOptions>(opts => {
    opts.BackgroundServiceExceptionBehavior = BackgroundServiceExceptionBehavior.Ignore;
});

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod();
    });
});

var app = builder.Build();
app.UseCors();

var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("Startup");

// Ensure data folder exists
Directory.CreateDirectory(Path.Combine(AppContext.BaseDirectory, "data"));
Directory.CreateDirectory(Path.Combine(AppContext.BaseDirectory, "models", "embeddings"));

app.MapGet("/health", (EmbeddingService embed, VectorStore store) =>
{
    return Results.Ok(new {
        status = "ok",
        modelLoaded = embed.ModelLoaded,
        dbReady = store.DbReady,
    });
});

app.MapPost("/embed", async (EmbedRequest req, EmbeddingService embed) =>
{
    if (req?.Texts == null || req.Texts.Length == 0) return Results.BadRequest(new { error = "No texts" });
    var vectors = new List<float[]>();
    foreach (var t in req.Texts)
    {
        var v = await embed.EmbedAsync(t);
        vectors.Add(v);
    }
    return Results.Ok(new { vectors });
});

app.MapPost("/ingest", async (IngestRequest req, EmbeddingService embed, VectorStore store) =>
{
    if (string.IsNullOrWhiteSpace(req?.DocId) || string.IsNullOrWhiteSpace(req.Text))
        return Results.BadRequest(new { error = "DocId and Text required" });

    var chunkChars = req.ChunkChars.GetValueOrDefault(1200);
    var overlap = req.OverlapChars.GetValueOrDefault(120);

    var chunks = Chunker.Split(req.Text, chunkChars, overlap);
    var count = 0;
    foreach (var ch in chunks)
    {
        var vec = await embed.EmbedAsync(ch);
        await store.InsertChunkAsync(req.DocId, ch, vec, req.Source ?? "user");
        count++;
    }
    return Results.Ok(new { chunks = count });
});

app.MapPost("/search", async (SearchRequest req, EmbeddingService embed, VectorStore store) =>
{
    if (string.IsNullOrWhiteSpace(req?.Query)) return Results.BadRequest(new { error = "Query required" });
    var k = Math.Clamp(req.K.GetValueOrDefault(5), 1, 50);
    var qv = await embed.EmbedAsync(req.Query);
    var hits = await store.FindTopKAsync(qv, k);
    return Results.Ok(new { items = hits.Select(h => new { id = h.Id, docId = h.DocId, text = h.Text, score = h.Score }) });
});

// Placeholder chat endpoint (RAG compose + stub)
app.MapPost("/chat", async (HttpContext ctx, ChatRequest req, EmbeddingService embed, VectorStore store) =>
{
    ctx.Response.Headers.Append("Content-Type", "text/event-stream");
    await ctx.Response.WriteAsync("event: meta\n");
    await ctx.Response.WriteAsync("data: {\"note\":\"This is a stub. Wire up a local LLM (e.g., LLamaSharp) to stream tokens.\"}\n\n");

    // RAG retrieve
    var k = Math.Clamp(req.K.GetValueOrDefault(5), 1, 20);
    var q = req.Messages?.LastOrDefault(m => m.Role == "user")?.Content ?? "";
    if (!string.IsNullOrWhiteSpace(q))
    {
        var qv = await embed.EmbedAsync(q);
        var hits = await store.FindTopKAsync(qv, k);
        foreach (var h in hits)
        {
            var chunk = JsonSerializer.Serialize(new { docId = h.DocId, score = h.Score, text = h.Text });
            await ctx.Response.WriteAsync($"event: context\n");
            await ctx.Response.WriteAsync($"data: {chunk}\n\n");
        }
    }

    // Stream a simple deterministic response
    var answer = "Local chat is not yet wired. Use the browser WebLLM client for generation, or integrate LLamaSharp here.";
    foreach (var token in answer.Split(' '))
    {
        await ctx.Response.WriteAsync($"data: {token} \n\n");
        await ctx.Response.Body.FlushAsync();
        await Task.Delay(5);
    }
    await ctx.Response.WriteAsync("event: done\n");
    await ctx.Response.WriteAsync("data: end\n\n");
    await ctx.Response.Body.FlushAsync();
});

app.Run();

namespace OfflineAi.Api.Models
{
    public record EmbedRequest(string[] Texts);
    public record IngestRequest(string DocId, string Text, int? ChunkChars, int? OverlapChars, string? Source);
    public record SearchRequest(string Query, int? K);
    public record ChatMessage(string Role, string Content);
    public record ChatRequest(List<ChatMessage> Messages, int? K);
}

namespace OfflineAi.Api.Services
{
    public static class Chunker
    {
        public static IEnumerable<string> Split(string text, int chunk, int overlap)
        {
            var i = 0;
            while (i < text.Length)
            {
                var end = Math.Min(i + chunk, text.Length);
                var part = text[i..end];
                // try to break on paragraph/line
                var lastBreak = part.LastIndexOfAny(new[] { '\n', '.', '!' });
                if (lastBreak > chunk * 0.6) part = part[..(lastBreak + 1)];
                yield return part.Trim();
                i += Math.Max(1, part.Length - overlap);
            }
        }
    }
}