import Dexie, { Table } from 'dexie'
import { wrap } from 'comlink'
import type { EmbedWorkerApi } from '../workers/embed.worker'

export interface Doc { id: string; source: string; createdAt: number }
export interface Chunk { id?: number; docId: string; text: string; embedding: ArrayBuffer }

export class DB extends Dexie {
  docs!: Table<Doc, string>
  chunks!: Table<Chunk, number>
  constructor() {
    super('local_rag')
    this.version(1).stores({
      docs: 'id, createdAt',
      chunks: '++id, docId'
    })
  }
}
export const db = new DB()

// Worker
const worker = new Worker(new URL('../workers/embed.worker.ts', import.meta.url), { type: 'module' })
const api = wrap<EmbedWorkerApi>(worker)

export async function addDocument({ id, text, source }:{ id:string, text:string, source:string }){
  await db.transaction('rw', db.docs, db.chunks, async () => {
    await db.docs.put({ id, source, createdAt: Date.now() })
    const chunks = split(text, 800, 120)
    for (const ch of chunks) {
      const vec = await api.embed(ch) // Float32Array
      await db.chunks.add({ docId: id, text: ch, embedding: vec.buffer.slice(0) })
    }
  })
}

export async function searchTopK(query: string, k: number){
  const qv = await api.embed(query)
  const all = await db.chunks.toArray()
  const results = all.map(ch => {
    const sim = cosine(new Float32Array(ch.embedding), qv)
    return { id: ch.id!, docId: ch.docId, text: ch.text, score: sim }
  }).sort((a,b) => b.score - a.score).slice(0, k)
  return results
}

function split(text: string, chunk: number, overlap: number){
  const out: string[] = []
  let i = 0
  while (i < text.length){
    let end = Math.min(i + chunk, text.length)
    let part = text.slice(i, end)
    const last = Math.max(part.lastIndexOf('\n'), part.lastIndexOf('.'))
    if (last > chunk * 0.6) part = part.slice(0, last + 1)
    out.push(part.trim())
    i += Math.max(1, part.length - overlap)
  }
  return out
}

function cosine(a: Float32Array, b: Float32Array){
  const n = Math.min(a.length, b.length)
  let dot = 0, na=0, nb=0
  for (let i=0;i<n;i++){ const x=a[i], y=b[i]; dot+=x*y; na+=x*x; nb+=y*y }
  return (na===0||nb===0)?0: dot / Math.sqrt(na*nb)
}