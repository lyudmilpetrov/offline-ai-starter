import { expose } from 'comlink'
import { pipeline } from '@xenova/transformers'

let extractor: any

export type EmbedWorkerApi = {
  embed: (text: string) => Promise<Float32Array>
}

async function getExtractor(){
  if (!extractor){
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
  }
  return extractor
}

const api: EmbedWorkerApi = {
  async embed(text: string){
    const ex = await getExtractor()
    const out = await ex(text, { pooling: 'mean', normalize: true })
    return out.data as Float32Array
  }
}

expose(api)