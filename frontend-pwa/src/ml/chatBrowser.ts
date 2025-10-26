import { CreateMLCEngine } from '@mlc-ai/web-llm'

type Msg = { role: 'system'|'user'|'assistant', content: string }

export class BrowserChat {
  private engine: any | null = null

  async init(){
    if (this.engine) return
    // Choose a small model to start; users can change later by modifying this id.
    this.engine = await CreateMLCEngine({
      model: 'Phi-3-mini-4k-instruct-q4f16_1-MLC'
    })
  }

  async ask(messages: Msg[]): Promise<string>{
    if (!this.engine) await this.init()
    const stream = await this.engine.chat.completions.create({ messages, stream: true })
    let out = ''
    for await (const chunk of stream){
      const delta = chunk.choices?.[0]?.delta?.content ?? ''
      if (delta) out += delta
    }
    return out
  }
}