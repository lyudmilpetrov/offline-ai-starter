const CACHE = 'offline-ai-v1'
const ASSETS = ['/', '/index.html', '/src/main.tsx']

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)))
})

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(r => r || fetch(event.request))
  )
})