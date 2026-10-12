import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath } from 'node:url'

export default defineConfig(({ mode }) => {
  const projectRoot = fileURLToPath(new URL('..', import.meta.url))
  const localEnv = loadEnv(mode, projectRoot, '')
  const demoEnabled = mode === 'development'
  const pairs = [
    [localEnv.DEMO_EMAIL, localEnv.DEMO_PASSWORD],
    [localEnv.SEED_ADMIN_EMAIL, localEnv.SEED_ADMIN_PASSWORD],
    [localEnv.ADMIN_EMAIL, localEnv.ADMIN_PASSWORD],
    [localEnv.DEFAULT_EMAIL, localEnv.DEFAULT_PASSWORD],
  ]
  const [demoEmail, demoPassword] = demoEnabled ? pairs.find(([email, password]) => email && password) || ['', ''] : ['', '']
  return {
  plugins: [react(), tailwindcss()],
  define: {
    'import.meta.env.VITE_DEMO_EMAIL': JSON.stringify(demoEmail && demoPassword ? demoEmail : ''),
    'import.meta.env.VITE_DEMO_PASSWORD': JSON.stringify(demoEmail && demoPassword ? demoPassword : ''),
  },
  server: {
    proxy: {
      '/api': {target: process.env.API_PROXY_TARGET || `http://127.0.0.1:${process.env.ADMIN_PORT || '8001'}`, configure(proxy) { proxy.on('proxyReq', (upstream, req, res) => { req.once('aborted', () => upstream.destroy()); res.once('close', () => { if (!('writableEnded' in res) || !res.writableEnded) upstream.destroy(); }); }); },},
    },
  },
  }
})
