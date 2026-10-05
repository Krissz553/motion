// Offline export: drives the page headless, renders every frame (with motion blur) and pipes PNGs to ffmpeg.
//   python3 -m http.server 8765 &   (from the repo root, so the fonts load over http)
//   node scripts/render.cjs [--samples 4] [--out nvidia-datamotion.mp4] [--from 0] [--to 27]
// Needs playwright (with a Chromium) and ffmpeg with libx264.
const { spawn } = require('child_process');
let pw;
try { pw = require('playwright'); } catch { pw = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright'); }

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const samples = +arg('samples', 4), out = arg('out', 'nvidia-datamotion.mp4');
const url = arg('url', 'http://localhost:8765/index.html?export'), fps = 60;

(async () => {
  const browser = await pw.chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(url);
  await page.evaluate(() => window.fontsReady);
  const ok = await page.evaluate(() => document.fonts.check("900 100px Archivo") && document.fonts.check("400 20px 'IBM Plex Mono'"));
  if (!ok) throw new Error('fonts did not load — serve the repo over http');
  const dur = await page.evaluate(() => window.DURATION);
  const t0 = +arg('from', 0), t1 = +arg('to', dur);
  const n = Math.round((t1 - t0) * fps);
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const canvas = await page.$('canvas');
  const start = Date.now();
  for (let i = 0; i < n; i++) {
    const t = t0 + i / fps;
    await page.evaluate(([t, s]) => window.renderFrame(t, s), [t, samples]);
    const png = await canvas.screenshot({ type: 'png' });
    if (!ff.stdin.write(png)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 60 === 0) console.log(`frame ${i}/${n}  t=${t.toFixed(2)}s  ${((Date.now() - start) / 1000).toFixed(0)}s elapsed`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  await browser.close();
  console.log('wrote', out);
})().catch((e) => { console.error(e); process.exit(1); });
