// Offline export: drives the page headless, renders every frame (with motion blur) and encodes it with ffmpeg.
//   python3 -m http.server 8765 &   (from the repo root, so the fonts load over http)
//   node scripts/render.cjs [--samples 4] [--workers 3] [--fps 60] [--w 1080 --h 1920] [--audio audio/soundtrack.m4a] [--out nvidia-datamotion.mp4] [--from 0] [--to 30]
// Needs playwright (with a Chromium) and ffmpeg with libx264. Each worker renders a contiguous slice of
// frames to its own segment; the segments are joined without re-encoding and the audio is muxed in.
const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
let pw;
try { pw = require('playwright'); } catch { pw = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright'); }

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const samples = +arg('samples', 4), workers = +arg('workers', 3), out = arg('out', 'nvidia-datamotion.mp4');
const audio = arg('audio', 'audio/soundtrack.m4a');
const url = arg('url', 'http://localhost:8765/index.html?export'), fps = +arg('fps', 60);
const VW = +arg('w', 1080), VH = +arg('h', 1920);

async function openPage(browser) {
  const page = await browser.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(url);
  await page.evaluate(() => window.fontsReady);
  const ok = await page.evaluate(() => document.fonts.check('900 100px Archivo') && document.fonts.check("400 20px 'IBM Plex Mono'"));
  if (!ok) throw new Error('fonts did not load: serve the repo over http');
  return page;
}

async function renderSlice(page, f0, f1, file, progress) {
  const ff = spawn('ffmpeg', ['-y', '-v', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', String(fps), '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', file], { stdio: ['pipe', 'inherit', 'inherit'] });
  const canvas = await page.$('canvas');
  for (let f = f0; f < f1; f++) {
    await page.evaluate(([t, s]) => window.renderFrame(t, s), [f / fps, samples]);
    const jpg = await canvas.screenshot({ type: 'jpeg', quality: 96 });
    if (!ff.stdin.write(jpg)) await new Promise((r) => ff.stdin.once('drain', r));
    progress();
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
}

(async () => {
  const browser = await pw.chromium.launch();
  const first = await openPage(browser);
  const dur = await first.evaluate(() => window.DURATION);
  const F0 = Math.round(+arg('from', 0) * fps), F1 = Math.round(+arg('to', dur) * fps);
  const n = F1 - F0, per = Math.ceil(n / workers);
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'render-'));
  const pages = [first];
  for (let i = 1; i < workers; i++) pages.push(await openPage(browser));
  let done = 0;
  const start = Date.now();
  const progress = () => {
    if (++done % 60 === 0) {
      const el = (Date.now() - start) / 1000;
      console.log(`${done}/${n} frames · ${el.toFixed(0)}s elapsed · ~${(el / done * (n - done)).toFixed(0)}s left`);
    }
  };
  const segs = [];
  await Promise.all(pages.map((page, i) => {
    const a = F0 + i * per, b = Math.min(F1, a + per);
    if (a >= b) return null;
    const file = path.join(tmp, `seg${i}.mp4`);
    segs.push(file);
    return renderSlice(page, a, b, file, progress);
  }));
  await browser.close();
  segs.sort();
  const list = path.join(tmp, 'list.txt');
  fs.writeFileSync(list, segs.map((s) => `file '${s}'`).join('\n'));
  const video = path.join(tmp, 'video.mp4');
  execFileSync('ffmpeg', ['-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', video]);
  const mux = audio && fs.existsSync(audio)
    ? ['-i', video, '-ss', String(F0 / fps), '-i', audio, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest']
    : ['-i', video, '-c', 'copy'];
  execFileSync('ffmpeg', ['-y', '-v', 'error', ...mux, '-movflags', '+faststart', out]);
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log('wrote', out, `(${((Date.now() - start) / 1000).toFixed(0)}s)`);
})().catch((e) => { console.error(e); process.exit(1); });
