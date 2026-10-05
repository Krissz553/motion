// Dumps a reel's cue times (window.CUES) to JSON, for the soundtrack generator.
//   node scripts/cues.cjs http://localhost:8765/index.html audio/pizza-cues.json
const fs = require('fs');
let pw;
try { pw = require('playwright'); } catch { pw = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright'); }
(async () => {
  const [url, out] = process.argv.slice(2);
  const b = await pw.chromium.launch();
  const p = await b.newPage();
  await p.goto(url + (url.includes('?') ? '&' : '?') + 'export');
  fs.writeFileSync(out, JSON.stringify(await p.evaluate(() => window.CUES()), null, 1));
  await b.close();
  console.log('wrote', out);
})();
