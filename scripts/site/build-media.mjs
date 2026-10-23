// Builds the site's videos from their edits: each media/<name>/<Name>.vid.py (a video in code, CommandAGI's
// SDK) renders to site/media/<name>.mp4, with its poster, site/media/<name>.jpg, taken at the edit's
// "poster" marker (or its first second).
//
//   node scripts/site/build-media.mjs            # every edit
//   node scripts/site/build-media.mjs robots     # one
//
// It needs a CommandAGI checkout (COMMANDAGI_HOME, default ~/Documents/commandagi) with its headless
// renderer installed once: `npm install` in packages/applets/video/eval. The ffmpeg is FFMPEG_PATH, or the one
// imageio-ffmpeg ships in tensacode/python/.venv; the renderer needs one with lavfi.
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '../..');
const commandagi = process.env.COMMANDAGI_HOME || join(homedir(), 'Documents/commandagi');
const renderer = join(commandagi, 'packages/applets/video/eval');
const sdk = join(commandagi, 'sdk/python');

function ffmpegPath() {
  if (process.env.FFMPEG_PATH) return process.env.FFMPEG_PATH;
  const python = join(root, 'tensacode/python/.venv/bin/python');
  if (!existsSync(python)) throw new Error('set FFMPEG_PATH: no tensacode/python/.venv to find imageio-ffmpeg in');
  return execFileSync(python, ['-c', 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'], { encoding: 'utf8' }).trim();
}

/** The time of the edit's "poster" marker, read through the SDK. */
function posterTime(edit) {
  const program = [
    'import json, sys',
    'from commandagi.design import run_module',
    "g = run_module(open(sys.argv[1]).read(), sys.argv[1])['graph']",
    "marks = [m for n in g['nodes'].values() if n['type'] == 'video.composite' for m in n['inputs'].get('markers', [])]",
    "print(next((m['time'] for m in marks if m['name'] == 'poster'), 1.0))",
  ].join('\n');
  return execFileSync('python3', ['-c', program, edit], { encoding: 'utf8', env: { ...process.env, PYTHONPATH: sdk } }).trim();
}

if (!existsSync(join(renderer, 'node_modules'))) {
  throw new Error(`the renderer is not installed: run \`npm install\` in ${renderer}`);
}
const ffmpeg = ffmpegPath();
const only = process.argv[2];
const names = readdirSync(join(root, 'media')).filter((n) => statSync(join(root, 'media', n)).isDirectory() && (!only || n === only));
if (!names.length) throw new Error(only ? `no media/${only}/` : 'no edits under media/');

for (const name of names) {
  const dir = join(root, 'media', name);
  const edit = readdirSync(dir).find((f) => f.endsWith('.vid.py'));
  if (!edit) throw new Error(`media/${name}/ has no .vid.py`);
  const video = join(root, 'site/media', `${name}.mp4`);
  const poster = join(root, 'site/media', `${name}.jpg`);
  execFileSync('npx', ['tsx', 'src/cli.ts', join(dir, edit), '--out', video, '--quality', 'medium'], {
    cwd: renderer, stdio: 'inherit', env: { ...process.env, FFMPEG_PATH: ffmpeg },
  });
  execFileSync(ffmpeg, ['-v', 'error', '-y', '-ss', posterTime(join(dir, edit)), '-i', video, '-frames:v', '1', '-q:v', '4', poster]);
  console.log(`site/media/${name}.mp4 (${(statSync(video).size / 1e6).toFixed(1)} MB), site/media/${name}.jpg`);
}
