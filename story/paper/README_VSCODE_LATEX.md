# VSCode LaTeX Local Build (Overleaf Sync)

## 1) What is configured

- VSCode uses `LaTeX Workshop` + local `tectonic`.
- Compiler path:
  - `/data/YBJ/cleansight/.conda/envs/latex/bin/tectonic`
- Local bundle path:
  - `/data/YBJ/cleansight/.cache/tectonic/bundle-v33`
- Build output directory:
  - `build/` (next to the root `.tex` file)

## 2) Use in VSCode

1. Install extension: `James-Yu.latex-workshop` (VSCode will recommend it automatically).
2. Open a root tex file, for example:
   - `story/paper/69df39dafea55d39f988a486/neurips_2026.tex`
3. Build:
   - `Ctrl+Alt+B` (or Command Palette: `LaTeX Workshop: Build LaTeX project`)
4. PDF preview:
   - opens in VSCode tab, with SyncTeX enabled.

## 3) CLI build command (same toolchain)

```bash
cd /data/YBJ/cleansight/story/paper/69df39dafea55d39f988a486
XDG_CACHE_HOME=/data/YBJ/cleansight/.cache \
/data/YBJ/cleansight/.conda/envs/latex/bin/tectonic \
  --bundle /data/YBJ/cleansight/.cache/tectonic/bundle-v33 \
  --synctex --keep-logs --keep-intermediates \
  --outdir build neurips_2026.tex
```

## 4) Push back to Overleaf (Git)

Overleaf project page provides a Git URL like:

```text
https://git.overleaf.com/<project_id>
```

One-time setup:

```bash
cd /data/YBJ/cleansight/story/paper/69df39dafea55d39f988a486
git remote add overleaf https://git.overleaf.com/<project_id>
```

Daily sync:

```bash
git add .
git commit -m "update paper"
git push overleaf HEAD:master
```

If your Overleaf project default branch is not `master`, replace branch name accordingly.
