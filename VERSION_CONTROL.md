# Version control (cfd-scientist + Foam-Agent submodule)

This repo includes **Foam-Agent** as a **git submodule** at `Foam-Agent/`.

## Clone / first-time setup (most users)
The submodule URL is stored in this repo’s committed `.gitmodules`, so most users only need:
```bash
git clone <cfd-scientist-repo-url>
cd cfd-scientist
git submodule update --init --recursive
```

## Optional: use a fork or HTTPS for the submodule
Only do this if you want a different Foam-Agent remote than the default.
```bash
cd cfd-scientist/Foam-Agent
git remote set-url origin <https-or-fork-url>
```

## Day-to-day: commit changes to cfd-scientist
Use this when you edited files outside `Foam-Agent/`.
```bash
cd cfd-scientist
git status
git add -A
git commit -m "<message>"
```

## Update Foam-Agent to a newer upstream version
This moves the submodule to a newer commit, then records that new pointer in cfd-scientist.
```bash
cd cfd-scientist/Foam-Agent
git fetch
git checkout main    # or the branch/tag you want
git pull

cd ..
git add Foam-Agent .gitmodules
git commit -m "Bump Foam-Agent submodule"
```

## If you edited Foam-Agent code
You must commit **inside the submodule** first, then commit the updated submodule pointer in cfd-scientist.

```bash
cd cfd-scientist/Foam-Agent
git add -A
git commit -m "<message>"

cd ..
git add Foam-Agent
git commit -m "Update Foam-Agent submodule pointer"
```

## Common pitfall
- Seeing `Foam-Agent` changed in `git status` at the top level means: the **submodule commit pointer changed**.
  Commit that change in the parent repo after you update the submodule.
