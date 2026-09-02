# Reinstalling the Python environment

XenoTagger needs a Python environment to run - either a **portable
environment** (a `system\python` folder inside the XenoTagger folder,
created by `setup-portable.bat`) or a **venv** (a `.venv` folder inside
the XenoTagger folder, created by `setup-venv.bat`). You have exactly
one of the two, never both.

If that folder is deleted, moved, or becomes corrupted, `run.cmd` and
`cli.cmd` will show:

```
No environment found. Run setup-portable.bat or setup-venv.bat first.
```

This guide walks through getting a working environment back. It does
not affect your images, captions, settings, or installed AI models -
see "What is not affected" below.

## Step 1: Find out which setup you have

1. Open the main XenoTagger folder in File Explorer.
2. Look for a folder named `system`. Open it - if it contains a folder
   named `python`, you are using the **portable environment**. Go to
   Step 2A.
3. If there is no `system\python` folder, look instead for a folder
   named `.venv` directly inside the main XenoTagger folder. If you
   find it, you are using **venv**. Go to Step 2B.
4. If you find neither folder, you have not completed setup at all.
   Follow the installation instructions in the main README instead of
   this guide.

## Step 2A: Recover the portable environment

1. Open the main XenoTagger folder in File Explorer.
2. Double-click **setup-portable.bat**.
3. A black command-window will open and print progress messages. This
   step downloads and reinstalls the Python runtime, and can take
   several minutes depending on your internet connection. Do not close
   the window while it is running.
4. When it finishes, it prints `Done.` and waits for a key press. Press
   any key to close the window.
5. Continue to Step 3.

If the window closes immediately or prints `Setup failed.`, check that
you are connected to the internet and try again.

## Step 2B: Recover the venv environment

1. Open the main XenoTagger folder in File Explorer.
2. Double-click **setup-venv.bat**.
3. A black command-window will open and print progress messages. This
   step recreates the environment and reinstalls its packages, and can
   take a few minutes. Do not close the window while it is running.
4. When it finishes, it prints `Done.` and waits for a key press. Press
   any key to close the window.
5. Continue to Step 3.

If the window prints `No systemwide "python" found on PATH.`, Python
itself is not installed on this computer outside of XenoTagger. Either
install Python from [python.org](https://python.org) first and run
`setup-venv.bat` again, or use `setup-portable.bat` instead, which does
not require anything to be installed system-wide.

If the window prints `Setup failed.` for any other reason, check that
you are connected to the internet and try again.

## Step 3: Launch XenoTagger

Double-click **run.cmd** (for the graphical interface) or **cli.cmd**
(for the command-line interface). Either one should now start normally.

## Step 4: Reinstall image tagging support, if you use it

If you use the optional Hydra tag classifier (Settings tab → Hydra
sub-tab), its dependencies are stored inside the environment folder you
just recreated, so they need to be reinstalled separately:

1. Launch XenoTagger (Step 3).
2. Go to the **Settings** tab, then the **Hydra** sub-tab.
3. If it shows "Dependencies: not installed", click **Install Hydra
   dependencies** and wait for it to finish.
4. The Hydra model file itself does not need to be re-downloaded - it
   is stored outside the environment folder and was not affected.

If you do not use Hydra tagging, you can skip this step entirely -
regular captioning works without it.

## What is not affected

Recreating the environment only replaces the Python runtime and its
installed packages. The following are stored elsewhere and survive
untouched:

- Your images and caption files, wherever they are stored.
- All settings (Settings tab, including server and captioning options).
- Downloaded AI vision models.
- The installed llama.cpp server.
- The downloaded Hydra model file (only its supporting packages need
  reinstalling, per Step 4).
