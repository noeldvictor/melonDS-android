# melonDS for AYN Thor — HD edition

A dual-screen Android fork of [melonDS](https://melonds.kuribo64.net/) tuned for the AYN Thor,
with HD texture pack support and per-layer upscaling filters.

Based on [SapphireRhodonite's Vulkan dual-screen fork](https://github.com/SapphireRhodonite/melonDS-android)
of [rafaelvcaetano's melonDS Android port](https://github.com/rafaelvcaetano/melonDS-android).
This repository is self-contained: the emulator core (`melonDS-android-lib/`, derived from the
[melonDS](https://github.com/melonDS-emu/melonDS) core) is part of the tree — no submodules.

## What this fork adds

### HD texture packs
* **3D texture dump & replace** — content-hash keyed (texture hash + palette hash), compatible
  with the desktop melonDS HD pack format, so packs can be authored and verified on PC and used
  on device unchanged.
* **2D sprite (OBJ) and BG tile dump & replace** — sprites and background tiles are dumped as
  assembled art and can be replaced with scaled versions (up to 4x).
* Packs live in `files/texturepacks/<GAMECODE>/`; dumps are written to `files/texturedumps/`.

### Per-layer upscaling filters
* Independent filter selection for **3D textures**, **OBJ sprites**, and **BG layers**
  (Settings → Video): nearest, bilinear, and a set of pixel-art filters including xBR, Eagle,
  SaI, and a faithful multi-pass **ScaleFX** port.
* Filters run in the Vulkan compositor as a cached pre-pass — static scenes cost nothing
  (content-hashed reuse), and 3D texture filtering is cached per texture at upload.
* No full-screen smoothing: original pixels stay sharp unless a layer's filter says otherwise.

### Dual-screen and stability work
* Fixes for dual-display presentation on the Thor's two panels (screen-swap alternation,
  capture-backed scenes, frame pacing under load), verified with per-display captures.
* Renderer thread-safety and Vulkan lifecycle fixes throughout the compositor and presenter.
* Optimized native build settings for smooth performance at 4x internal resolution.

## Building

Requirements: JDK 21, Android NDK 28.x, CMake 3.22+, Rust (for librashader), Git Bash on Windows.

```
git clone --recurse-submodules https://github.com/noeldvictor/melonDS-android.git
cd melonDS-android
./gradlew :app:assembleGitHubProdDebug
```

The emulator core is part of this repository; the remaining submodules are third-party
libraries (oboe, faad2, enet).

The APK lands in `app/build/outputs/apk/gitHubProd/debug/`. On Windows, set `JAVA_HOME`,
`ANDROID_NDK_HOME`, and ensure `cargo` is on the path (or set `CARGO`).

## Credits

* [melonDS](https://github.com/melonDS-emu/melonDS) by Arisotura and the melonDS team
* [melonDS Android port](https://github.com/rafaelvcaetano/melonDS-android) by rafaelvcaetano
* [Vulkan dual-screen fork](https://github.com/SapphireRhodonite/melonDS-android) by SapphireRhodonite
* HD pack format inspired by the texture replacement systems of Dolphin and DuckStation

melonDS is free software licensed under the GPLv3; this fork retains that license.
