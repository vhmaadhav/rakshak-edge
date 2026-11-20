#!/usr/bin/env bash
# Headless build of the Rakshak Edge app on Debian 13 x86_64: JDK 17, Android command-line tools,
# the platform and build-tools packages only (no emulator, no system image, no NDK), then
#   ./gradlew :app:assembleDebug :app:testDebugUnitTest --no-daemon
#
# Disk footprint (estimated, not measured on this VM):
#   JDK 17 headless ~200 MB | cmdline-tools ~150 MB | platform-tools ~80 MB
#   platforms;android-37.2 ~80 MB | build-tools;37.0.0 ~60 MB | Gradle 9.8.0 ~150 MB
#   Gradle caches (AGP, Kotlin, Compose, test deps) 0.7-1.2 GB
#   ExecuTorch AAR 100-200 MB | ONNX Runtime AAR 50-80 MB
#   Total roughly 1.6-2.3 GB. Set GRADLE_USER_HOME to a disk with room if needed.
#
# Inputs:
#   CMDLINE_TOOLS_ZIP  path to commandlinetools-linux-*_latest.zip from developer.android.com
#                      (the download page gives the current build number; it is not guessed here)
#   ANDROID_HOME       default $HOME/android-sdk
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ANDROID_HOME="${ANDROID_HOME:-$HOME/android-sdk}"
export ANDROID_HOME

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1" >&2; exit 1; }; }

need java
JAVA_MAJOR="$(java -version 2>&1 | sed -n 's/.*version "\([0-9]*\).*/\1/p' | head -n1)"
if [ "${JAVA_MAJOR:-0}" -lt 17 ]; then
  echo "JDK 17 or newer is required (found ${JAVA_MAJOR:-none}). On Debian: sudo apt-get install openjdk-17-jdk-headless" >&2
  exit 1
fi

if [ ! -x "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" ]; then
  if [ -z "${CMDLINE_TOOLS_ZIP:-}" ] || [ ! -f "$CMDLINE_TOOLS_ZIP" ]; then
    echo "Set CMDLINE_TOOLS_ZIP to the command-line tools zip (see the comment at the top)." >&2
    exit 1
  fi
  mkdir -p "$ANDROID_HOME/cmdline-tools"
  unzip -q "$CMDLINE_TOOLS_ZIP" -d "$ANDROID_HOME/cmdline-tools"
  mv "$ANDROID_HOME/cmdline-tools/cmdline-tools" "$ANDROID_HOME/cmdline-tools/latest"
fi
SDKMANAGER="$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager"

# `yes` dies of SIGPIPE when sdkmanager exits; with pipefail that would fail the script (rc 141).
set +o pipefail
yes | "$SDKMANAGER" --licenses >/dev/null
set -o pipefail
"$SDKMANAGER" "platform-tools" "platforms;android-37.2" "build-tools;37.0.0"

printf 'sdk.dir=%s\n' "$ANDROID_HOME" > "$ROOT/local.properties"

if [ ! -x "$ROOT/gradlew" ]; then
  echo "gradlew is missing. Create the wrapper once: gradle wrapper --gradle-version 9.8.0 (see android/README.md)." >&2
  exit 1
fi
if [ ! -f "$ROOT/app/src/main/assets/sms_student.onnx" ]; then
  echo "note: app/src/main/assets/sms_student.onnx is not present. The build still compiles; the app" \
       "reports the missing detector at run time." >&2
fi

cd "$ROOT"
./gradlew :app:assembleDebug :app:testDebugUnitTest --no-daemon
echo "APK: $ROOT/app/build/outputs/apk/debug/app-debug.apk"
