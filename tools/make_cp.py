"""Build a Java @argfile classpath for the DumpTracks tool.

ReplayStudio inside the mod jar is not self-contained: gson, commons-lang3 and
guava are all expected from Minecraft's runtime, and opennbt et al are jarjar'd
inside the mod under META-INF/jars/. This gathers all three sources.
"""

import pathlib
import sys

PROFILE = pathlib.Path.home() / "AppData/Roaming/ModrinthApp/profiles/HarrisonsLongHaulWorld"
LIBRARIES = pathlib.Path.home() / "AppData/Roaming/ModrinthApp/meta/libraries"
MOD_JAR = PROFILE / "mods" / "reforgedplaymod-1.21.1-0.3.jar"


def main(out_path="build/cp.txt", nested_dir="build/libs", classes_dir="build"):
    entries = [MOD_JAR]
    entries += sorted(pathlib.Path(nested_dir).glob("*.jar"))
    entries += sorted(LIBRARIES.rglob("*.jar"))
    entries.append(pathlib.Path(classes_dir))

    cp = ";".join(str(p).replace("\\", "/") for p in entries)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('-cp "' + cp + '"\n', encoding="utf-8")
    print("classpath entries: %d -> %s" % (len(entries), out))


if __name__ == "__main__":
    main(*sys.argv[1:])
