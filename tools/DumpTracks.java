// Dump every entity's position track from a .mcpr as JSON.
//
// Rather than reimplement Minecraft packet parsing, this drives ReplayStudio's
// own EntityPositionTracker -- the same class ReplayMod uses to make spectator
// keyframes work -- straight out of the ReForgedPlay jar. Whatever the mod
// believes about entity positions is exactly what comes out here.
//
// Compile and run with the mod jar on the classpath:
//   javac -cp <modjar> -d build tools/DumpTracks.java
//   java  -cp <modjar>;build DumpTracks <replay.mcpr> <out.json>
//
// Always point it at a COPY: EntityPositionTracker writes a cache entry back
// into the replay file it is given.

import com.replaymod.replaystudio.Studio;
import com.replaymod.replaystudio.filter.StreamFilter;
import com.replaymod.replaystudio.replay.ZipReplayFile;
import com.replaymod.replaystudio.util.EntityPositionTracker;
import com.replaymod.replaystudio.util.Location;

import java.io.File;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.NavigableMap;

public class DumpTracks {

    /** Minimal Studio: EntityPositionTracker only needs it to open the file. */
    static class Stub implements Studio {
        public String getName() { return "mcmontage"; }
        public int getVersion() { return 1; }
        public StreamFilter loadStreamFilter(String name) { return null; }
        public boolean isCompatible(int fileVersion, int a, int b) { return true; }
        public int getCurrentFileFormatVersion() { return 14; }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: DumpTracks <replay.mcpr> <out.json>");
            System.exit(2);
        }
        File replay = new File(args[0]);
        ZipReplayFile file = new ZipReplayFile(new Stub(), replay);

        EntityPositionTracker tracker = new EntityPositionTracker(file);
        tracker.load(progress -> { });

        Field field = EntityPositionTracker.class.getDeclaredField("entityPositions");
        field.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<Integer, NavigableMap<Long, Location>> positions =
                (Map<Integer, NavigableMap<Long, Location>>) field.get(tracker);

        try (PrintWriter out = new PrintWriter(args[1], StandardCharsets.UTF_8)) {
            out.print("{");
            boolean firstEntity = true;
            for (Map.Entry<Integer, NavigableMap<Long, Location>> e : positions.entrySet()) {
                if (!firstEntity) out.print(",");
                firstEntity = false;
                out.print("\"" + e.getKey() + "\":[");
                boolean firstPoint = true;
                for (Map.Entry<Long, Location> p : e.getValue().entrySet()) {
                    if (!firstPoint) out.print(",");
                    firstPoint = false;
                    Location l = p.getValue();
                    out.print("[" + p.getKey() + ","
                            + l.getX() + "," + l.getY() + "," + l.getZ() + ","
                            + l.getYaw() + "," + l.getPitch() + "]");
                }
                out.print("]");
            }
            out.print("}");
        }

        System.err.println("entities: " + positions.size());
        file.close();
    }
}
