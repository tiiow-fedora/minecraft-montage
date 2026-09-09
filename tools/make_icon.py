"""Generate the app icon (a play triangle on a dark rounded square).

Drawn at 256px and downsampled into a multi-resolution .ico so it stays crisp
at desktop, taskbar and alt-tab sizes.
"""

from PIL import Image, ImageDraw

SIZE = 256
BG = (24, 26, 31, 255)
ACCENT = (45, 212, 191, 255)  # teal
RADIUS = 52


def main(out="icon.ico"):
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=RADIUS, fill=BG)

    # subtle inner border so it reads on both light and dark desktops
    d.rounded_rectangle([3, 3, SIZE - 4, SIZE - 4], radius=RADIUS - 3,
                        outline=(255, 255, 255, 28), width=2)

    # play triangle, optically centred (shifted right of true centre)
    cx, cy, r = SIZE * 0.54, SIZE * 0.5, SIZE * 0.20
    d.polygon(
        [(cx - r * 0.85, cy - r), (cx - r * 0.85, cy + r), (cx + r, cy)],
        fill=ACCENT,
    )

    # film-strip perforations down the left edge, to say "video"
    for i in range(4):
        y = SIZE * 0.30 + i * SIZE * 0.14
        d.rounded_rectangle(
            [SIZE * 0.16, y, SIZE * 0.16 + SIZE * 0.055, y + SIZE * 0.075],
            radius=6, fill=(255, 255, 255, 42),
        )

    img.save(out, sizes=[(256, 256), (128, 128), (64, 64), (48, 48),
                         (32, 32), (16, 16)])
    print("wrote", out)


if __name__ == "__main__":
    main()
