
def visualize_pr_heatmap_on_floorplan(
    pr_map,
    floorplan_path,
    transmitter=None,
    title="Power Received",
    heatmap_alpha=0.55,
    vmin=None,
    vmax=None,
    flip_floorplan=True,
):
    pr_map = np.asarray(pr_map, dtype=float)
    h, w = pr_map.shape

    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Floorplan
    ax.imshow(
        floorplan,
        origin="lower",
        extent=(0, w, 0, h)
    )

    # Heatmap
    im = ax.imshow(
        pr_map,
        origin="lower",
        extent=(0, w, 0, h),
        cmap="inferno",
        interpolation="nearest",
        alpha=heatmap_alpha,
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power received (dBm)")

    # TX marker
    if transmitter is not None:
        tx_x, tx_y = transmitter
        ax.scatter(tx_x, tx_y, s=200, marker="*", edgecolors="black", label="TX")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if transmitter is not None:
        ax.legend()

    plt.tight_layout()
    plt.show()

def visualize_snr_contours_on_floorplan(
    snr_map,
    floorplan_path,
    thresholds,
    transmitter=None,
    title="SNR Coverage Classes",
    show_labels=True,
    flip_floorplan=True,
):
    snr_map = np.asarray(snr_map, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim != 1 or thresholds.size != 2:
        raise ValueError("thresholds must contain exactly 2 values to create 3 classes")

    h, w = snr_map.shape

    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Floorplan
    ax.imshow(
        floorplan,
        origin="lower",
        extent=(0, w, 0, h)
    )

    # Contours at cell centers
    x = np.arange(w) + 0.5
    y = np.arange(h) + 0.5
    X, Y = np.meshgrid(x, y)

    cs = ax.contour(
        X, Y, snr_map,
        levels=thresholds,
        linewidths=2
    )

    if show_labels:
        fmt = {t: f"{t:g} dB" for t in thresholds}
        ax.clabel(cs, inline=True, fontsize=9, fmt=fmt)

    # TX marker
    if transmitter is not None:
        tx_x, tx_y = transmitter
        ax.scatter(tx_x + 0.5, tx_y + 0.5, s=200, marker="*", edgecolors="black", label="TX")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if transmitter is not None:
        ax.legend()

    plt.tight_layout()
    plt.show()

def select_tx_on_floorplan(
    floorplan_path,
    grid_shape,
    title="Click transmitter location",
    flip_floorplan=True,
):
    """
    Show the floorplan and let the user click once to select TX.

    Parameters
    ----------
    floorplan_path : str
        Path to the floorplan image.
    grid_shape : tuple[int, int]
        Shape of the simulation grid: (height, width).
    title : str
        Window title.
    flip_floorplan : bool
        Flip vertically to match origin='lower'.

    Returns
    -------
    tx : tuple[int, int]
        Transmitter position as (x, y) grid cell coordinates.
    """
    h, w = grid_shape

    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        floorplan,
        origin="lower",
        extent=(0, w, 0, h)
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    print("Click once to place the transmitter...")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        raise ValueError("No transmitter point was selected.")

    x, y = pts[0]

    # Convert click position to grid cell
    tx = (int(np.floor(x)), int(np.floor(y)))

    print(f"Selected TX: {tx}")
    return tx
