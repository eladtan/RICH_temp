import sys
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    """Densmore 2012 heterogeneous MC: both MPI (no RW) and serial (RW) vs reference."""
    profile_mpi = "desmore2012_mc_profile.txt"
    ref_file = "data/densmore2012_fig4_mc.csv"

    if not os.path.exists(profile_mpi):
        print(f"  [desmore2012_mc] no profile found for either MPI or serial variant")
        sys.exit(1)

    keV_K = 1.602176634e-9 / 1.380649e-16

    fig, ax = plt.subplots(figsize=(8, 5))

    if os.path.exists(ref_file):
        ref = np.loadtxt(str(ref_file), delimiter=",", comments="#")
        ax.plot(ref[:, 0], ref[:, 1], "b-", linewidth=1.5,
                label="Densmore 2012 Fig.\u20094 (MC)")

    if os.path.exists(profile_mpi):
        raw = np.loadtxt(str(profile_mpi))
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        ax.plot(raw[:, 0], raw[:, 1] / keV_K, "ko", markersize=3,
                markerfacecolor="none", label="RICH MC (MPI, no RW)")

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Material Temperature [keV]")
    ax.set_title("Densmore 2012 Heterogeneous Step-Opacity -- MC IMC")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("desmore2012_mc.png", dpi=200)
    fig.savefig("desmore2012_mc.pdf")
    plt.show()
    plt.close(fig)
    print(f"Saved desmore2012_mc.png/pdf")
